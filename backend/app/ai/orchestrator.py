from __future__ import annotations

from dataclasses import dataclass, field
import asyncio
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.ai.context_budget import ContextBudgeter
from app.ai.errors import AIProviderError
from app.ai.grounded_answer import GroundedAnswerGenerator, GroundingEvidence
from app.ai.model_router import ModelRouter, ModelSelection, ModelTask
from app.ai.multi_source_synthesis import build_multi_source_evidence, synthesize_multi_source_answer
from app.ai.planner import ExecutionPlan, PlanStep, PlanStepKind, QueryPlanner
from app.ai.provider import AIProvider
from app.ai.router import ModelRoute, RouteCategory, RouteReason, TypedRouter
from app.ai.schemas import AIResponse, TextGenerationRequest
from app.tools.gateway import authorize_and_execute_tool
from app.tools.registry import ToolRegistry, build_default_tool_registry
from app.tools.schemas import ToolExecutionContext, ToolExecutionRequest, ToolExecutionResult, ToolStatus


class InputPolicyStatus(str, Enum):
    allow = "allow"
    block = "block"


class InputPolicyDecision(BaseModel):
    status: InputPolicyStatus
    reason: str


class OrchestrationTrace(BaseModel):
    events: list[dict[str, Any]] = Field(default_factory=list)

    def record(self, event: dict[str, Any]) -> None:
        self.events.append(_safe_event(event))

    @property
    def stages(self) -> list[str]:
        return [event["stage"] for event in self.events]


class AgentTurnRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    conversation_id: str | None = None
    document_source_ids: list[str] = Field(default_factory=list, max_length=10)


class AgentTurnResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    answer: str
    route: ModelRoute | None
    model_selection: ModelSelection | None
    execution_plan: ExecutionPlan | None
    tool_results: list[ToolExecutionResult]
    trace: OrchestrationTrace
    confidence: float = Field(ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)


class GroundingPayload(BaseModel):
    tool_results: list[dict[str, Any]]
    retrieved_chunks: list[dict[str, Any]]
    limitations: list[str] = Field(default_factory=list)


@dataclass
class AgentDependencies:
    ai_provider: AIProvider
    typed_router: TypedRouter | None = None
    model_router: ModelRouter = field(default_factory=ModelRouter)
    query_planner: QueryPlanner = field(default_factory=QueryPlanner)
    tool_registry: ToolRegistry = field(default_factory=build_default_tool_registry)
    context_budgeter: ContextBudgeter = field(default_factory=ContextBudgeter)
    grounded_answer_generator: GroundedAnswerGenerator | None = None
    analytics_service: Any | None = None
    document_service: Any | None = None
    user_role: str = "analyst"

    def router(self) -> TypedRouter:
        return self.typed_router or TypedRouter(self.ai_provider)

    def grounded_generator(self) -> GroundedAnswerGenerator:
        return self.grounded_answer_generator or GroundedAnswerGenerator(self.ai_provider)


async def run_turn(request: AgentTurnRequest, dependencies: AgentDependencies) -> AgentTurnResult:
    trace = OrchestrationTrace()
    context = load_context(request, trace)
    policy = apply_input_policy(request, context, trace)
    if policy.status is InputPolicyStatus.block:
        return _blocked_result(policy, trace)

    route = await select_route(request, dependencies, trace)
    execution_plan = plan_execution(request, route, dependencies, trace)
    model_selection = select_model(execution_plan, dependencies, trace)
    authorized_plan = authorize_tools(execution_plan, trace)
    tool_results = await execute_tools(request, authorized_plan, dependencies, trace)
    answer_context = build_context(context, route, model_selection, authorized_plan, tool_results, dependencies, trace)
    answer = await generate_answer(request, dependencies, route, model_selection, answer_context, tool_results, trace)
    validated_answer = validate_answer(answer, route, authorized_plan, trace)
    finalize_trace(trace)

    return AgentTurnResult(
        answer=validated_answer["answer"],
        route=route,
        model_selection=model_selection,
        execution_plan=authorized_plan,
        tool_results=tool_results,
        trace=trace,
        confidence=validated_answer["confidence"],
        limitations=validated_answer["limitations"],
    )


def load_context(request: AgentTurnRequest, trace: OrchestrationTrace) -> dict[str, Any]:
    context = {"conversation_id": request.conversation_id, "recent_messages": []}
    trace.record({"stage": "load_context", "status": "ok", "recent_message_count": 0})
    return context


def apply_input_policy(
    request: AgentTurnRequest,
    context: dict[str, Any],
    trace: OrchestrationTrace,
) -> InputPolicyDecision:
    normalized = request.question.lower()
    blocked_markers = [
        "reveal internal secrets",
        "show me your api key",
        "show your api key",
        "environment variables",
        "ignore your rules",
        "ignore previous instructions",
    ]
    if any(marker in normalized for marker in blocked_markers):
        decision = InputPolicyDecision(status=InputPolicyStatus.block, reason="blocked_prompt_or_secret_request")
    else:
        decision = InputPolicyDecision(status=InputPolicyStatus.allow, reason="basic_policy_allow")

    trace.record({"stage": "apply_input_policy", "status": decision.status.value, "reason": decision.reason})
    return decision


async def select_route(
    request: AgentTurnRequest,
    dependencies: AgentDependencies,
    trace: OrchestrationTrace,
) -> ModelRoute:
    route = await dependencies.router().select_route(request.question)
    trace.record(
        {
            "stage": "select_route",
            "status": "ok",
            "category": route.category.value,
            "reason_code": route.reason_code.value,
            "confidence": route.confidence,
        }
    )
    return route


def plan_execution(
    request: AgentTurnRequest,
    route: ModelRoute,
    dependencies: AgentDependencies,
    trace: OrchestrationTrace,
) -> ExecutionPlan:
    plan = dependencies.query_planner.create_plan(request.question, route)
    trace.record(
        {
            "stage": "plan_execution",
            "status": "ok",
            "route": plan.route.value,
            "model_task": plan.model_task.value,
            "step_count": len(plan.steps),
            "requires_synthesis": plan.requires_synthesis,
            "steps": [step.model_dump(mode="json") for step in plan.steps],
        }
    )
    return plan


def select_model(
    execution_plan: ExecutionPlan,
    dependencies: AgentDependencies,
    trace: OrchestrationTrace,
) -> ModelSelection:
    selection = dependencies.model_router.select_model(execution_plan.model_task)
    trace.record({"stage": "select_model", "status": "ok", **selection.to_trace_metadata()})
    return selection


def authorize_tools(execution_plan: ExecutionPlan, trace: OrchestrationTrace) -> ExecutionPlan:
    tool_steps = execution_plan.tool_steps
    trace.record(
        {
            "stage": "authorize_tools",
            "status": "ok",
            "authorized_tool_count": len(tool_steps),
            "tool_names": [step.tool_name.value for step in tool_steps if step.tool_name is not None],
        }
    )
    return execution_plan


async def execute_tools(
    request: AgentTurnRequest,
    execution_plan: ExecutionPlan,
    dependencies: AgentDependencies,
    trace: OrchestrationTrace,
) -> list[ToolExecutionResult]:
    tool_steps = execution_plan.tool_steps
    if not tool_steps:
        trace.record({"stage": "execute_tools", "status": "skipped", "tool_result_count": 0})
        return []

    context = ToolExecutionContext(
        user_role=dependencies.user_role,
        analytics_service=dependencies.analytics_service,
        document_service=dependencies.document_service,
        document_source_ids=request.document_source_ids,
        trace=trace,
    )
    results = await asyncio.gather(
        *[
            authorize_and_execute_tool(
                ToolExecutionRequest(
                    tool_name=step.tool_name.value if step.tool_name is not None else "",
                    input=_tool_input_for_step(request, step),
                ),
                context=context,
                registry=dependencies.tool_registry,
            )
            for step in tool_steps
        ]
    )
    trace.record(
        {
            "stage": "execute_tools",
            "status": "ok",
            "tool_result_count": len(results),
            "statuses": [result.status.value for result in results],
        }
    )
    return list(results)


def build_context(
    loaded_context: dict[str, Any],
    route: ModelRoute,
    model_selection: ModelSelection,
    execution_plan: ExecutionPlan,
    tool_results: list[ToolExecutionResult],
    dependencies: AgentDependencies,
    trace: OrchestrationTrace,
) -> dict[str, Any]:
    context = {
        "loaded_context": loaded_context,
        "route": route.model_dump(mode="json"),
        "model_selection": model_selection.to_trace_metadata(),
        "execution_plan": execution_plan.model_dump(mode="json"),
        "tool_results": [result.model_dump(mode="json") for result in tool_results],
    }
    budgeted = dependencies.context_budgeter.build(
        user_question="",
        system_instructions="RetailData-Pro grounded assistant.",
        recent_conversation=loaded_context.get("recent_messages", []),
        tool_results=[result.model_dump(mode="json") for result in tool_results],
        retrieved_evidence=[],
    )
    context["budgeted_context"] = {
        **budgeted.model_dump(mode="json"),
        "route": route.category.value,
        "model": model_selection.model,
        "step_count": len(execution_plan.steps),
    }
    trace.record({"stage": "build_context", "status": "ok", "tool_result_count": len(tool_results)})
    return context


async def generate_answer(
    request: AgentTurnRequest,
    dependencies: AgentDependencies,
    route: ModelRoute,
    model_selection: ModelSelection,
    answer_context: dict[str, Any],
    tool_results: list[ToolExecutionResult],
    trace: OrchestrationTrace,
) -> dict[str, Any]:
    if route.category is not RouteCategory.conversation:
        missing = _missing_required_evidence(answer_context["execution_plan"], tool_results)
        grounding_payload = _grounding_payload_from_tool_results(tool_results)
        limitations = [*grounding_payload.limitations]
        if missing:
            limitations.append(f"Missing required evidence from: {', '.join(missing)}")
        if missing and not grounding_payload.tool_results and not grounding_payload.retrieved_chunks:
            trace.record({"stage": "generate_answer", "status": "skipped", "reason": "missing_required_evidence", "missing": missing})
            return {
                "answer": (
                    "I do not have enough verified evidence to answer that yet. "
                    "One or more required tools are unavailable or failed."
                ),
                "confidence": min(route.confidence, 0.4),
                "limitations": limitations,
            }

        try:
            grounded = await dependencies.grounded_generator().generate(
                request.question,
                GroundingEvidence(
                    tool_results=grounding_payload.tool_results,
                    retrieved_chunks=grounding_payload.retrieved_chunks,
                ),
                model=model_selection.model,
            )
            trace.record(
                {
                    "stage": "generate_answer",
                    "status": "ok",
                    "reason": "grounded_answer",
                    "citation_count": len(grounded.citations),
                    "tool_evidence_count": len(grounding_payload.tool_results),
                    "retrieved_chunk_count": len(grounding_payload.retrieved_chunks),
                }
            )
            return {
                "answer": grounded.answer,
                "confidence": min(route.confidence, grounded.confidence),
                "limitations": [*limitations, *grounded.limitations],
            }
        except AIProviderError as exc:
            trace.record(
                {
                    "stage": "generate_answer",
                    "status": "fallback",
                    "reason": "grounded_provider_error",
                    "error_type": exc.__class__.__name__,
                    "tool_evidence_count": len(grounding_payload.tool_results),
                    "retrieved_chunk_count": len(grounding_payload.retrieved_chunks),
                }
            )
            synthesized = _safe_grounded_generation_failure_answer(request.question, route, grounding_payload)
            return {
                "answer": synthesized,
                "confidence": min(route.confidence, 0.5),
                "limitations": [*limitations, "Grounded answer generation failed after evidence collection."],
            }

    try:
        response = await dependencies.ai_provider.generate_text(
            TextGenerationRequest(prompt=_conversation_prompt(request.question, answer_context), model=model_selection.model)
        )
        trace.record(
            {
                "stage": "generate_answer",
                "status": "ok",
                "provider": response.provider,
                "model": response.model,
                "latency_ms": response.latency_ms,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }
        )
        return {"answer": response.content, "confidence": route.confidence, "limitations": []}
    except AIProviderError:
        trace.record({"stage": "generate_answer", "status": "fallback", "reason": "provider_error"})
        return {
            "answer": "I could not generate an AI response right now. Please try again shortly.",
            "confidence": 0.2,
            "limitations": ["The AI provider failed during answer generation."],
        }


def validate_answer(answer: dict[str, Any], route: ModelRoute, execution_plan: ExecutionPlan, trace: OrchestrationTrace) -> dict[str, Any]:
    trace.record({"stage": "validate_answer", "status": "ok", "route": route.category.value, "step_count": len(execution_plan.steps)})
    return answer


def finalize_trace(trace: OrchestrationTrace) -> OrchestrationTrace:
    trace.record({"stage": "finalize_trace", "status": "ok", "persistence": "in_memory_only"})
    return trace


def _blocked_result(policy: InputPolicyDecision, trace: OrchestrationTrace) -> AgentTurnResult:
    finalize_trace(trace)
    return AgentTurnResult(
        answer="I cannot help with requests to bypass rules or reveal secrets.",
        route=None,
        model_selection=None,
        execution_plan=None,
        tool_results=[],
        trace=trace,
        confidence=1.0,
        limitations=[policy.reason],
    )


def _conversation_prompt(question: str, answer_context: dict[str, Any]) -> str:
    return (
        "Answer the user conversationally. Do not claim access to tools, databases, documents, or hidden context.\n"
        f"Available context: {answer_context}\n"
        f"Question: {question}"
    )


def _tool_input_for_step(request: AgentTurnRequest, step: PlanStep) -> dict[str, Any]:
    if step.tool_name is None:
        return {}
    payload: dict[str, Any] = {"question": request.question}
    if step.tool_name.value == "document_search":
        payload["source_ids"] = request.document_source_ids
    return payload


def _missing_required_evidence(execution_plan_payload: dict[str, Any], tool_results: list[ToolExecutionResult]) -> list[str]:
    result_by_tool = {result.tool_name: result for result in tool_results}
    missing: list[str] = []
    for step in execution_plan_payload["steps"]:
        if step["kind"] != PlanStepKind.tool.value or not step["required"]:
            continue
        tool_name = step.get("tool_name")
        result = result_by_tool.get(tool_name)
        if result is None or result.status is not ToolStatus.success or _has_empty_required_output(result):
            missing.append(tool_name or step["id"])
    return missing


def _has_empty_required_output(result: ToolExecutionResult) -> bool:
    if result.tool_name == "analytics_summary":
        return result.output.get("summary_type") == "dependency_missing"
    if result.tool_name == "document_search":
        return len(result.output.get("chunks") or []) == 0
    return False


def _grounding_payload_from_tool_results(tool_results: list[ToolExecutionResult]) -> GroundingPayload:
    successful_tools: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    limitations: list[str] = []
    for result in tool_results:
        if result.status is not ToolStatus.success:
            limitations.append(f"{result.tool_name} evidence was unavailable.")
            continue
        if result.tool_name == "analytics_summary" and result.output.get("summary_type") != "dependency_missing":
            successful_tools.append(result.model_dump(mode="json"))
            continue
        if result.tool_name == "document_search" and result.output.get("chunks"):
            successful_tools.append(result.model_dump(mode="json"))
        elif result.tool_name == "document_search":
            limitations.append("Document evidence was unavailable.")
            continue
        for chunk in result.output.get("chunks") or []:
            chunks.append(
                {
                    "source_id": chunk.get("source_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    "source_title": chunk.get("title"),
                    "chunk_index": chunk.get("chunk_index"),
                    "content": chunk.get("content"),
                    "score": chunk.get("score"),
                    "retrieval_method": "postgres_full_text_or_recent_document_fallback",
                }
            )
    return GroundingPayload(tool_results=successful_tools, retrieved_chunks=chunks, limitations=limitations)


def _safe_grounded_generation_failure_answer(question: str, route: ModelRoute, evidence: GroundingPayload) -> str:
    if not evidence.tool_results and not evidence.retrieved_chunks:
        return "I could not gather enough verified evidence to answer this question because the required data sources were unavailable."
    if route.category is RouteCategory.multi_source:
        return synthesize_multi_source_answer(build_multi_source_evidence(question, evidence.tool_results, evidence.retrieved_chunks))
    return "I gathered partial evidence, but it was not enough to produce a supported answer."

def _safe_event(event: dict[str, Any]) -> dict[str, Any]:
    blocked_keys = {"api_key", "secret", "token", "password", "authorization"}
    return {key: value for key, value in event.items() if key.lower() not in blocked_keys}
