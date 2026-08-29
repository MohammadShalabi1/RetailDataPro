from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.ai.errors import AIProviderError
from app.ai.model_router import ModelRouter, ModelSelection, ModelTask
from app.ai.provider import AIProvider
from app.ai.router import ModelRoute, RouteCategory, RouteReason, TypedRouter
from app.ai.schemas import AIResponse, TextGenerationRequest

MAX_PLAN_STEPS = 3


class InputPolicyStatus(str, Enum):
    allow = "allow"
    block = "block"


class PlanStep(BaseModel):
    name: str
    goal: str
    requires_tool: bool = False


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    route: RouteCategory
    model_task: ModelTask
    steps: list[PlanStep] = Field(max_length=MAX_PLAN_STEPS)


class ToolExecutionResult(BaseModel):
    tool_name: str
    status: str
    output: dict[str, Any] = Field(default_factory=dict)


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


@dataclass
class AgentDependencies:
    ai_provider: AIProvider
    typed_router: TypedRouter | None = None
    model_router: ModelRouter = field(default_factory=ModelRouter)

    def router(self) -> TypedRouter:
        return self.typed_router or TypedRouter(self.ai_provider)


async def run_turn(request: AgentTurnRequest, dependencies: AgentDependencies) -> AgentTurnResult:
    trace = OrchestrationTrace()
    context = load_context(request, trace)
    policy = apply_input_policy(request, context, trace)
    if policy.status is InputPolicyStatus.block:
        return _blocked_result(policy, trace)

    route = await select_route(request, dependencies, trace)
    execution_plan = plan_execution(route, trace)
    model_selection = select_model(execution_plan, dependencies, trace)
    authorized_plan = authorize_tools(execution_plan, trace)
    tool_results = await execute_tools(authorized_plan, trace)
    answer_context = build_context(context, route, model_selection, authorized_plan, tool_results, trace)
    answer = await generate_answer(request, dependencies, route, model_selection, answer_context, trace)
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


def plan_execution(route: ModelRoute, trace: OrchestrationTrace) -> ExecutionPlan:
    if route.category is RouteCategory.conversation:
        model_task = ModelTask.analytics_answer
        steps = [PlanStep(name="answer_conversation", goal="Answer the user conversationally.")]
    elif route.category is RouteCategory.multi_source:
        model_task = ModelTask.multi_source_synthesis
        steps = [
            PlanStep(name="await_tool_gateway", goal="Use multiple sources after Phase 8 tools exist.", requires_tool=True)
        ]
    elif route.category is RouteCategory.retail_analytics:
        model_task = ModelTask.analytics_answer
        steps = [PlanStep(name="await_analytics_tool", goal="Use deterministic analytics after Phase 8 tools exist.", requires_tool=True)]
    elif route.category is RouteCategory.document_search:
        model_task = ModelTask.structured_generation
        steps = [PlanStep(name="await_document_tool", goal="Search documents after Phase 8 tools exist.", requires_tool=True)]
    else:
        model_task = ModelTask.structured_generation
        steps = [PlanStep(name="await_website_tool", goal="Search websites after Phase 8 tools exist.", requires_tool=True)]

    plan = ExecutionPlan(route=route.category, model_task=model_task, steps=steps)
    trace.record(
        {
            "stage": "plan_execution",
            "status": "ok",
            "route": plan.route.value,
            "model_task": plan.model_task.value,
            "step_count": len(plan.steps),
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
    trace.record(
        {
            "stage": "authorize_tools",
            "status": "ok",
            "authorized_tool_count": 0,
            "requires_future_gateway": any(step.requires_tool for step in execution_plan.steps),
        }
    )
    return execution_plan


async def execute_tools(execution_plan: ExecutionPlan, trace: OrchestrationTrace) -> list[ToolExecutionResult]:
    trace.record(
        {
            "stage": "execute_tools",
            "status": "skipped",
            "tool_result_count": 0,
            "reason": "typed_tool_gateway_not_implemented",
        }
    )
    return []


def build_context(
    loaded_context: dict[str, Any],
    route: ModelRoute,
    model_selection: ModelSelection,
    execution_plan: ExecutionPlan,
    tool_results: list[ToolExecutionResult],
    trace: OrchestrationTrace,
) -> dict[str, Any]:
    context = {
        "loaded_context": loaded_context,
        "route": route.model_dump(mode="json"),
        "model_selection": model_selection.to_trace_metadata(),
        "execution_plan": execution_plan.model_dump(mode="json"),
        "tool_results": [result.model_dump(mode="json") for result in tool_results],
    }
    trace.record({"stage": "build_context", "status": "ok", "tool_result_count": len(tool_results)})
    return context


async def generate_answer(
    request: AgentTurnRequest,
    dependencies: AgentDependencies,
    route: ModelRoute,
    model_selection: ModelSelection,
    answer_context: dict[str, Any],
    trace: OrchestrationTrace,
) -> dict[str, Any]:
    if route.category is not RouteCategory.conversation:
        trace.record({"stage": "generate_answer", "status": "skipped", "reason": "tools_not_implemented"})
        return {
            "answer": (
                f"Route `{route.category.value}` was selected, but tool execution is not available until the typed "
                "tool gateway is implemented."
            ),
            "confidence": min(route.confidence, 0.5),
            "limitations": ["Tool execution is not implemented until Phase 8."],
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


def _safe_event(event: dict[str, Any]) -> dict[str, Any]:
    blocked_keys = {"api_key", "secret", "token", "password", "authorization"}
    return {key: value for key, value in event.items() if key.lower() not in blocked_keys}
