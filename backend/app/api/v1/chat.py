from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from app.ai.gemini_provider import GeminiProvider
from app.ai.orchestrator import AgentDependencies, AgentTurnRequest, AgentTurnResult, run_turn
from app.ai.provider import AIProvider
from app.services.analytics_service import AnalyticsService, get_analytics_service
from app.services.document_service import DocumentService, get_document_service
from app.services.observability_service import TraceRecord, observability_service

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    conversation_id: str | None = None
    document_source_ids: list[str] = Field(default_factory=list, max_length=10)


class ChatResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    answer: str
    trace_id: str
    route: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)
    model: str | None = None
    tool_results: list[dict] = Field(default_factory=list)


def get_ai_provider() -> AIProvider:
    return GeminiProvider()


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    ai_provider: AIProvider = Depends(get_ai_provider),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    document_service: DocumentService = Depends(get_document_service),
) -> ChatResponse:
    result = await run_turn(
        AgentTurnRequest(
            question=request.question,
            conversation_id=request.conversation_id,
            document_source_ids=request.document_source_ids,
        ),
        AgentDependencies(ai_provider=ai_provider, analytics_service=analytics_service, document_service=document_service),
    )
    trace_id = observability_service.new_trace_id()
    observability_service.add_trace(_to_trace_record(trace_id, result))
    return ChatResponse(
        answer=result.answer,
        trace_id=trace_id,
        route=result.route.category.value if result.route else None,
        confidence=result.confidence,
        limitations=result.limitations,
        model=result.model_selection.model if result.model_selection else None,
        tool_results=[tool.model_dump(mode="json") for tool in result.tool_results],
    )


def _to_trace_record(trace_id: str, result: AgentTurnResult) -> TraceRecord:
    route = result.route.category.value if result.route else "blocked"
    model = result.model_selection.model if result.model_selection else "none"
    tool_results = [tool.model_dump(mode="json") for tool in result.tool_results]
    token_event = next((event for event in result.trace.events if event.get("stage") == "generate_answer"), {})
    return TraceRecord(
        trace_id=trace_id,
        route=route,
        model=model,
        plan_steps=len(result.execution_plan.steps) if result.execution_plan else 0,
        tools=[tool["tool_name"] for tool in tool_results],
        retrieved=0,
        reranked=0,
        cache_hit=False,
        generation_ms=int(token_event.get("latency_ms") or 0),
        total_ms=sum(int(event.get("latency_ms") or 0) for event in result.trace.events),
        input_tokens=int(token_event.get("input_tokens") or 0),
        output_tokens=int(token_event.get("output_tokens") or 0),
        confidence=result.confidence,
        generated_sql=None,
        events=result.trace.events,
    )
