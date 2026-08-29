from __future__ import annotations

import time
import uuid

from pydantic import BaseModel, Field


class TraceRecord(BaseModel):
    trace_id: str
    route: str
    model: str
    plan_steps: int = 0
    tools: list[str] = Field(default_factory=list)
    retrieved: int = 0
    reranked: int = 0
    cache_hit: bool = False
    routing_ms: int = 0
    retrieval_ms: int = 0
    generation_ms: int = 0
    total_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    confidence: float = 0.0
    generated_sql: str | None = None
    events: list[dict] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)


class ObservabilityService:
    def __init__(self) -> None:
        self._traces: dict[str, TraceRecord] = {}

    def new_trace_id(self) -> str:
        return f"tr_{uuid.uuid4().hex[:12]}"

    def add_trace(self, trace: TraceRecord) -> TraceRecord:
        self._traces[trace.trace_id] = trace
        return trace

    def list_traces(self) -> list[TraceRecord]:
        return sorted(self._traces.values(), key=lambda trace: trace.created_at, reverse=True)

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        return self._traces.get(trace_id)


observability_service = ObservabilityService()
