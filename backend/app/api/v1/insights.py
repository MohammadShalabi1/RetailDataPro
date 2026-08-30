from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.security import require_development
from app.services.insight_service import InsightResponse, RetailInsightService
from app.services.observability_service import TraceRecord, observability_service

router = APIRouter(prefix="/insights", tags=["insights"], dependencies=[Depends(require_development)])


@router.post("/generate", response_model=InsightResponse)
def generate_insights() -> InsightResponse:
    trace_id = observability_service.new_trace_id()
    response = RetailInsightService().generate(trace_id)
    observability_service.add_trace(
        TraceRecord(
            trace_id=trace_id,
            route="retail_analytics",
            model="deterministic_stats",
            plan_steps=2,
            tools=["analytics_summary"],
            total_ms=1,
            confidence=0.78,
            events=[{"stage": "insight_engine", "status": "ok", "insight_count": len(response.insights)}],
        )
    )
    return response
