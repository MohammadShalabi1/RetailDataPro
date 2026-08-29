from __future__ import annotations

from fastapi import APIRouter

from app.services.insight_service import RetailInsightService
from app.services.observability_service import TraceRecord, observability_service
from app.services.report_service import ExecutiveReport, ExecutiveReportService

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("/weekly-brief", response_model=ExecutiveReport)
def generate_weekly_brief() -> ExecutiveReport:
    trace_id = observability_service.new_trace_id()
    insights = RetailInsightService().generate(trace_id)
    report = ExecutiveReportService().generate_weekly_brief(trace_id, insights.insights)
    observability_service.add_trace(
        TraceRecord(
            trace_id=trace_id,
            route="multi_source",
            model="gemini-3.6-flash",
            plan_steps=4,
            tools=["analytics_summary", "document_search"],
            retrieved=0,
            reranked=0,
            total_ms=2,
            confidence=0.72,
            events=[{"stage": "executive_report", "status": "safe_draft", "section_count": len(report.sections)}],
        )
    )
    return report
