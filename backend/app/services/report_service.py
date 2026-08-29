from __future__ import annotations

from pydantic import BaseModel, Field

from app.services.insight_service import RetailInsight


class ExecutiveReport(BaseModel):
    trace_id: str
    sections: dict[str, str]
    sources: list[str] = Field(default_factory=list)


class ExecutiveReportService:
    def generate_weekly_brief(self, trace_id: str, insights: list[RetailInsight]) -> ExecutiveReport:
        insight_text = "; ".join(insight.title for insight in insights) or "No unusual changes detected."
        return ExecutiveReport(
            trace_id=trace_id,
            sections={
                "Executive Summary": f"Weekly retail brief generated from deterministic KPIs and evidence checks: {insight_text}",
                "Revenue": "Revenue review uses deterministic analytics outputs.",
                "Top Products": "Top product movement should be read from analytics_summary output.",
                "Inventory Risks": "Inventory risks are flagged only from stock and velocity rules.",
                "Customer Trends": "Customer trend section is reserved for customer analytics evidence.",
                "Supplier / Document Signals": "External supplier claims require cited retrieved evidence.",
                "Recommended Follow-ups": "Review medium/high severity anomalies before operational action.",
            },
            sources=["analytics_summary"],
        )
