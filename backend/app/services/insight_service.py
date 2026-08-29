from __future__ import annotations

from pydantic import BaseModel, Field


class RetailInsight(BaseModel):
    title: str
    explanation: str
    severity: str
    metric: str
    change_percent: float
    evidence: list[str] = Field(default_factory=list)


class InsightResponse(BaseModel):
    trace_id: str
    insights: list[RetailInsight]


class RetailInsightService:
    def generate(self, trace_id: str) -> InsightResponse:
        insights = [
            RetailInsight(
                title="Accessories revenue dropped",
                explanation="Accessories revenue is unusually below the recent baseline and should be reviewed with supplier context.",
                severity="medium",
                metric="category_revenue",
                change_percent=-24.0,
                evidence=["deterministic_anomaly_rule", "category_performance"],
            ),
            RetailInsight(
                title="Inventory risk for fast movers",
                explanation="One high-demand product group appears near reorder threshold based on deterministic stock checks.",
                severity="high",
                metric="inventory_days_remaining",
                change_percent=-18.0,
                evidence=["inventory_summary", "sales_velocity_rule"],
            ),
        ]
        return InsightResponse(trace_id=trace_id, insights=insights)
