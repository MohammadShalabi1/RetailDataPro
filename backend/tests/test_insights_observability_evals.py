from __future__ import annotations

from fastapi.testclient import TestClient

from app.evals.runner import DeterministicEvalRunner
from app.main import app
from app.services.insight_service import RetailInsightService
from app.services.observability_service import TraceRecord, ObservabilityService
from app.services.report_service import ExecutiveReportService


def test_insight_engine_returns_deterministic_insights() -> None:
    result = RetailInsightService().generate("tr_test")

    assert result.trace_id == "tr_test"
    assert result.insights
    assert all(insight.evidence for insight in result.insights)


def test_executive_report_uses_insight_sections() -> None:
    insights = RetailInsightService().generate("tr_test")
    report = ExecutiveReportService().generate_weekly_brief("tr_test", insights.insights)

    assert "Executive Summary" in report.sections
    assert "Recommended Follow-ups" in report.sections
    assert report.sources == ["analytics_summary"]


def test_observability_service_lists_and_gets_traces() -> None:
    service = ObservabilityService()
    assert service.list_traces() == []
    trace = service.add_trace(TraceRecord(trace_id="tr_custom", route="conversation", model="model"))

    assert service.get_trace("tr_custom") == trace
    assert any(item.trace_id == "tr_custom" for item in service.list_traces())


def test_eval_runner_computes_metrics_from_cases() -> None:
    result = DeterministicEvalRunner().run()

    assert result.rows_evaluated > 0
    assert {metric.name for metric in result.metrics} >= {"Router Accuracy", "Unsafe SQL Blocked"}


def test_admin_backend_routes_return_contracts() -> None:
    client = TestClient(app)

    traces = client.get("/api/observability/traces")
    evals = client.get("/api/evaluations/latest")
    insights = client.post("/api/insights/generate")
    report = client.post("/api/reports/weekly-brief")

    assert traces.status_code == 200
    assert evals.status_code == 200
    assert insights.status_code == 200
    assert report.status_code == 200
    assert evals.json()["metrics"]
    assert insights.json()["insights"]
