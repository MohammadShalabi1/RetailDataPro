from __future__ import annotations

import asyncio
from datetime import date

import pytest
from pydantic import BaseModel

from app.schemas.analytics import RevenueResponse
from app.tools.gateway import authorize_and_execute_tool
from app.tools.registry import ToolDefinition, ToolRegistry, build_default_tool_registry
from app.tools.schemas import ToolExecutionContext, ToolExecutionRequest, ToolName, ToolStatus


class FakeAnalyticsService:
    def __init__(self) -> None:
        self.revenue_calls = []
        self.top_product_calls = []

    def get_revenue(self, start_date=None, end_date=None) -> RevenueResponse:
        self.revenue_calls.append((start_date, end_date))
        return RevenueResponse(
            start_date=start_date or date(2026, 1, 1),
            end_date=end_date or date(2026, 1, 31),
            total_revenue_cents=25_000,
            order_count=5,
            average_order_value_cents=5_000,
        )

    def get_top_products(self, start_date=None, end_date=None, limit=10):
        self.top_product_calls.append((start_date, end_date, limit))
        return {
            "start_date": start_date or date(2026, 1, 1),
            "end_date": end_date or date(2026, 1, 31),
            "items": [
                {
                    "product_id": "prod_1",
                    "product_name": "Everyday Coffee",
                    "sku": "COF-001",
                    "category_name": "Grocery",
                    "revenue_cents": 50_000,
                    "units_sold": 25,
                    "order_count": 12,
                }
            ],
        }

    def get_category_performance(self, start_date=None, end_date=None, limit=10):
        return {
            "start_date": start_date or date(2026, 8, 1),
            "end_date": end_date or date(2026, 8, 31),
            "items": [
                {
                    "category_id": "cat_1",
                    "category_name": "Household",
                    "revenue_cents": 90_000,
                    "units_sold": 120,
                    "order_count": 40,
                    "gross_margin_cents": 20_000,
                }
            ],
        }


class SlowInput(BaseModel):
    question: str


class SlowOutput(BaseModel):
    ok: bool


async def slow_executor(tool_input: BaseModel, context: ToolExecutionContext) -> SlowOutput:
    await asyncio.sleep(0.05)
    return SlowOutput(ok=True)


def test_default_registry_contains_expected_tools() -> None:
    registry = build_default_tool_registry()

    assert registry.names() == {
        ToolName.analytics_summary,
        ToolName.retail_sql,
        ToolName.document_search,
        ToolName.website_search,
    }
    assert registry.get(ToolName.analytics_summary) is not None
    assert registry.get(ToolName.analytics_summary).timeout_seconds == 5.0


@pytest.mark.asyncio
async def test_unknown_tool_is_rejected_by_gateway() -> None:
    result = await authorize_and_execute_tool(
        ToolExecutionRequest(tool_name="unknown_tool", input={}),
        ToolExecutionContext(user_role="analyst"),
    )

    assert result.status == ToolStatus.rejected
    assert result.authorized is False
    assert result.error_code == "unknown_tool"


@pytest.mark.asyncio
async def test_unavailable_tools_do_not_execute() -> None:
    result = await authorize_and_execute_tool(
        ToolExecutionRequest(tool_name="retail_sql", input={"question": "Generate SQL"}),
        ToolExecutionContext(user_role="analyst"),
    )

    assert result.status == ToolStatus.unavailable
    assert result.authorized is True
    assert result.error_code == "tool_unavailable"
    assert result.output == {}


@pytest.mark.asyncio
async def test_authorization_is_required_before_execution() -> None:
    analytics_service = FakeAnalyticsService()

    result = await authorize_and_execute_tool(
        ToolExecutionRequest(tool_name="analytics_summary", input={"question": "What was revenue?"}),
        ToolExecutionContext(user_role="guest", analytics_service=analytics_service),
    )

    assert result.status == ToolStatus.rejected
    assert result.authorized is False
    assert result.error_code == "unauthorized"
    assert analytics_service.revenue_calls == []


@pytest.mark.asyncio
async def test_invalid_input_fails_validation() -> None:
    result = await authorize_and_execute_tool(
        ToolExecutionRequest(tool_name="analytics_summary", input={"question": ""}),
        ToolExecutionContext(user_role="analyst", analytics_service=FakeAnalyticsService()),
    )

    assert result.status == ToolStatus.validation_error
    assert result.authorized is True
    assert result.error_code == "invalid_input"


@pytest.mark.asyncio
async def test_analytics_summary_returns_typed_output_through_gateway() -> None:
    analytics_service = FakeAnalyticsService()

    result = await authorize_and_execute_tool(
        ToolExecutionRequest(
            tool_name="analytics_summary",
            input={"question": "What was revenue last month?", "start_date": "2026-01-01", "end_date": "2026-01-31"},
        ),
        ToolExecutionContext(user_role="analyst", analytics_service=analytics_service),
    )

    assert result.status == ToolStatus.success
    assert result.authorized is True
    assert result.error_code is None
    assert result.output["summary_type"] == "revenue"
    assert result.output["data"]["total_revenue_cents"] == 25_000
    assert analytics_service.revenue_calls == [(date(2026, 1, 1), date(2026, 1, 31))]


@pytest.mark.asyncio
async def test_best_selling_question_uses_top_products_summary() -> None:
    result = await authorize_and_execute_tool(
        ToolExecutionRequest(tool_name="analytics_summary", input={"question": "What is our best selling product last week?"}),
        ToolExecutionContext(user_role="analyst", analytics_service=FakeAnalyticsService()),
    )

    assert result.status == ToolStatus.success
    assert result.output["summary_type"] == "top_products"


@pytest.mark.asyncio
async def test_category_question_takes_precedence_over_supplier_word() -> None:
    result = await authorize_and_execute_tool(
        ToolExecutionRequest(
            tool_name="analytics_summary",
            input={"question": "Which product categories were weak and do supplier issues explain them?"},
        ),
        ToolExecutionContext(user_role="analyst", analytics_service=FakeAnalyticsService()),
    )

    assert result.status == ToolStatus.success
    assert result.output["summary_type"] == "category_performance"


@pytest.mark.asyncio
async def test_gateway_returns_timeout_result() -> None:
    registry = ToolRegistry(
        [
            ToolDefinition(
                name=ToolName.analytics_summary,
                input_model=SlowInput,
                output_model=SlowOutput,
                allowed_roles=frozenset({"analyst"}),
                timeout_seconds=0.001,
                is_available=True,
                executor=slow_executor,
            )
        ]
    )

    result = await authorize_and_execute_tool(
        ToolExecutionRequest(tool_name="analytics_summary", input={"question": "slow"}),
        ToolExecutionContext(user_role="analyst"),
        registry=registry,
    )

    assert result.status == ToolStatus.timeout
    assert result.error_code == "tool_timeout"


@pytest.mark.asyncio
async def test_trace_metadata_contains_no_secrets() -> None:
    trace: list[dict] = []

    await authorize_and_execute_tool(
        ToolExecutionRequest(tool_name="analytics_summary", input={"question": "What was revenue?"}),
        ToolExecutionContext(user_role="analyst", analytics_service=FakeAnalyticsService(), trace=trace),
    )

    assert trace
    trace_text = str(trace)
    assert "api_key" not in trace_text
    assert "secret" not in trace_text.lower()
    assert trace[0]["tool_name"] == "analytics_summary"
    assert trace[0]["status"] == "success"
