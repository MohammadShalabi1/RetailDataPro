from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.tools.schemas import AnalyticsSummaryInput, AnalyticsSummaryOutput, ToolExecutionContext, ToolName

ToolExecutor = Callable[[BaseModel, ToolExecutionContext], Awaitable[BaseModel]]


@dataclass(frozen=True)
class ToolDefinition:
    name: ToolName
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    allowed_roles: frozenset[str]
    timeout_seconds: float
    is_available: bool
    executor: ToolExecutor | None = None


class ToolRegistry:
    def __init__(self, definitions: list[ToolDefinition]) -> None:
        self._definitions = {definition.name: definition for definition in definitions}

    def get(self, name: ToolName) -> ToolDefinition | None:
        return self._definitions.get(name)

    def names(self) -> set[ToolName]:
        return set(self._definitions)


def build_default_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolDefinition(
                name=ToolName.analytics_summary,
                input_model=AnalyticsSummaryInput,
                output_model=AnalyticsSummaryOutput,
                allowed_roles=frozenset({"analyst", "admin", "system"}),
                timeout_seconds=5.0,
                is_available=True,
                executor=execute_analytics_summary,
            ),
            _unavailable_tool(ToolName.retail_sql),
            _unavailable_tool(ToolName.document_search),
            _unavailable_tool(ToolName.website_search),
        ]
    )


async def execute_analytics_summary(
    tool_input: BaseModel,
    context: ToolExecutionContext,
) -> AnalyticsSummaryOutput:
    parsed_input = _ensure_analytics_input(tool_input)
    if context.analytics_service is None:
        return AnalyticsSummaryOutput(
            summary_type="dependency_missing",
            question=parsed_input.question,
            data={"message": "Analytics service is required to execute analytics_summary."},
        )

    question = parsed_input.question.lower()
    analytics_service = context.analytics_service
    if "inventory" in question or "stock" in question:
        response = analytics_service.get_inventory(limit=parsed_input.limit)
        summary_type = "inventory"
    elif "supplier" in question:
        response = analytics_service.get_supplier_performance(
            parsed_input.start_date,
            parsed_input.end_date,
            parsed_input.limit,
        )
        summary_type = "supplier_performance"
    elif "category" in question or "categories" in question:
        response = analytics_service.get_category_performance(
            parsed_input.start_date,
            parsed_input.end_date,
            parsed_input.limit,
        )
        summary_type = "category_performance"
    elif "product" in question or "products" in question:
        response = analytics_service.get_top_products(
            parsed_input.start_date,
            parsed_input.end_date,
            parsed_input.limit,
        )
        summary_type = "top_products"
    elif "customer" in question or "customers" in question:
        response = analytics_service.get_top_customers(
            parsed_input.start_date,
            parsed_input.end_date,
            parsed_input.limit,
        )
        summary_type = "top_customers"
    elif "trend" in question or "trends" in question:
        response = analytics_service.get_sales_trends(parsed_input.start_date, parsed_input.end_date, "month")
        summary_type = "sales_trends"
    else:
        response = analytics_service.get_revenue(parsed_input.start_date, parsed_input.end_date)
        summary_type = "revenue"

    return AnalyticsSummaryOutput(
        summary_type=summary_type,
        question=parsed_input.question,
        data=response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response),
    )


def _unavailable_tool(name: ToolName) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        input_model=BaseModel,
        output_model=BaseModel,
        allowed_roles=frozenset({"analyst", "admin", "system"}),
        timeout_seconds=5.0,
        is_available=False,
        executor=None,
    )


def _ensure_analytics_input(tool_input: BaseModel) -> AnalyticsSummaryInput:
    if isinstance(tool_input, AnalyticsSummaryInput):
        return tool_input
    return AnalyticsSummaryInput.model_validate(tool_input)
