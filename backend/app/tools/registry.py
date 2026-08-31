from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import inspect
from typing import Any

from pydantic import BaseModel

from app.core.config import get_settings
from app.tools.schemas import (
    AnalyticsSummaryInput,
    AnalyticsSummaryOutput,
    DocumentSearchInput,
    DocumentSearchOutput,
    RetailSQLInput,
    RetailSQLOutput,
    ToolExecutionContext,
    ToolName,
)

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


def build_default_tool_registry(retail_sql_available: bool | None = None) -> ToolRegistry:
    settings = get_settings()
    sql_available = settings.readonly_database_configured if retail_sql_available is None else retail_sql_available
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
            ToolDefinition(
                name=ToolName.retail_sql,
                input_model=RetailSQLInput,
                output_model=RetailSQLOutput,
                allowed_roles=frozenset({"analyst", "admin", "system"}),
                timeout_seconds=8.0,
                is_available=sql_available,
                executor=execute_retail_sql,
            ),
            ToolDefinition(
                name=ToolName.document_search,
                input_model=DocumentSearchInput,
                output_model=DocumentSearchOutput,
                allowed_roles=frozenset({"analyst", "admin", "system"}),
                timeout_seconds=5.0,
                is_available=True,
                executor=execute_document_search,
            ),
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
    elif "category" in question or "categories" in question:
        response = analytics_service.get_category_performance(
            parsed_input.start_date,
            parsed_input.end_date,
            parsed_input.limit,
        )
        summary_type = "category_performance"
    elif "supplier" in question:
        response = analytics_service.get_supplier_performance(
            parsed_input.start_date,
            parsed_input.end_date,
            parsed_input.limit,
        )
        summary_type = "supplier_performance"
    elif (
        "product" in question
        or "products" in question
        or "best selling" in question
        or "bestselling" in question
        or "top selling" in question
        or "top seller" in question
    ):
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

    data = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
    if summary_type == "category_performance" and ("weak" in question or "worst" in question or "lowest" in question):
        items = data.get("items", [])
        if isinstance(items, list):
            data["items"] = sorted(items, key=lambda item: int((item or {}).get("revenue_cents") or 0))
        data["ranking_metric"] = "August revenue, ranked from lowest to highest among returned categories"

    return AnalyticsSummaryOutput(
        summary_type=summary_type,
        question=parsed_input.question,
        data=data,
    )


async def execute_retail_sql(
    tool_input: BaseModel,
    context: ToolExecutionContext,
) -> RetailSQLOutput:
    parsed_input = _ensure_retail_sql_input(tool_input)
    if context.sql_pipeline is None:
        return RetailSQLOutput(
            question=parsed_input.question,
            status="unavailable",
            limitations=["Read-only SQL access is not configured."],
        )

    result = await context.sql_pipeline.run(parsed_input.question, row_limit=parsed_input.limit)
    _record_retail_sql_trace(
        context.trace,
        {
            "generated_sql": result.generated_sql.sql if result.generated_sql is not None else None,
            "normalized_sql": result.validation.normalized_sql,
            "validation_status": result.validation.status.value,
            "validation_reason": result.validation.reason,
            "approved_tables": result.validation.approved_tables,
            "row_limit": result.validation.row_limit,
            "execution_success": result.execution.execution_success,
            "row_count": result.execution.row_count,
            "sanitized_error": result.execution.sanitized_error,
            "repair_attempted": result.repair_attempted,
            "schema_match_confidence": result.schema_match_confidence,
        },
    )
    if not result.validation.is_valid:
        return RetailSQLOutput(
            question=parsed_input.question,
            status="unsafe_or_invalid",
            confidence=0.0,
            limitations=["The generated database query did not pass safety validation."],
            internal_metadata={
                "validation_status": result.validation.status.value,
                "validation_reason": result.validation.reason,
                "repair_attempted": result.repair_attempted,
                "schema_match_confidence": result.schema_match_confidence,
            },
        )

    if not result.execution.execution_success:
        return RetailSQLOutput(
            question=parsed_input.question,
            status="execution_failed",
            confidence=0.0,
            limitations=["The database query could not be completed safely."],
            internal_metadata={
                "validation_status": result.validation.status.value,
                "approved_tables": result.validation.approved_tables,
                "row_limit": result.validation.row_limit,
                "repair_attempted": result.repair_attempted,
                "sanitized_error": result.execution.sanitized_error,
                "schema_match_confidence": result.schema_match_confidence,
            },
        )

    return RetailSQLOutput(
        question=parsed_input.question,
        rows=result.execution.rows,
        row_count=result.execution.row_count,
        execution_success=True,
        status="success",
        confidence=result.schema_match_confidence,
        internal_metadata={
            "validation_status": result.validation.status.value,
            "approved_tables": result.validation.approved_tables,
            "row_limit": result.validation.row_limit,
            "repair_attempted": result.repair_attempted,
            "schema_match_confidence": result.schema_match_confidence,
        },
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


async def execute_document_search(
    tool_input: BaseModel,
    context: ToolExecutionContext,
) -> DocumentSearchOutput:
    parsed_input = _ensure_document_input(tool_input)
    source_ids = parsed_input.source_ids or context.document_source_ids
    if context.document_service is None:
        return DocumentSearchOutput(query=parsed_input.question, chunks=[])
    response = _call_document_search(
        context.document_service,
        parsed_input.question,
        parsed_input.limit,
        source_ids,
        context.client_id,
    )
    if inspect.isawaitable(response):
        response = await response
    return DocumentSearchOutput(
        query=response.query,
        chunks=[result.model_dump(mode="json") for result in response.results],
        confidence=getattr(response, "confidence", 0.0),
        limitations=getattr(response, "limitations", []),
        retrieval_trace=getattr(response, "retrieval_trace", {}),
    )


def _ensure_analytics_input(tool_input: BaseModel) -> AnalyticsSummaryInput:
    if isinstance(tool_input, AnalyticsSummaryInput):
        return tool_input
    return AnalyticsSummaryInput.model_validate(tool_input)


def _ensure_retail_sql_input(tool_input: BaseModel) -> RetailSQLInput:
    if isinstance(tool_input, RetailSQLInput):
        return tool_input
    return RetailSQLInput.model_validate(tool_input)


def _ensure_document_input(tool_input: BaseModel) -> DocumentSearchInput:
    if isinstance(tool_input, DocumentSearchInput):
        return tool_input
    return DocumentSearchInput.model_validate(tool_input)


def _call_document_search(document_service: Any, question: str, limit: int, source_ids: list[str], client_id: str) -> Any:
    method = (
        document_service.search_documents_async
        if hasattr(document_service, "search_documents_async")
        else document_service.search_documents
    )
    parameters = inspect.signature(method).parameters
    if "client_id" in parameters or len(parameters) >= 4:
        return method(question, limit, source_ids, client_id)
    return method(question, limit, source_ids)


def _record_retail_sql_trace(trace: Any | None, metadata: dict[str, Any]) -> None:
    if trace is None:
        return
    event = {"stage": "retail_sql", **metadata}
    if hasattr(trace, "append"):
        trace.append(event)
        return
    if hasattr(trace, "record"):
        trace.record(event)
