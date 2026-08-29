from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolName(str, Enum):
    analytics_summary = "analytics_summary"
    retail_sql = "retail_sql"
    document_search = "document_search"
    website_search = "website_search"


class ToolStatus(str, Enum):
    success = "success"
    rejected = "rejected"
    unavailable = "unavailable"
    validation_error = "validation_error"
    execution_error = "execution_error"
    timeout = "timeout"


class ToolExecutionRequest(BaseModel):
    tool_name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionContext(BaseModel):
    user_role: str = "analyst"
    trace: Any | None = None
    analytics_service: Any | None = None

    model_config = {"arbitrary_types_allowed": True}


class ToolExecutionResult(BaseModel):
    tool_name: str
    status: ToolStatus
    output: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int
    authorized: bool
    error_code: str | None = None

    def to_trace_metadata(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "authorized": self.authorized,
            "error_code": self.error_code,
        }


class AnalyticsSummaryInput(BaseModel):
    question: str = Field(min_length=1, max_length=1_000)
    start_date: date | None = None
    end_date: date | None = None
    limit: int = Field(default=5, ge=1, le=20)


class AnalyticsSummaryOutput(BaseModel):
    summary_type: str
    question: str
    data: dict[str, Any]
