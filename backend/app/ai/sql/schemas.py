from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SQLSafetyStatus(str, Enum):
    valid = "valid"
    invalid = "invalid"


class GeneratedSQL(BaseModel):
    sql: str = Field(min_length=1, max_length=5_000)
    explanation: str = Field(min_length=1, max_length=1_000)


class SchemaTable(BaseModel):
    name: str
    description: str
    columns: list[str]
    score: float = Field(ge=0.0, le=1.0)


class SchemaLinkResult(BaseModel):
    tables: list[SchemaTable]
    confidence: float = Field(ge=0.0, le=1.0)

    @property
    def table_names(self) -> set[str]:
        return {table.name for table in self.tables}


class SQLValidationResult(BaseModel):
    status: SQLSafetyStatus
    normalized_sql: str | None = None
    reason: str | None = None
    approved_tables: list[str] = Field(default_factory=list)
    row_limit: int | None = None

    @property
    def is_valid(self) -> bool:
        return self.status is SQLSafetyStatus.valid


class SQLExecutionResult(BaseModel):
    execution_success: bool
    rows: list[dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    sanitized_error: str | None = None


class SQLPipelineResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    generated_sql: GeneratedSQL | None
    validation: SQLValidationResult
    execution: SQLExecutionResult
    schema_match_confidence: float = Field(ge=0.0, le=1.0)
    repair_attempted: bool = False

    @property
    def confidence_metadata(self) -> dict[str, Any]:
        return {
            "schema_match_confidence": self.schema_match_confidence,
            "execution_success": self.execution.execution_success,
            "row_count": self.execution.row_count,
            "repair_attempted": self.repair_attempted,
        }
