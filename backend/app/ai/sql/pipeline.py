from __future__ import annotations

from typing import Any

from sqlalchemy import text

from app.ai.provider import AIProvider
from app.ai.schemas import StructuredGenerationRequest
from app.ai.sql.guard import SQLGuard
from app.ai.sql.schema_linker import SchemaLinker
from app.ai.sql.schemas import GeneratedSQL, SQLExecutionResult, SQLPipelineResult, SQLSafetyStatus


class TextToSQLPipeline:
    def __init__(
        self,
        ai_provider: AIProvider,
        schema_linker: SchemaLinker | None = None,
        db: Any | None = None,
        row_limit: int = 100,
    ) -> None:
        self._ai_provider = ai_provider
        self._schema_linker = schema_linker or SchemaLinker()
        self._db = db
        self._row_limit = row_limit

    async def run(self, question: str) -> SQLPipelineResult:
        schema = self._schema_linker.link(question)
        generated = await self._generate(question, schema.table_names)
        guard = SQLGuard(schema.table_names, row_limit=self._row_limit)
        validation = guard.validate(generated.sql)
        if not validation.is_valid:
            return SQLPipelineResult(
                generated_sql=generated,
                validation=validation,
                execution=SQLExecutionResult(execution_success=False, sanitized_error="sql_failed_safety_validation"),
                schema_match_confidence=schema.confidence,
            )

        execution = self._execute(validation.normalized_sql)
        repair_attempted = False
        if not execution.execution_success and validation.status is SQLSafetyStatus.valid:
            repaired = await self._repair(question, generated.sql, execution.sanitized_error or "execution_error")
            repair_validation = guard.validate(repaired.sql)
            repair_attempted = True
            if repair_validation.is_valid:
                execution = self._execute(repair_validation.normalized_sql)
                generated = repaired
                validation = repair_validation

        return SQLPipelineResult(
            generated_sql=generated,
            validation=validation,
            execution=execution,
            schema_match_confidence=schema.confidence,
            repair_attempted=repair_attempted,
        )

    async def _generate(self, question: str, tables: set[str]) -> GeneratedSQL:
        prompt = (
            "Generate one read-only PostgreSQL SELECT statement for the retail database.\n"
            "Return only structured fields matching GeneratedSQL.\n"
            f"Approved tables: {sorted(tables)}\n"
            f"Question: {question}"
        )
        response = await self._ai_provider.generate_structured(
            StructuredGenerationRequest(prompt=prompt, response_model=GeneratedSQL)
        )
        return response.content

    async def _repair(self, question: str, sql: str, sanitized_error: str) -> GeneratedSQL:
        prompt = (
            "Repair this PostgreSQL SELECT only if the error is a normal execution mistake. "
            "Do not use unsafe functions, DDL, DML, or unapproved tables.\n"
            f"Question: {question}\n"
            f"SQL: {sql}\n"
            f"Sanitized error: {sanitized_error}"
        )
        response = await self._ai_provider.generate_structured(
            StructuredGenerationRequest(prompt=prompt, response_model=GeneratedSQL)
        )
        return response.content

    def _execute(self, normalized_sql: str | None) -> SQLExecutionResult:
        if normalized_sql is None:
            return SQLExecutionResult(execution_success=False, sanitized_error="missing_valid_sql")
        if self._db is None:
            return SQLExecutionResult(execution_success=False, sanitized_error="readonly_database_not_configured")

        try:
            self._db.execute(text("SET LOCAL statement_timeout = '3000ms'"))
            self._db.execute(text("SET TRANSACTION READ ONLY"))
            rows = [dict(row._mapping) for row in self._db.execute(text(normalized_sql)).all()]
            return SQLExecutionResult(execution_success=True, rows=rows, row_count=len(rows))
        except Exception as exc:
            return SQLExecutionResult(execution_success=False, sanitized_error=_sanitize_error(exc))


def _sanitize_error(exc: Exception) -> str:
    return exc.__class__.__name__
