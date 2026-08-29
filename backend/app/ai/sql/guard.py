from __future__ import annotations

from sqlglot import exp, parse
from sqlglot.errors import ParseError

from app.ai.sql.schemas import SQLSafetyStatus, SQLValidationResult

DEFAULT_ROW_LIMIT = 100
MAX_ROW_LIMIT = 500
APPROVED_SCHEMA = "public"
UNSAFE_FUNCTIONS = {
    "pg_read_file",
    "pg_ls_dir",
    "pg_sleep",
    "dblink",
    "lo_import",
    "lo_export",
}


class SQLGuard:
    def __init__(
        self,
        approved_tables: set[str],
        row_limit: int = DEFAULT_ROW_LIMIT,
        approved_schema: str = APPROVED_SCHEMA,
    ) -> None:
        self._approved_tables = approved_tables
        self._row_limit = min(row_limit, MAX_ROW_LIMIT)
        self._approved_schema = approved_schema

    def validate(self, sql: str) -> SQLValidationResult:
        try:
            statements = parse(sql, read="postgres")
        except ParseError:
            return self._invalid("parse_error")

        if len(statements) != 1:
            return self._invalid("multiple_statements")

        statement = statements[0]
        if not isinstance(statement, exp.Select):
            return self._invalid("select_only")

        tables = self._tables(statement)
        if not tables:
            return self._invalid("missing_table")

        disallowed_tables = tables - self._approved_tables
        if disallowed_tables:
            return self._invalid("unapproved_table")

        if self._has_unapproved_schema(statement):
            return self._invalid("unapproved_schema")

        if self._has_unsafe_function(statement):
            return self._invalid("unsafe_function")

        limited = self._with_safe_limit(statement)
        return SQLValidationResult(
            status=SQLSafetyStatus.valid,
            normalized_sql=limited.sql(dialect="postgres"),
            approved_tables=sorted(tables),
            row_limit=self._row_limit,
        )

    def _invalid(self, reason: str) -> SQLValidationResult:
        return SQLValidationResult(status=SQLSafetyStatus.invalid, reason=reason)

    def _tables(self, statement: exp.Expression) -> set[str]:
        return {table.name for table in statement.find_all(exp.Table)}

    def _has_unapproved_schema(self, statement: exp.Expression) -> bool:
        for table in statement.find_all(exp.Table):
            schema = table.db
            if schema and schema != self._approved_schema:
                return True
        return False

    def _has_unsafe_function(self, statement: exp.Expression) -> bool:
        for function in statement.find_all(exp.Func):
            if function.sql_name().lower() in UNSAFE_FUNCTIONS:
                return True
        for function in statement.find_all(exp.Anonymous):
            if str(function.this).lower() in UNSAFE_FUNCTIONS:
                return True
        return False

    def _with_safe_limit(self, statement: exp.Expression) -> exp.Expression:
        existing_limit = statement.args.get("limit")
        if existing_limit is None:
            return statement.limit(self._row_limit)

        try:
            current_limit = int(existing_limit.expression.this)
        except (AttributeError, TypeError, ValueError):
            return statement.limit(self._row_limit)

        if current_limit > self._row_limit:
            return statement.limit(self._row_limit)
        return statement
