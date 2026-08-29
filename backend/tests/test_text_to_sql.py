from __future__ import annotations

import pytest

from app.ai.provider import AIProvider
from app.ai.schemas import AIResponse, StructuredGenerationRequest, TextGenerationRequest
from app.ai.sql.guard import SQLGuard
from app.ai.sql.pipeline import TextToSQLPipeline
from app.ai.sql.schema_linker import SchemaLinker
from app.ai.sql.schemas import GeneratedSQL, SQLSafetyStatus


class FakeSQLProvider(AIProvider):
    def __init__(self, responses: list[GeneratedSQL]) -> None:
        self.responses = responses
        self.structured_calls: list[StructuredGenerationRequest] = []

    async def generate_text(self, request: TextGenerationRequest) -> AIResponse[str]:
        raise NotImplementedError

    async def generate_structured(self, request: StructuredGenerationRequest) -> AIResponse[GeneratedSQL]:
        self.structured_calls.append(request)
        return AIResponse(content=self.responses.pop(0), model="fake", provider="fake", latency_ms=1)


class FakeRow:
    def __init__(self, values: dict) -> None:
        self._mapping = values


class FakeResult:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self._rows = rows or []

    def all(self):
        return [FakeRow(row) for row in self._rows]


class FakeDB:
    def __init__(self, fail_once: bool = False) -> None:
        self.fail_once = fail_once
        self.executed: list[str] = []

    def execute(self, statement):
        sql = str(statement)
        self.executed.append(sql)
        if sql.startswith("SELECT") and self.fail_once:
            self.fail_once = False
            raise RuntimeError("column does not exist: private detail")
        if sql.startswith("SELECT"):
            return FakeResult([{"customer_name": "Ava", "revenue_cents": 12000}])
        return FakeResult()


def test_schema_linking_selects_relevant_tables_without_full_schema() -> None:
    result = SchemaLinker().link("Who were the five highest spending customers this quarter?")

    assert {"customers", "orders"}.issubset(result.table_names)
    assert 0.0 <= result.confidence <= 1.0


def test_sql_guard_accepts_select_and_adds_row_limit() -> None:
    result = SQLGuard({"orders"}).validate("select id from orders")

    assert result.status == SQLSafetyStatus.valid
    assert result.normalized_sql == "SELECT id FROM orders LIMIT 100"
    assert result.row_limit == 100


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("select id from orders; select id from customers", "multiple_statements"),
        ("delete from orders", "select_only"),
        ("select * from ai_traces", "unapproved_table"),
        ("select pg_sleep(10) from orders", "unsafe_function"),
        ("select * from private.orders", "unapproved_schema"),
    ],
)
def test_sql_guard_blocks_unsafe_sql(sql: str, reason: str) -> None:
    result = SQLGuard({"orders"}).validate(sql)

    assert result.status == SQLSafetyStatus.invalid
    assert result.reason == reason


@pytest.mark.asyncio
async def test_pipeline_executes_valid_sql_and_returns_confidence_metadata() -> None:
    provider = FakeSQLProvider([GeneratedSQL(sql="select id from orders", explanation="Read orders.")])
    db = FakeDB()

    result = await TextToSQLPipeline(provider, db=db).run("Show orders")

    assert result.execution.execution_success is True
    assert result.execution.row_count == 1
    assert result.confidence_metadata["execution_success"] is True
    assert result.confidence_metadata["row_count"] == 1


@pytest.mark.asyncio
async def test_pipeline_repairs_only_execution_errors_once() -> None:
    provider = FakeSQLProvider(
        [
            GeneratedSQL(sql="select bad_column from orders", explanation="Initial."),
            GeneratedSQL(sql="select id from orders", explanation="Repair."),
        ]
    )

    result = await TextToSQLPipeline(provider, db=FakeDB(fail_once=True)).run("Show orders")

    assert result.repair_attempted is True
    assert len(provider.structured_calls) == 2
    assert result.execution.execution_success is True


@pytest.mark.asyncio
async def test_pipeline_never_repairs_sql_that_fails_safety_validation() -> None:
    provider = FakeSQLProvider([GeneratedSQL(sql="drop table orders", explanation="Unsafe.")])

    result = await TextToSQLPipeline(provider, db=FakeDB()).run("Show orders")

    assert result.validation.status == SQLSafetyStatus.invalid
    assert result.repair_attempted is False
    assert len(provider.structured_calls) == 1
