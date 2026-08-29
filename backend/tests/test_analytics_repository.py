from datetime import date
from typing import cast

from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from app.repositories.analytics_repository import AnalyticsRepository


def test_analytics_repository_statements_compile_for_postgresql() -> None:
    repository = AnalyticsRepository(cast(Session, None))
    start_date = date(2026, 1, 1)
    end_date = date(2026, 3, 31)

    statements = [
        repository.build_revenue_statement(start_date, end_date),
        repository.build_sales_trends_statement(start_date, end_date, "month"),
        repository.build_top_products_statement(start_date, end_date, 10),
        repository.build_top_customers_statement(start_date, end_date, 10),
        repository.build_category_performance_statement(start_date, end_date, 10),
        repository.build_inventory_statement(10),
        repository.build_supplier_performance_statement(start_date, end_date, 10),
    ]

    compiled_sql = [str(statement.compile(dialect=postgresql.dialect())) for statement in statements]

    assert all("SELECT" in sql for sql in compiled_sql)
    assert any("orders" in sql for sql in compiled_sql)
    assert any("order_items" in sql for sql in compiled_sql)
    assert any("products" in sql for sql in compiled_sql)


def test_sales_trends_statement_uses_date_bucket() -> None:
    repository = AnalyticsRepository(cast(Session, None))

    statement = repository.build_sales_trends_statement(date(2026, 1, 1), date(2026, 1, 31), "week")

    assert "date_trunc" in str(statement.compile(dialect=postgresql.dialect()))
