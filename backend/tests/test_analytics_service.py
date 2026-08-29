from datetime import date, datetime

import pytest

from app.services.analytics_service import AnalyticsService


class FakeAnalyticsRepository:
    def __init__(self) -> None:
        self.last_limit: int | None = None

    def get_revenue(self, start_date: date, end_date: date) -> dict:
        return {"total_revenue_cents": 12_000, "order_count": 3}

    def get_sales_trends(self, start_date: date, end_date: date, interval: str) -> list[dict]:
        return [
            {
                "bucket_start": datetime(2026, 1, 1),
                "revenue_cents": 10_000,
                "order_count": 4,
                "units_sold": 9,
            }
        ]

    def get_top_products(self, start_date: date, end_date: date, limit: int) -> list[dict]:
        self.last_limit = limit
        return []

    def get_top_customers(self, start_date: date, end_date: date, limit: int) -> list[dict]:
        return [
            {
                "customer_id": "cus_1",
                "customer_name": "Ava Retail",
                "email": "ava@example.test",
                "segment": "vip",
                "revenue_cents": 9_000,
                "order_count": 3,
            }
        ]

    def get_category_performance(self, start_date: date, end_date: date, limit: int) -> list[dict]:
        return []

    def get_inventory(self, limit: int) -> list[dict]:
        return [
            {
                "product_id": "prod_1",
                "sku": "ELE-0001",
                "product_name": "Essential Light",
                "category_name": "Electronics",
                "supplier_name": "Demo Supplier",
                "current_stock": 4,
                "reorder_point": 10,
                "stock_status": "reorder",
            }
        ]

    def get_supplier_performance(self, start_date: date, end_date: date, limit: int) -> list[dict]:
        return []


def test_revenue_response_shapes_totals_and_average_order_value() -> None:
    service = AnalyticsService(FakeAnalyticsRepository())

    response = service.get_revenue(date(2026, 1, 1), date(2026, 1, 31))

    assert response.total_revenue_cents == 12_000
    assert response.order_count == 3
    assert response.average_order_value_cents == 4_000
    assert response.start_date == date(2026, 1, 1)
    assert response.end_date == date(2026, 1, 31)


def test_sales_trends_response_shapes_bucket_points() -> None:
    service = AnalyticsService(FakeAnalyticsRepository())

    response = service.get_sales_trends(date(2026, 1, 1), date(2026, 1, 31), "week")

    assert response.interval == "week"
    assert response.points[0].bucket_start == date(2026, 1, 1)
    assert response.points[0].average_order_value_cents == 2_500


def test_top_customers_adds_average_order_value() -> None:
    service = AnalyticsService(FakeAnalyticsRepository())

    response = service.get_top_customers(date(2026, 1, 1), date(2026, 1, 31))

    assert response.items[0].average_order_value_cents == 3_000


def test_limit_is_capped_to_maximum() -> None:
    repository = FakeAnalyticsRepository()
    service = AnalyticsService(repository)

    service.get_top_products(date(2026, 1, 1), date(2026, 1, 31), limit=250)

    assert repository.last_limit == 100


def test_invalid_date_window_raises_value_error() -> None:
    service = AnalyticsService(FakeAnalyticsRepository())

    with pytest.raises(ValueError, match="start_date"):
        service.get_revenue(date(2026, 2, 1), date(2026, 1, 1))


def test_inventory_response_allows_empty_or_ranked_rows() -> None:
    service = AnalyticsService(FakeAnalyticsRepository())

    response = service.get_inventory(limit=10)

    assert response.items[0].stock_status == "reorder"
