from datetime import date

from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.analytics import (
    InventoryResponse,
    RevenueResponse,
    SalesTrendsResponse,
    TopProductsResponse,
)
from app.services.analytics_service import get_analytics_service


class FakeAnalyticsService:
    def get_revenue(self, start_date=None, end_date=None) -> RevenueResponse:
        return RevenueResponse(
            start_date=start_date or date(2026, 1, 1),
            end_date=end_date or date(2026, 1, 31),
            total_revenue_cents=10_000,
            order_count=2,
            average_order_value_cents=5_000,
        )

    def get_sales_trends(self, start_date=None, end_date=None, interval="month") -> SalesTrendsResponse:
        return SalesTrendsResponse(
            start_date=start_date or date(2026, 1, 1),
            end_date=end_date or date(2026, 1, 31),
            interval=interval,
            points=[],
        )

    def get_top_products(self, start_date=None, end_date=None, limit=10) -> TopProductsResponse:
        return TopProductsResponse(
            start_date=start_date or date(2026, 1, 1),
            end_date=end_date or date(2026, 1, 31),
            items=[],
        )

    def get_top_customers(self, start_date=None, end_date=None, limit=10):
        return {
            "start_date": start_date or date(2026, 1, 1),
            "end_date": end_date or date(2026, 1, 31),
            "items": [],
        }

    def get_category_performance(self, start_date=None, end_date=None, limit=10):
        return {
            "start_date": start_date or date(2026, 1, 1),
            "end_date": end_date or date(2026, 1, 31),
            "items": [],
        }

    def get_inventory(self, limit=10) -> InventoryResponse:
        return InventoryResponse(items=[])

    def get_supplier_performance(self, start_date=None, end_date=None, limit=10):
        return {
            "start_date": start_date or date(2026, 1, 1),
            "end_date": end_date or date(2026, 1, 31),
            "items": [],
        }


def test_analytics_revenue_route_contract() -> None:
    client = _client()

    response = client.get("/api/analytics/revenue?start_date=2026-01-01&end_date=2026-01-31")

    assert response.status_code == 200
    assert response.json()["total_revenue_cents"] == 10_000
    assert response.json()["start_date"] == "2026-01-01"


def test_analytics_routes_are_registered() -> None:
    client = _client()

    paths = [
        "/api/analytics/sales-trends",
        "/api/analytics/top-products",
        "/api/analytics/top-customers",
        "/api/analytics/category-performance",
        "/api/analytics/inventory",
        "/api/analytics/supplier-performance",
    ]

    for path in paths:
        response = client.get(path)
        assert response.status_code == 200


def test_invalid_limit_returns_validation_error() -> None:
    client = _client()

    response = client.get("/api/analytics/top-products?limit=101")

    assert response.status_code == 422


def _client() -> TestClient:
    app = create_app()
    app.dependency_overrides[get_analytics_service] = lambda: FakeAnalyticsService()
    return TestClient(app)
