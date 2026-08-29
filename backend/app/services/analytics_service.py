from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from fastapi import Depends
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.repositories.analytics_repository import AnalyticsRepository
from app.schemas.analytics import (
    CategoryPerformanceResponse,
    DateWindow,
    Interval,
    InventoryResponse,
    InventoryRow,
    RevenueResponse,
    SalesTrendPoint,
    SalesTrendsResponse,
    SupplierPerformanceResponse,
    TopCustomersResponse,
    TopProductsResponse,
)

DEFAULT_WINDOW_DAYS = 90
DEFAULT_LIMIT = 10
MAX_LIMIT = 100


class AnalyticsService:
    def __init__(self, analytics_repository: AnalyticsRepository) -> None:
        self._analytics_repository = analytics_repository

    def get_revenue(self, start_date: date | None = None, end_date: date | None = None) -> RevenueResponse:
        window = self._resolve_window(start_date, end_date)
        row = self._analytics_repository.get_revenue(window.start_date, window.end_date)
        total_revenue = _int_value(row.get("total_revenue_cents"))
        order_count = _int_value(row.get("order_count"))
        return RevenueResponse(
            start_date=window.start_date,
            end_date=window.end_date,
            total_revenue_cents=total_revenue,
            order_count=order_count,
            average_order_value_cents=_average(total_revenue, order_count),
        )

    def get_sales_trends(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        interval: Interval = "month",
    ) -> SalesTrendsResponse:
        window = self._resolve_window(start_date, end_date)
        rows = self._analytics_repository.get_sales_trends(window.start_date, window.end_date, interval)
        points = [
            SalesTrendPoint(
                bucket_start=_date_value(row["bucket_start"]),
                revenue_cents=_int_value(row.get("revenue_cents")),
                order_count=_int_value(row.get("order_count")),
                units_sold=_int_value(row.get("units_sold")),
                average_order_value_cents=_average(
                    _int_value(row.get("revenue_cents")),
                    _int_value(row.get("order_count")),
                ),
            )
            for row in rows
        ]
        return SalesTrendsResponse(
            start_date=window.start_date,
            end_date=window.end_date,
            interval=interval,
            points=points,
        )

    def get_top_products(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> TopProductsResponse:
        window = self._resolve_window(start_date, end_date)
        rows = self._analytics_repository.get_top_products(window.start_date, window.end_date, self._resolve_limit(limit))
        return TopProductsResponse(start_date=window.start_date, end_date=window.end_date, items=rows)

    def get_top_customers(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> TopCustomersResponse:
        window = self._resolve_window(start_date, end_date)
        rows = self._analytics_repository.get_top_customers(window.start_date, window.end_date, self._resolve_limit(limit))
        items = [
            {
                **row,
                "average_order_value_cents": _average(
                    _int_value(row.get("revenue_cents")),
                    _int_value(row.get("order_count")),
                ),
            }
            for row in rows
        ]
        return TopCustomersResponse(start_date=window.start_date, end_date=window.end_date, items=items)

    def get_category_performance(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> CategoryPerformanceResponse:
        window = self._resolve_window(start_date, end_date)
        rows = self._analytics_repository.get_category_performance(
            window.start_date,
            window.end_date,
            self._resolve_limit(limit),
        )
        return CategoryPerformanceResponse(start_date=window.start_date, end_date=window.end_date, items=rows)

    def get_inventory(self, limit: int = DEFAULT_LIMIT) -> InventoryResponse:
        rows = self._analytics_repository.get_inventory(self._resolve_limit(limit))
        return InventoryResponse(items=[InventoryRow(**row) for row in rows])

    def get_supplier_performance(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> SupplierPerformanceResponse:
        window = self._resolve_window(start_date, end_date)
        rows = self._analytics_repository.get_supplier_performance(
            window.start_date,
            window.end_date,
            self._resolve_limit(limit),
        )
        return SupplierPerformanceResponse(start_date=window.start_date, end_date=window.end_date, items=rows)

    def _resolve_window(self, start_date: date | None, end_date: date | None) -> DateWindow:
        resolved_end = end_date or date.today()
        resolved_start = start_date or (resolved_end - timedelta(days=DEFAULT_WINDOW_DAYS))
        if resolved_start > resolved_end:
            raise ValueError("start_date must be before or equal to end_date")
        return DateWindow(start_date=resolved_start, end_date=resolved_end)

    def _resolve_limit(self, limit: int) -> int:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        return min(limit, MAX_LIMIT)


def get_analytics_service(db: Session = Depends(get_db)) -> AnalyticsService:
    return AnalyticsService(AnalyticsRepository(db))


def _average(total: int, count: int) -> int:
    if count == 0:
        return 0
    return total // count


def _int_value(value: Any) -> int:
    return int(value or 0)


def _date_value(value: Any) -> date:
    if isinstance(value, date):
        return value
    return value.date()
