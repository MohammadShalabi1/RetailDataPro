from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.schemas.analytics import (
    CategoryPerformanceResponse,
    Interval,
    InventoryResponse,
    RevenueResponse,
    SalesTrendsResponse,
    SupplierPerformanceResponse,
    TopCustomersResponse,
    TopProductsResponse,
)
from app.services.analytics_service import AnalyticsService, get_analytics_service

router = APIRouter(prefix="/analytics", tags=["analytics"])

LimitQuery = Annotated[int, Query(ge=1, le=100)]


@router.get("/revenue", response_model=RevenueResponse)
def get_revenue(
    start_date: date | None = None,
    end_date: date | None = None,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> RevenueResponse:
    return _handle_validation(lambda: analytics_service.get_revenue(start_date, end_date))


@router.get("/sales-trends", response_model=SalesTrendsResponse)
def get_sales_trends(
    start_date: date | None = None,
    end_date: date | None = None,
    interval: Interval = "month",
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> SalesTrendsResponse:
    return _handle_validation(lambda: analytics_service.get_sales_trends(start_date, end_date, interval))


@router.get("/top-products", response_model=TopProductsResponse)
def get_top_products(
    start_date: date | None = None,
    end_date: date | None = None,
    limit: LimitQuery = 10,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> TopProductsResponse:
    return _handle_validation(lambda: analytics_service.get_top_products(start_date, end_date, limit))


@router.get("/top-customers", response_model=TopCustomersResponse)
def get_top_customers(
    start_date: date | None = None,
    end_date: date | None = None,
    limit: LimitQuery = 10,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> TopCustomersResponse:
    return _handle_validation(lambda: analytics_service.get_top_customers(start_date, end_date, limit))


@router.get("/category-performance", response_model=CategoryPerformanceResponse)
def get_category_performance(
    start_date: date | None = None,
    end_date: date | None = None,
    limit: LimitQuery = 10,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> CategoryPerformanceResponse:
    return _handle_validation(lambda: analytics_service.get_category_performance(start_date, end_date, limit))


@router.get("/inventory", response_model=InventoryResponse)
def get_inventory(
    limit: LimitQuery = 10,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> InventoryResponse:
    return _handle_validation(lambda: analytics_service.get_inventory(limit))


@router.get("/supplier-performance", response_model=SupplierPerformanceResponse)
def get_supplier_performance(
    start_date: date | None = None,
    end_date: date | None = None,
    limit: LimitQuery = 10,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
) -> SupplierPerformanceResponse:
    return _handle_validation(lambda: analytics_service.get_supplier_performance(start_date, end_date, limit))


def _handle_validation(call):
    try:
        return call()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
