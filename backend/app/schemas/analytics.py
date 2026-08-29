from datetime import date
from typing import Literal

from pydantic import BaseModel


Interval = Literal["day", "week", "month"]


class DateWindow(BaseModel):
    start_date: date
    end_date: date


class RevenueResponse(DateWindow):
    total_revenue_cents: int
    order_count: int
    average_order_value_cents: int


class SalesTrendPoint(BaseModel):
    bucket_start: date
    revenue_cents: int
    order_count: int
    units_sold: int
    average_order_value_cents: int


class SalesTrendsResponse(DateWindow):
    interval: Interval
    points: list[SalesTrendPoint]


class TopProductRow(BaseModel):
    product_id: str
    sku: str
    product_name: str
    category_name: str
    supplier_name: str
    revenue_cents: int
    units_sold: int
    order_count: int
    gross_margin_cents: int


class TopProductsResponse(DateWindow):
    items: list[TopProductRow]


class TopCustomerRow(BaseModel):
    customer_id: str
    customer_name: str
    email: str
    segment: str
    revenue_cents: int
    order_count: int
    average_order_value_cents: int


class TopCustomersResponse(DateWindow):
    items: list[TopCustomerRow]


class CategoryPerformanceRow(BaseModel):
    category_id: str
    category_name: str
    revenue_cents: int
    units_sold: int
    order_count: int
    gross_margin_cents: int


class CategoryPerformanceResponse(DateWindow):
    items: list[CategoryPerformanceRow]


class InventoryRow(BaseModel):
    product_id: str
    sku: str
    product_name: str
    category_name: str
    supplier_name: str
    current_stock: int
    reorder_point: int
    stock_status: Literal["ok", "reorder", "out_of_stock"]


class InventoryResponse(BaseModel):
    items: list[InventoryRow]


class SupplierPerformanceRow(BaseModel):
    supplier_id: str
    supplier_name: str
    country: str
    lead_time_days: int
    reliability_score: int
    revenue_cents: int
    units_sold: int
    product_count: int
    gross_margin_cents: int


class SupplierPerformanceResponse(DateWindow):
    items: list[SupplierPerformanceRow]
