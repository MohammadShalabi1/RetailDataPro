from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from sqlalchemy import Select, String, case, cast, desc, distinct, func, select
from sqlalchemy.orm import Session

from app.models import Category, Customer, Order, OrderItem, Product, Supplier
from app.schemas.analytics import Interval

COMPLETED_ORDER_STATUS = "completed"


class AnalyticsRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def get_revenue(self, start_date: date, end_date: date) -> dict[str, Any]:
        statement = self.build_revenue_statement(start_date, end_date)
        row = self._db.execute(statement).one()
        return dict(row._mapping)

    def get_sales_trends(self, start_date: date, end_date: date, interval: Interval) -> list[dict[str, Any]]:
        statement = self.build_sales_trends_statement(start_date, end_date, interval)
        return self._fetch_all(statement)

    def get_top_products(self, start_date: date, end_date: date, limit: int) -> list[dict[str, Any]]:
        statement = self.build_top_products_statement(start_date, end_date, limit)
        return self._fetch_all(statement)

    def get_top_customers(self, start_date: date, end_date: date, limit: int) -> list[dict[str, Any]]:
        statement = self.build_top_customers_statement(start_date, end_date, limit)
        return self._fetch_all(statement)

    def get_category_performance(self, start_date: date, end_date: date, limit: int) -> list[dict[str, Any]]:
        statement = self.build_category_performance_statement(start_date, end_date, limit)
        return self._fetch_all(statement)

    def get_inventory(self, limit: int) -> list[dict[str, Any]]:
        statement = self.build_inventory_statement(limit)
        return self._fetch_all(statement)

    def get_supplier_performance(self, start_date: date, end_date: date, limit: int) -> list[dict[str, Any]]:
        statement = self.build_supplier_performance_statement(start_date, end_date, limit)
        return self._fetch_all(statement)

    def build_revenue_statement(self, start_date: date, end_date: date) -> Select:
        return (
            select(
                func.coalesce(func.sum(Order.total_cents), 0).label("total_revenue_cents"),
                func.count(Order.id).label("order_count"),
            )
            .select_from(Order)
            .where(*self._completed_order_filters(start_date, end_date))
        )

    def build_sales_trends_statement(self, start_date: date, end_date: date, interval: Interval) -> Select:
        bucket = func.date_trunc(interval, Order.ordered_at).label("bucket_start")
        return (
            select(
                bucket,
                func.coalesce(func.sum(Order.total_cents), 0).label("revenue_cents"),
                func.count(distinct(Order.id)).label("order_count"),
                func.coalesce(func.sum(OrderItem.quantity), 0).label("units_sold"),
            )
            .select_from(Order)
            .join(OrderItem, OrderItem.order_id == Order.id)
            .where(*self._completed_order_filters(start_date, end_date))
            .group_by(bucket)
            .order_by(bucket)
        )

    def build_top_products_statement(self, start_date: date, end_date: date, limit: int) -> Select:
        revenue = func.coalesce(func.sum(OrderItem.line_total_cents), 0)
        cost = func.coalesce(func.sum(OrderItem.unit_cost_cents * OrderItem.quantity), 0)
        return (
            select(
                cast(Product.id, String).label("product_id"),
                Product.sku,
                Product.name.label("product_name"),
                Category.name.label("category_name"),
                Supplier.name.label("supplier_name"),
                revenue.label("revenue_cents"),
                func.coalesce(func.sum(OrderItem.quantity), 0).label("units_sold"),
                func.count(distinct(Order.id)).label("order_count"),
                (revenue - cost).label("gross_margin_cents"),
            )
            .select_from(OrderItem)
            .join(Order, Order.id == OrderItem.order_id)
            .join(Product, Product.id == OrderItem.product_id)
            .join(Category, Category.id == Product.category_id)
            .join(Supplier, Supplier.id == Product.supplier_id)
            .where(*self._completed_order_filters(start_date, end_date))
            .group_by(Product.id, Product.sku, Product.name, Category.name, Supplier.name)
            .order_by(desc("revenue_cents"))
            .limit(limit)
        )

    def build_top_customers_statement(self, start_date: date, end_date: date, limit: int) -> Select:
        revenue = func.coalesce(func.sum(Order.total_cents), 0)
        return (
            select(
                cast(Customer.id, String).label("customer_id"),
                (Customer.first_name + " " + Customer.last_name).label("customer_name"),
                Customer.email,
                Customer.segment,
                revenue.label("revenue_cents"),
                func.count(Order.id).label("order_count"),
            )
            .select_from(Order)
            .join(Customer, Customer.id == Order.customer_id)
            .where(*self._completed_order_filters(start_date, end_date))
            .group_by(Customer.id, Customer.first_name, Customer.last_name, Customer.email, Customer.segment)
            .order_by(desc("revenue_cents"))
            .limit(limit)
        )

    def build_category_performance_statement(self, start_date: date, end_date: date, limit: int) -> Select:
        revenue = func.coalesce(func.sum(OrderItem.line_total_cents), 0)
        cost = func.coalesce(func.sum(OrderItem.unit_cost_cents * OrderItem.quantity), 0)
        return (
            select(
                cast(Category.id, String).label("category_id"),
                Category.name.label("category_name"),
                revenue.label("revenue_cents"),
                func.coalesce(func.sum(OrderItem.quantity), 0).label("units_sold"),
                func.count(distinct(Order.id)).label("order_count"),
                (revenue - cost).label("gross_margin_cents"),
            )
            .select_from(OrderItem)
            .join(Order, Order.id == OrderItem.order_id)
            .join(Product, Product.id == OrderItem.product_id)
            .join(Category, Category.id == Product.category_id)
            .where(*self._completed_order_filters(start_date, end_date))
            .group_by(Category.id, Category.name)
            .order_by(desc("revenue_cents"))
            .limit(limit)
        )

    def build_inventory_statement(self, limit: int) -> Select:
        stock_status = case(
            (Product.current_stock <= 0, "out_of_stock"),
            (Product.current_stock <= Product.reorder_point, "reorder"),
            else_="ok",
        ).label("stock_status")
        return (
            select(
                cast(Product.id, String).label("product_id"),
                Product.sku,
                Product.name.label("product_name"),
                Category.name.label("category_name"),
                Supplier.name.label("supplier_name"),
                Product.current_stock,
                Product.reorder_point,
                stock_status,
            )
            .select_from(Product)
            .join(Category, Category.id == Product.category_id)
            .join(Supplier, Supplier.id == Product.supplier_id)
            .order_by(Product.current_stock, Product.name)
            .limit(limit)
        )

    def build_supplier_performance_statement(self, start_date: date, end_date: date, limit: int) -> Select:
        revenue = func.coalesce(func.sum(OrderItem.line_total_cents), 0)
        cost = func.coalesce(func.sum(OrderItem.unit_cost_cents * OrderItem.quantity), 0)
        return (
            select(
                cast(Supplier.id, String).label("supplier_id"),
                Supplier.name.label("supplier_name"),
                Supplier.country,
                Supplier.lead_time_days,
                Supplier.reliability_score,
                revenue.label("revenue_cents"),
                func.coalesce(func.sum(OrderItem.quantity), 0).label("units_sold"),
                func.count(distinct(Product.id)).label("product_count"),
                (revenue - cost).label("gross_margin_cents"),
            )
            .select_from(OrderItem)
            .join(Order, Order.id == OrderItem.order_id)
            .join(Product, Product.id == OrderItem.product_id)
            .join(Supplier, Supplier.id == Product.supplier_id)
            .where(*self._completed_order_filters(start_date, end_date))
            .group_by(Supplier.id, Supplier.name, Supplier.country, Supplier.lead_time_days, Supplier.reliability_score)
            .order_by(desc("revenue_cents"))
            .limit(limit)
        )

    def _fetch_all(self, statement: Select) -> list[dict[str, Any]]:
        return [dict(row._mapping) for row in self._db.execute(statement).all()]

    def _completed_order_filters(self, start_date: date, end_date: date) -> list:
        return [
            Order.status == COMPLETED_ORDER_STATUS,
            Order.ordered_at >= _start_datetime(start_date),
            Order.ordered_at < _exclusive_end_datetime(end_date),
        ]


def _start_datetime(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def _exclusive_end_datetime(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=timezone.utc) + timedelta(days=1)
