from __future__ import annotations

import re
from dataclasses import dataclass

from app.ai.sql.schemas import SchemaLinkResult, SchemaTable


@dataclass(frozen=True)
class TableDescription:
    name: str
    description: str
    columns: tuple[str, ...]
    keywords: tuple[str, ...]


RETAIL_SCHEMA: tuple[TableDescription, ...] = (
    TableDescription("customers", "Retail customers and segments.", ("id", "email", "first_name", "last_name", "segment"), ("customer", "customers", "segment", "buyer", "spending")),
    TableDescription("orders", "Orders with status, channel, ordered timestamp, and totals.", ("id", "customer_id", "ordered_at", "status", "channel", "total_cents"), ("order", "orders", "revenue", "sales", "quarter", "month", "channel")),
    TableDescription("order_items", "Line items connecting products to orders.", ("id", "order_id", "product_id", "quantity", "line_total_cents"), ("item", "items", "units", "quantity", "product", "products", "margin")),
    TableDescription("products", "Product catalog with SKU, category, supplier, price, and stock.", ("id", "sku", "name", "category_id", "supplier_id", "unit_price_cents"), ("product", "products", "sku", "price", "stock")),
    TableDescription("categories", "Product categories and descriptions.", ("id", "name", "description"), ("category", "categories", "department")),
    TableDescription("suppliers", "Supplier attributes and reliability.", ("id", "name", "country", "lead_time_days", "reliability_score"), ("supplier", "suppliers", "vendor", "lead time", "reliability")),
    TableDescription("inventory_events", "Inventory adjustments over time.", ("id", "product_id", "event_type", "quantity_delta", "occurred_at"), ("inventory", "stock", "reorder", "shipment")),
)


class SchemaLinker:
    def __init__(self, tables: tuple[TableDescription, ...] = RETAIL_SCHEMA) -> None:
        self._tables = tables

    def link(self, question: str, embedding_scores: dict[str, float] | None = None, max_tables: int = 4) -> SchemaLinkResult:
        tokens = set(_tokens(question))
        scored: list[SchemaTable] = []
        for table in self._tables:
            lexical_score = _lexical_score(tokens, table)
            embedding_score = (embedding_scores or {}).get(table.name, 0.0)
            score = min(1.0, max(lexical_score, embedding_score))
            if score > 0:
                scored.append(
                    SchemaTable(
                        name=table.name,
                        description=table.description,
                        columns=list(table.columns),
                        score=round(score, 3),
                    )
                )

        scored.sort(key=lambda item: item.score, reverse=True)
        selected = _expand_required_join_tables(scored[:max_tables])
        confidence = max((table.score for table in selected), default=0.0)
        return SchemaLinkResult(tables=selected[:max_tables], confidence=confidence)


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", value.lower())


def _lexical_score(tokens: set[str], table: TableDescription) -> float:
    haystack = set(_tokens(" ".join((table.name, table.description, *table.columns, *table.keywords))))
    matches = tokens & haystack
    if not matches:
        return 0.0
    return min(1.0, 0.35 + (len(matches) * 0.15))


def _expand_required_join_tables(selected: list[SchemaTable]) -> list[SchemaTable]:
    names = {table.name for table in selected}
    lookup = {table.name: table for table in selected}
    known = {table.name: table for table in _tables_as_schema()}
    if "order_items" in names:
        for name in ("orders", "products"):
            lookup.setdefault(name, known[name])
    if "products" in names:
        for name in ("categories", "suppliers"):
            lookup.setdefault(name, known[name])
    if "orders" in names:
        lookup.setdefault("customers", known["customers"])
    return list(lookup.values())


def _tables_as_schema() -> list[SchemaTable]:
    return [
        SchemaTable(name=table.name, description=table.description, columns=list(table.columns), score=0.4)
        for table in RETAIL_SCHEMA
    ]
