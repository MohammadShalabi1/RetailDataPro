from app import models  # noqa: F401
from app.database.base import Base


def test_phase_two_tables_are_registered() -> None:
    expected_tables = {
        "customers",
        "categories",
        "products",
        "orders",
        "order_items",
        "suppliers",
        "inventory_events",
        "sources",
        "source_chunks",
        "conversations",
        "messages",
        "ai_traces",
    }

    assert expected_tables.issubset(set(Base.metadata.tables))


def test_core_relationship_foreign_keys_are_present() -> None:
    metadata = Base.metadata

    assert "customers.id" in _foreign_key_targets(metadata.tables["orders"])
    assert "products.id" in _foreign_key_targets(metadata.tables["order_items"])
    assert "orders.id" in _foreign_key_targets(metadata.tables["order_items"])
    assert "categories.id" in _foreign_key_targets(metadata.tables["products"])
    assert "suppliers.id" in _foreign_key_targets(metadata.tables["products"])
    assert "sources.id" in _foreign_key_targets(metadata.tables["source_chunks"])
    assert "conversations.id" in _foreign_key_targets(metadata.tables["messages"])
    assert "conversations.id" in _foreign_key_targets(metadata.tables["ai_traces"])


def test_analytics_indexes_are_present() -> None:
    metadata = Base.metadata

    assert "ix_orders_customer_ordered_at" in _index_names(metadata.tables["orders"])
    assert "ix_order_items_product_order" in _index_names(metadata.tables["order_items"])
    assert "ix_inventory_events_product_occurred_at" in _index_names(metadata.tables["inventory_events"])
    assert "ix_source_chunks_source_chunk" in _index_names(metadata.tables["source_chunks"])
    assert "ix_ai_traces_route_model" in _index_names(metadata.tables["ai_traces"])


def _foreign_key_targets(table) -> set[str]:
    return {str(foreign_key.column) for foreign_key in table.foreign_keys}


def _index_names(table) -> set[str]:
    return {index.name for index in table.indexes}
