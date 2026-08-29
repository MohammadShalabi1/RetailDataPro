from app.seed.generator import build_seed_data


def test_seed_generator_creates_requested_record_counts() -> None:
    seed_data = build_seed_data(customer_count=12, product_count=15, order_count=30, month_count=3, seed=7)

    assert len(seed_data.customers) == 12
    assert len(seed_data.products) == 15
    assert len(seed_data.orders) == 30
    assert len(seed_data.categories) >= 10
    assert len(seed_data.suppliers) >= 10
    assert len(seed_data.order_items) >= 30
    assert len(seed_data.inventory_events) >= len(seed_data.products)
    assert len(seed_data.source_chunks) == len(seed_data.sources) * 3
    assert len(seed_data.messages) == len(seed_data.conversations) * 2
    assert len(seed_data.ai_traces) == len(seed_data.conversations)


def test_seed_generator_is_referentially_consistent() -> None:
    seed_data = build_seed_data(customer_count=8, product_count=12, order_count=20, month_count=2, seed=11)

    customer_ids = {customer.id for customer in seed_data.customers}
    category_ids = {category.id for category in seed_data.categories}
    supplier_ids = {supplier.id for supplier in seed_data.suppliers}
    product_ids = {product.id for product in seed_data.products}
    order_ids = {order.id for order in seed_data.orders}
    source_ids = {source.id for source in seed_data.sources}
    conversation_ids = {conversation.id for conversation in seed_data.conversations}

    assert {order.customer_id for order in seed_data.orders}.issubset(customer_ids)
    assert {product.category_id for product in seed_data.products}.issubset(category_ids)
    assert {product.supplier_id for product in seed_data.products}.issubset(supplier_ids)
    assert {item.product_id for item in seed_data.order_items}.issubset(product_ids)
    assert {item.order_id for item in seed_data.order_items}.issubset(order_ids)
    assert {event.product_id for event in seed_data.inventory_events}.issubset(product_ids)
    assert {chunk.source_id for chunk in seed_data.source_chunks}.issubset(source_ids)
    assert {message.conversation_id for message in seed_data.messages}.issubset(conversation_ids)


def test_seed_generator_is_deterministic_for_business_identifiers() -> None:
    first = build_seed_data(customer_count=5, product_count=8, order_count=10, month_count=2, seed=23)
    second = build_seed_data(customer_count=5, product_count=8, order_count=10, month_count=2, seed=23)

    assert [customer.id for customer in first.customers] == [customer.id for customer in second.customers]
    assert [customer.email for customer in first.customers] == [customer.email for customer in second.customers]
    assert [product.sku for product in first.products] == [product.sku for product in second.products]
    assert [order.order_number for order in first.orders] == [order.order_number for order in second.orders]
