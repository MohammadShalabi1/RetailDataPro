from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from faker import Faker

from app.models import (
    AITrace,
    Category,
    Conversation,
    Customer,
    InventoryEvent,
    Message,
    Order,
    OrderItem,
    Product,
    Source,
    SourceChunk,
    Supplier,
)


DEFAULT_CUSTOMER_COUNT = 500
DEFAULT_PRODUCT_COUNT = 120
DEFAULT_ORDER_COUNT = 4_000
DEFAULT_MONTH_COUNT = 9
DEFAULT_SEED = 42
DEFAULT_ANCHOR_AT = datetime(2026, 8, 29, tzinfo=timezone.utc)

CATEGORY_NAMES = [
    "Electronics",
    "Home Office",
    "Kitchen",
    "Furniture",
    "Apparel",
    "Footwear",
    "Beauty",
    "Outdoor",
    "Toys",
    "Grocery",
]

CHANNELS = ["web", "mobile_app", "marketplace", "store"]
CUSTOMER_SEGMENTS = ["new", "returning", "loyal", "vip"]
ORDER_STATUSES = ["completed", "completed", "completed", "completed", "returned", "cancelled"]
INVENTORY_EVENT_TYPES = ["restock", "sale", "return", "adjustment"]


@dataclass(frozen=True)
class SeedData:
    customers: list[Customer]
    categories: list[Category]
    suppliers: list[Supplier]
    products: list[Product]
    orders: list[Order]
    order_items: list[OrderItem]
    inventory_events: list[InventoryEvent]
    sources: list[Source]
    source_chunks: list[SourceChunk]
    conversations: list[Conversation]
    messages: list[Message]
    ai_traces: list[AITrace]


def build_seed_data(
    customer_count: int = DEFAULT_CUSTOMER_COUNT,
    product_count: int = DEFAULT_PRODUCT_COUNT,
    order_count: int = DEFAULT_ORDER_COUNT,
    month_count: int = DEFAULT_MONTH_COUNT,
    seed: int = DEFAULT_SEED,
) -> SeedData:
    if customer_count < 1 or product_count < 1 or order_count < 1 or month_count < 1:
        raise ValueError("customer_count, product_count, order_count, and month_count must be positive")

    rng = random.Random(seed)
    fake = Faker("en_US")
    Faker.seed(seed)

    now = DEFAULT_ANCHOR_AT
    start_at = now - timedelta(days=month_count * 30)

    categories = _build_categories(rng)
    suppliers = _build_suppliers(fake, rng)
    customers = _build_customers(fake, rng, customer_count)
    products = _build_products(fake, rng, product_count, categories, suppliers)
    orders, order_items = _build_orders(fake, rng, order_count, start_at, now, customers, products)
    inventory_events = _build_inventory_events(rng, products, order_items, start_at, now)
    sources, source_chunks = _build_sources_and_chunks(rng, now, categories, suppliers)
    conversations, messages, ai_traces = _build_conversations(rng, now, customers)

    return SeedData(
        customers=customers,
        categories=categories,
        suppliers=suppliers,
        products=products,
        orders=orders,
        order_items=order_items,
        inventory_events=inventory_events,
        sources=sources,
        source_chunks=source_chunks,
        conversations=conversations,
        messages=messages,
        ai_traces=ai_traces,
    )


def _build_categories(rng: random.Random) -> list[Category]:
    return [
        Category(
            id=_make_uuid(rng),
            name=name,
            description=f"Retail products and operational metrics for {name.lower()} performance.",
        )
        for name in CATEGORY_NAMES
    ]


def _build_suppliers(fake: Faker, rng: random.Random) -> list[Supplier]:
    suppliers: list[Supplier] = []
    for index in range(16):
        company = fake.unique.company()
        suppliers.append(
            Supplier(
                id=_make_uuid(rng),
                name=company,
                contact_email=f"supply-{index + 1}@{fake.domain_name()}",
                country=fake.country(),
                lead_time_days=rng.randint(4, 28),
                reliability_score=rng.randint(72, 99),
            )
        )
    return suppliers


def _build_customers(fake: Faker, rng: random.Random, count: int) -> list[Customer]:
    customers: list[Customer] = []
    for index in range(count):
        first_name = fake.first_name()
        last_name = fake.last_name()
        customers.append(
            Customer(
                id=_make_uuid(rng),
                email=f"{first_name}.{last_name}.{index}@example-retail.test".lower(),
                first_name=first_name,
                last_name=last_name,
                city=fake.city(),
                state=fake.state_abbr(),
                segment=rng.choices(CUSTOMER_SEGMENTS, weights=[25, 42, 25, 8], k=1)[0],
                acquired_channel=rng.choice(["organic_search", "paid_social", "email", "referral", "store"]),
            )
        )
    return customers


def _build_products(
    fake: Faker,
    rng: random.Random,
    count: int,
    categories: list[Category],
    suppliers: list[Supplier],
) -> list[Product]:
    products: list[Product] = []
    adjectives = ["Essential", "Prime", "Urban", "Classic", "Pro", "Eco", "Signature", "Daily"]
    nouns = ["Kit", "Pack", "Station", "Set", "Case", "Stand", "Blend", "Layer", "Console", "Light"]

    for index in range(count):
        category = categories[index % len(categories)]
        supplier = suppliers[index % len(suppliers)]
        unit_cost = rng.randint(450, 21_000)
        margin = rng.uniform(1.25, 2.4)
        unit_price = int(unit_cost * margin)
        products.append(
            Product(
                id=_make_uuid(rng),
                category_id=category.id,
                supplier_id=supplier.id,
                sku=f"{category.name[:3].upper()}-{index + 1:04d}",
                name=f"{rng.choice(adjectives)} {fake.color_name()} {rng.choice(nouns)}",
                unit_price_cents=unit_price,
                unit_cost_cents=unit_cost,
                current_stock=rng.randint(20, 380),
                reorder_point=rng.randint(12, 80),
            )
        )
    return products


def _build_orders(
    fake: Faker,
    rng: random.Random,
    count: int,
    start_at: datetime,
    end_at: datetime,
    customers: list[Customer],
    products: list[Product],
) -> tuple[list[Order], list[OrderItem]]:
    orders: list[Order] = []
    order_items: list[OrderItem] = []
    seconds_range = int((end_at - start_at).total_seconds())

    for index in range(count):
        ordered_at = start_at + timedelta(seconds=rng.randint(0, seconds_range))
        customer = rng.choice(customers)
        selected_products = rng.sample(products, k=rng.randint(1, min(5, len(products))))
        status = rng.choice(ORDER_STATUSES)
        subtotal = 0
        item_rows: list[OrderItem] = []

        for product in selected_products:
            quantity = rng.choices([1, 2, 3, 4, 5], weights=[48, 28, 14, 7, 3], k=1)[0]
            discount = int(product.unit_price_cents * quantity * rng.choice([0, 0, 0, 0.05, 0.1]))
            line_total = product.unit_price_cents * quantity - discount
            subtotal += line_total
            item_rows.append(
                OrderItem(
                    id=_make_uuid(rng),
                    product_id=product.id,
                    quantity=quantity,
                    unit_price_cents=product.unit_price_cents,
                    unit_cost_cents=product.unit_cost_cents,
                    discount_cents=discount,
                    line_total_cents=line_total,
                )
            )

        discount_cents = int(subtotal * rng.choice([0, 0, 0.03, 0.07]))
        shipping_cents = 0 if subtotal > 7_500 else rng.choice([499, 799, 999])
        tax_cents = int((subtotal - discount_cents) * 0.0725)
        total_cents = subtotal - discount_cents + shipping_cents + tax_cents
        order = Order(
            id=_make_uuid(rng),
            customer_id=customer.id,
            order_number=f"RDP-{ordered_at:%Y%m%d}-{index + 1:06d}",
            ordered_at=ordered_at,
            status=status,
            channel=rng.choice(CHANNELS),
            subtotal_cents=subtotal,
            discount_cents=discount_cents,
            shipping_cents=shipping_cents,
            tax_cents=tax_cents,
            total_cents=total_cents,
        )

        for item in item_rows:
            item.order_id = order.id
        orders.append(order)
        order_items.extend(item_rows)

    orders.sort(key=lambda order: order.ordered_at)
    return orders, order_items


def _build_inventory_events(
    rng: random.Random,
    products: list[Product],
    order_items: list[OrderItem],
    start_at: datetime,
    end_at: datetime,
) -> list[InventoryEvent]:
    events: list[InventoryEvent] = []
    seconds_range = int((end_at - start_at).total_seconds())

    for product in products:
        events.append(
            InventoryEvent(
                id=_make_uuid(rng),
                product_id=product.id,
                event_type="restock",
                quantity_delta=rng.randint(80, 360),
                occurred_at=start_at + timedelta(seconds=rng.randint(0, seconds_range)),
                reason="Initial seasonal replenishment",
            )
        )

    for item in order_items[: min(len(order_items), 1_200)]:
        events.append(
            InventoryEvent(
                id=_make_uuid(rng),
                product_id=item.product_id,
                event_type=rng.choices(INVENTORY_EVENT_TYPES, weights=[8, 70, 12, 10], k=1)[0],
                quantity_delta=-item.quantity,
                occurred_at=start_at + timedelta(seconds=rng.randint(0, seconds_range)),
                reason="Order activity",
            )
        )

    return events


def _build_sources_and_chunks(
    rng: random.Random,
    now: datetime,
    categories: list[Category],
    suppliers: list[Supplier],
) -> tuple[list[Source], list[SourceChunk]]:
    sources: list[Source] = []
    chunks: list[SourceChunk] = []

    for index, supplier in enumerate(suppliers[:6]):
        category = categories[index % len(categories)]
        source = Source(
            id=_make_uuid(rng),
            title=f"Q{rng.randint(1, 4)} supplier brief - {supplier.name}",
            source_type="supplier_report",
            uri=f"s3://retaildata-pro-demo/supplier-report-{index + 1}.pdf",
            uploaded_at=now - timedelta(days=rng.randint(5, 150)),
            metadata_={"supplier_id": str(supplier.id), "category": category.name},
        )
        sources.append(source)

        chunk_texts = [
            f"{supplier.name} reports lead times for {category.name} are trending near {supplier.lead_time_days} days.",
            f"Demand planning notes mention promotion timing, stock coverage, and margin pressure in {category.name}.",
            "Operations should monitor late shipments, return rates, and replenishment windows before major campaigns.",
        ]
        for chunk_index, content in enumerate(chunk_texts):
            chunks.append(
                SourceChunk(
                    id=_make_uuid(rng),
                    source_id=source.id,
                    chunk_index=chunk_index,
                    content=content,
                    token_count=max(8, len(content.split())),
                    metadata_={"section": "supplier_signal", "quarter": f"Q{rng.randint(1, 4)}"},
                )
            )

    return sources, chunks


def _build_conversations(
    rng: random.Random,
    now: datetime,
    customers: list[Customer],
) -> tuple[list[Conversation], list[Message], list[AITrace]]:
    conversations: list[Conversation] = []
    messages: list[Message] = []
    traces: list[AITrace] = []

    for index in range(8):
        customer = rng.choice(customers)
        conversation = Conversation(
            id=_make_uuid(rng),
            customer_id=customer.id,
            title=f"Retail analysis session {index + 1}",
            summary="Seeded conversation for future memory and trace viewer development.",
        )
        conversations.append(conversation)
        user_message = Message(
            id=_make_uuid(rng),
            conversation_id=conversation.id,
            role="user",
            content="Which categories changed most this month?",
            metadata_={"seeded": True},
        )
        assistant_message = Message(
            id=_make_uuid(rng),
            conversation_id=conversation.id,
            role="assistant",
            content="This seeded placeholder will be replaced by grounded AI responses in later phases.",
            metadata_={"seeded": True},
        )
        messages.extend([user_message, assistant_message])
        traces.append(
            AITrace(
                id=_make_uuid(rng),
                conversation_id=conversation.id,
                trace_id=f"seed-trace-{index + 1:03d}",
                route="seeded",
                model_name="none",
                latency_ms=0,
                input_tokens=0,
                output_tokens=0,
                tool_calls=[],
                metadata_={"seeded": True, "created_at": now.isoformat()},
            )
        )

    return conversations, messages, traces


def _make_uuid(rng: random.Random) -> uuid.UUID:
    return uuid.UUID(int=rng.getrandbits(128), version=4)
