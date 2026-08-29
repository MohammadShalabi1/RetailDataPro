from __future__ import annotations

import argparse

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.session import SessionLocal
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
from app.seed.generator import (
    DEFAULT_CUSTOMER_COUNT,
    DEFAULT_MONTH_COUNT,
    DEFAULT_ORDER_COUNT,
    DEFAULT_PRODUCT_COUNT,
    DEFAULT_SEED,
    SeedData,
    build_seed_data,
)


DELETE_ORDER = [
    AITrace,
    Message,
    Conversation,
    SourceChunk,
    Source,
    InventoryEvent,
    OrderItem,
    Order,
    Product,
    Category,
    Supplier,
    Customer,
]


def main() -> None:
    args = parse_args()
    settings = get_settings()

    if not settings.is_development and not args.allow_production_reset:
        raise SystemExit("Refusing to reset seed data outside development without --allow-production-reset.")

    seed_data = build_seed_data(
        customer_count=args.customers,
        product_count=args.products,
        order_count=args.orders,
        month_count=args.months,
        seed=args.seed,
    )

    with SessionLocal() as session:
        reset_database(session)
        insert_seed_data(session, seed_data)
        session.commit()

    print(
        "Seeded "
        f"{len(seed_data.customers)} customers, "
        f"{len(seed_data.products)} products, "
        f"{len(seed_data.orders)} orders, "
        f"{len(seed_data.order_items)} order items."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed RetailData-Pro development data.")
    parser.add_argument("--customers", type=int, default=DEFAULT_CUSTOMER_COUNT)
    parser.add_argument("--products", type=int, default=DEFAULT_PRODUCT_COUNT)
    parser.add_argument("--orders", type=int, default=DEFAULT_ORDER_COUNT)
    parser.add_argument("--months", type=int, default=DEFAULT_MONTH_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--allow-production-reset",
        action="store_true",
        help="Explicitly allow destructive reset outside APP_ENV=development.",
    )
    return parser.parse_args()


def reset_database(session: Session) -> None:
    for model in DELETE_ORDER:
        session.query(model).delete()


def insert_seed_data(session: Session, seed_data: SeedData) -> None:
    session.add_all(seed_data.categories)
    session.add_all(seed_data.suppliers)
    session.add_all(seed_data.customers)
    session.add_all(seed_data.products)
    session.add_all(seed_data.orders)
    session.add_all(seed_data.order_items)
    session.add_all(seed_data.inventory_events)
    session.add_all(seed_data.sources)
    session.add_all(seed_data.source_chunks)
    session.add_all(seed_data.conversations)
    session.add_all(seed_data.messages)
    session.add_all(seed_data.ai_traces)


if __name__ == "__main__":
    main()
