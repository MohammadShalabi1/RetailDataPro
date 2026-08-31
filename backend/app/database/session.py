from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
)
readonly_engine = (
    create_engine(
        settings.readonly_database_url,
        pool_pre_ping=True,
    )
    if settings.readonly_database_configured and settings.readonly_database_url is not None
    else None
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)
ReadonlySessionLocal = (
    sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=readonly_engine,
    )
    if readonly_engine is not None
    else None
)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_readonly_db() -> Generator[Session | None, None, None]:
    if ReadonlySessionLocal is None:
        yield None
        return
    db = ReadonlySessionLocal()
    try:
        yield db
    finally:
        db.close()
