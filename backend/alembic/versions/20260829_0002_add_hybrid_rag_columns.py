"""add hybrid rag columns

Revision ID: 20260829_0002
Revises: 20260829_0001
Create Date: 2026-08-29 00:02:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from app.database.types import PGVector


revision: str = "20260829_0002"
down_revision: Union[str, None] = "20260829_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.add_column("source_chunks", sa.Column("embedding", PGVector(768), nullable=True))
    op.add_column("source_chunks", sa.Column("search_vector", postgresql.TSVECTOR(), nullable=True))
    op.create_index("ix_source_chunks_search_vector", "source_chunks", ["search_vector"], unique=False, postgresql_using="gin")
    op.create_index(
        "ix_source_chunks_embedding",
        "source_chunks",
        ["embedding"],
        unique=False,
        postgresql_using="ivfflat",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_source_chunks_embedding", table_name="source_chunks")
    op.drop_index("ix_source_chunks_search_vector", table_name="source_chunks")
    op.drop_column("source_chunks", "search_vector")
    op.drop_column("source_chunks", "embedding")
