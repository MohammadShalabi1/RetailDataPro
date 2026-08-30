"""add client scope and conversation activity

Revision ID: 20260829_0003
Revises: 20260829_0002
Create Date: 2026-08-29 00:03:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20260829_0003"
down_revision: Union[str, None] = "20260829_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("sources", sa.Column("client_id", sa.String(length=80), nullable=False, server_default="single-client"))
    op.add_column("conversations", sa.Column("client_id", sa.String(length=80), nullable=False, server_default="single-client"))
    op.add_column("conversations", sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True))
    op.execute("UPDATE conversations SET last_message_at = updated_at WHERE last_message_at IS NULL")
    op.create_index(op.f("ix_sources_client_id"), "sources", ["client_id"], unique=False)
    op.create_index("ix_sources_client_type_uploaded", "sources", ["client_id", "source_type", "uploaded_at"], unique=False)
    op.create_index(op.f("ix_conversations_client_id"), "conversations", ["client_id"], unique=False)
    op.create_index(op.f("ix_conversations_last_message_at"), "conversations", ["last_message_at"], unique=False)
    op.create_index("ix_conversations_client_last_message", "conversations", ["client_id", "last_message_at"], unique=False)
    op.alter_column("sources", "client_id", server_default=None)
    op.alter_column("conversations", "client_id", server_default=None)


def downgrade() -> None:
    op.drop_index("ix_conversations_client_last_message", table_name="conversations")
    op.drop_index(op.f("ix_conversations_last_message_at"), table_name="conversations")
    op.drop_index(op.f("ix_conversations_client_id"), table_name="conversations")
    op.drop_index("ix_sources_client_type_uploaded", table_name="sources")
    op.drop_index(op.f("ix_sources_client_id"), table_name="sources")
    op.drop_column("conversations", "last_message_at")
    op.drop_column("conversations", "client_id")
    op.drop_column("sources", "client_id")
