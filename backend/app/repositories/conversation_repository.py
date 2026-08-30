from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session, selectinload

from app.models import AITrace, Conversation, Message


class ConversationRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create_conversation(self, client_id: str, title: str = "New chat") -> Conversation:
        now = datetime.now(timezone.utc)
        conversation = Conversation(client_id=client_id, title=title, last_message_at=now)
        self._db.add(conversation)
        self._db.commit()
        self._db.refresh(conversation)
        return conversation

    def list_conversations(self, client_id: str) -> list[Conversation]:
        return list(
            self._db.scalars(
                select(Conversation)
                .where(Conversation.client_id == client_id)
                .order_by(desc(Conversation.last_message_at), desc(Conversation.updated_at))
            ).all()
        )

    def get_conversation(self, client_id: str, conversation_id: str) -> Conversation | None:
        parsed_id = _parse_uuid(conversation_id)
        if parsed_id is None:
            return None
        return self._db.scalar(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.client_id == client_id, Conversation.id == parsed_id)
        )

    def get_messages(self, client_id: str, conversation_id: str) -> list[Message] | None:
        conversation = self.get_conversation(client_id, conversation_id)
        if conversation is None:
            return None
        return sorted(conversation.messages, key=lambda message: message.created_at)

    def get_recent_messages(self, client_id: str, conversation_id: str, limit: int = 12) -> list[Message]:
        parsed_id = _parse_uuid(conversation_id)
        if parsed_id is None:
            return []
        rows = self._db.scalars(
            select(Message)
            .join(Conversation, Conversation.id == Message.conversation_id)
            .where(Conversation.client_id == client_id, Conversation.id == parsed_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        ).all()
        return list(reversed(rows))

    def count_messages(self, client_id: str, conversation_id: str) -> int:
        parsed_id = _parse_uuid(conversation_id)
        if parsed_id is None:
            return 0
        return int(
            self._db.scalar(
                select(func.count(Message.id))
                .join(Conversation, Conversation.id == Message.conversation_id)
                .where(Conversation.client_id == client_id, Conversation.id == parsed_id)
            )
            or 0
        )

    def add_message(
        self,
        client_id: str,
        conversation_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message | None:
        conversation = self.get_conversation(client_id, conversation_id)
        if conversation is None:
            return None
        now = datetime.now(timezone.utc)
        message = Message(conversation_id=conversation.id, role=role, content=content, metadata_=metadata or {})
        conversation.last_message_at = now
        conversation.updated_at = now
        self._db.add(message)
        self._db.commit()
        self._db.refresh(message)
        return message

    def update_title(self, client_id: str, conversation_id: str, title: str) -> Conversation | None:
        conversation = self.get_conversation(client_id, conversation_id)
        if conversation is None:
            return None
        conversation.title = title[:220]
        conversation.updated_at = datetime.now(timezone.utc)
        self._db.commit()
        self._db.refresh(conversation)
        return conversation

    def delete_conversation(self, client_id: str, conversation_id: str) -> bool:
        conversation = self.get_conversation(client_id, conversation_id)
        if conversation is None:
            return False
        self._db.delete(conversation)
        self._db.commit()
        return True

    def add_trace(
        self,
        client_id: str,
        conversation_id: str,
        trace_id: str,
        route: str,
        model_name: str,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        tool_calls: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        conversation = self.get_conversation(client_id, conversation_id)
        if conversation is None:
            return
        self._db.add(
            AITrace(
                conversation_id=conversation.id,
                trace_id=trace_id,
                route=route,
                model_name=model_name,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=tool_calls,
                metadata_=metadata,
            )
        )
        self._db.commit()


def _parse_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except ValueError:
        return None
