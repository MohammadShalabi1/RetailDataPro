from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ClientCitation(BaseModel):
    label: str
    claim: str | None = None
    excerpt: str | None = None


class ClientMessage(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: datetime
    citations: list[ClientCitation] = Field(default_factory=list)
    status: str = "complete"


class ConversationCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=220)


class ConversationUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=220)


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None = None


class ConversationDetail(ConversationSummary):
    messages: list[ClientMessage] = Field(default_factory=list)


class ConversationMessageCreateRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4_000)
    document_source_ids: list[str] = Field(default_factory=list, max_length=10)


class ConversationMessageResponse(BaseModel):
    conversation_id: str
    message: ClientMessage
