from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.ai.dependencies import get_ai_provider
from app.ai.provider import AIProvider
from app.core.security import ClientContext, get_client_context
from app.schemas.conversations import ClientMessage
from app.services.conversation_service import ConversationNotFoundError, ConversationService, get_conversation_service

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str | None = Field(default=None, min_length=1, max_length=4_000)
    message: str | None = Field(default=None, min_length=1, max_length=4_000)
    conversation_id: str | None = None
    document_source_ids: list[str] = Field(default_factory=list, max_length=10)


class ChatResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    conversation_id: str
    message: ClientMessage


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    client: ClientContext = Depends(get_client_context),
    conversation_service: ConversationService = Depends(get_conversation_service),
    ai_provider: AIProvider = Depends(get_ai_provider),
) -> ChatResponse:
    del ai_provider
    question = request.message or request.question
    if not question:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="A message is required.")
    conversation_id = request.conversation_id
    if conversation_id is None:
        conversation = conversation_service.create_conversation(client.client_id)
        conversation_id = conversation.id
    try:
        response = await conversation_service.send_message(
            client.client_id,
            conversation_id,
            question,
            request.document_source_ids,
        )
    except ConversationNotFoundError:
        conversation = conversation_service.create_conversation(client.client_id)
        response = await conversation_service.send_message(
            client.client_id,
            conversation.id,
            question,
            request.document_source_ids,
        )
    return ChatResponse(
        conversation_id=response.conversation_id,
        message=response.message,
    )
