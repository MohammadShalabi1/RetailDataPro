from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.core.security import ClientContext, get_client_context
from app.schemas.conversations import (
    ClientMessage,
    ConversationCreateRequest,
    ConversationDetail,
    ConversationMessageCreateRequest,
    ConversationMessageResponse,
    ConversationSummary,
    ConversationUpdateRequest,
)
from app.services.conversation_service import ConversationNotFoundError, ConversationService, get_conversation_service

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationSummary, status_code=status.HTTP_201_CREATED)
def create_conversation(
    request: ConversationCreateRequest,
    client: ClientContext = Depends(get_client_context),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationSummary:
    return conversation_service.create_conversation(client.client_id, request.title)


@router.get("", response_model=list[ConversationSummary])
def list_conversations(
    client: ClientContext = Depends(get_client_context),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> list[ConversationSummary]:
    return conversation_service.list_conversations(client.client_id)


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str,
    client: ClientContext = Depends(get_client_context),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationDetail:
    try:
        return conversation_service.get_conversation(client.client_id, conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found") from exc


@router.get("/{conversation_id}/messages", response_model=list[ClientMessage])
def get_messages(
    conversation_id: str,
    client: ClientContext = Depends(get_client_context),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> list[ClientMessage]:
    try:
        return conversation_service.get_messages(client.client_id, conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found") from exc


@router.patch("/{conversation_id}", response_model=ConversationSummary)
def update_conversation(
    conversation_id: str,
    request: ConversationUpdateRequest,
    client: ClientContext = Depends(get_client_context),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationSummary:
    try:
        return conversation_service.update_title(client.client_id, conversation_id, request.title)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found") from exc


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str,
    client: ClientContext = Depends(get_client_context),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> Response:
    try:
        conversation_service.delete_conversation(client.client_id, conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{conversation_id}/messages", response_model=ConversationMessageResponse)
async def send_message(
    conversation_id: str,
    request: ConversationMessageCreateRequest,
    client: ClientContext = Depends(get_client_context),
    conversation_service: ConversationService = Depends(get_conversation_service),
) -> ConversationMessageResponse:
    try:
        return await conversation_service.send_message(
            client.client_id,
            conversation_id,
            request.message,
            request.document_source_ids,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found") from exc
