from __future__ import annotations

import logging
from typing import Any

from fastapi import Depends
from sqlalchemy.orm import Session

from app.ai.dependencies import get_ai_provider
from app.ai.orchestrator import AgentDependencies, AgentTurnRequest, AgentTurnResult, run_turn
from app.ai.provider import AIProvider
from app.ai.sql.dependencies import get_text_to_sql_pipeline
from app.ai.sql.pipeline import TextToSQLPipeline
from app.database.session import get_db
from app.repositories.conversation_repository import ConversationRepository
from app.schemas.conversations import (
    ClientCitation,
    ClientMessage,
    ConversationDetail,
    ConversationMessageResponse,
    ConversationSummary,
)
from app.services.analytics_service import AnalyticsService, get_analytics_service
from app.services.document_service import DocumentService, get_document_service
from app.services.observability_service import TraceRecord, observability_service

logger = logging.getLogger(__name__)

CLIENT_FALLBACK_ERROR = "I couldn't complete that request right now. Please try again."


class ConversationNotFoundError(Exception):
    pass


class ConversationService:
    def __init__(
        self,
        repository: ConversationRepository,
        ai_provider: AIProvider,
        analytics_service: AnalyticsService,
        document_service: DocumentService,
        sql_pipeline: TextToSQLPipeline | None = None,
    ) -> None:
        self._repository = repository
        self._ai_provider = ai_provider
        self._analytics_service = analytics_service
        self._document_service = document_service
        self._sql_pipeline = sql_pipeline

    def create_conversation(self, client_id: str, title: str | None = None) -> ConversationSummary:
        return _conversation_summary(self._repository.create_conversation(client_id, _clean_title(title) or "New chat"))

    def list_conversations(self, client_id: str) -> list[ConversationSummary]:
        return [_conversation_summary(conversation) for conversation in self._repository.list_conversations(client_id)]

    def get_conversation(self, client_id: str, conversation_id: str) -> ConversationDetail:
        conversation = self._repository.get_conversation(client_id, conversation_id)
        if conversation is None:
            raise ConversationNotFoundError
        return ConversationDetail(
            **_conversation_summary(conversation).model_dump(),
            messages=[_client_message(message) for message in sorted(conversation.messages, key=lambda item: item.created_at)],
        )

    def get_messages(self, client_id: str, conversation_id: str) -> list[ClientMessage]:
        messages = self._repository.get_messages(client_id, conversation_id)
        if messages is None:
            raise ConversationNotFoundError
        return [_client_message(message) for message in messages]

    def update_title(self, client_id: str, conversation_id: str, title: str) -> ConversationSummary:
        conversation = self._repository.update_title(client_id, conversation_id, _clean_title(title) or "New chat")
        if conversation is None:
            raise ConversationNotFoundError
        return _conversation_summary(conversation)

    def delete_conversation(self, client_id: str, conversation_id: str) -> None:
        if not self._repository.delete_conversation(client_id, conversation_id):
            raise ConversationNotFoundError

    async def send_message(
        self,
        client_id: str,
        conversation_id: str,
        question: str,
        document_source_ids: list[str] | None = None,
    ) -> ConversationMessageResponse:
        conversation = self._repository.get_conversation(client_id, conversation_id)
        if conversation is None:
            raise ConversationNotFoundError

        previous_messages = self._repository.get_recent_messages(client_id, conversation_id, limit=12)
        user_message = self._repository.add_message(client_id, conversation_id, "user", question)
        if user_message is None:
            raise ConversationNotFoundError

        if conversation.title == "New chat" and self._repository.count_messages(client_id, conversation_id) <= 1:
            self._repository.update_title(client_id, conversation_id, _title_from_question(question))

        try:
            result = await run_turn(
                AgentTurnRequest(
                    question=question,
                    conversation_id=conversation_id,
                    document_source_ids=document_source_ids or [],
                    recent_messages=[
                        {"role": message.role, "content": message.content}
                        for message in previous_messages
                        if message.role in {"user", "assistant"}
                    ],
                ),
                AgentDependencies(
                    ai_provider=self._ai_provider,
                    analytics_service=self._analytics_service,
                    document_service=self._document_service,
                    sql_pipeline=self._sql_pipeline,
                    client_id=client_id,
                ),
            )
            trace_id = observability_service.new_trace_id()
            trace_record = _to_trace_record(trace_id, result)
            observability_service.add_trace(trace_record)
            self._repository.add_trace(
                client_id=client_id,
                conversation_id=conversation_id,
                trace_id=trace_id,
                route=trace_record.route,
                model_name=trace_record.model,
                latency_ms=trace_record.total_ms,
                input_tokens=trace_record.input_tokens,
                output_tokens=trace_record.output_tokens,
                tool_calls=[tool.to_trace_metadata() for tool in result.tool_results],
                metadata={"events": result.trace.events, "confidence": result.confidence},
            )
            assistant_message = self._repository.add_message(
                client_id,
                conversation_id,
                "assistant",
                result.answer,
                {
                    "citations": [citation.model_dump(mode="json") for citation in _friendly_citations(result)],
                    "internal": {"trace_id": trace_id, "limitations": result.limitations},
                },
            )
        except Exception:
            logger.exception("Chat turn failed for conversation %s", conversation_id)
            assistant_message = self._repository.add_message(
                client_id,
                conversation_id,
                "assistant",
                CLIENT_FALLBACK_ERROR,
                {"status": "failed"},
            )

        if assistant_message is None:
            raise ConversationNotFoundError
        return ConversationMessageResponse(conversation_id=conversation_id, message=_client_message(assistant_message))


def get_conversation_service(
    db: Session = Depends(get_db),
    ai_provider: AIProvider = Depends(get_ai_provider),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    document_service: DocumentService = Depends(get_document_service),
    sql_pipeline: TextToSQLPipeline | None = Depends(get_text_to_sql_pipeline),
) -> ConversationService:
    return ConversationService(ConversationRepository(db), ai_provider, analytics_service, document_service, sql_pipeline)


def _conversation_summary(conversation: Any) -> ConversationSummary:
    return ConversationSummary(
        id=str(conversation.id),
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        last_message_at=conversation.last_message_at,
    )


def _client_message(message: Any) -> ClientMessage:
    metadata = message.metadata_ or {}
    citations = [ClientCitation.model_validate(citation) for citation in metadata.get("citations", [])]
    return ClientMessage(
        id=str(message.id),
        conversation_id=str(message.conversation_id),
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        citations=citations,
        status=metadata.get("status", "complete"),
    )


def _clean_title(title: str | None) -> str | None:
    cleaned = " ".join((title or "").strip().split())
    return cleaned[:220] or None


def _title_from_question(question: str) -> str:
    normalized = " ".join(question.replace("?", " ").replace(".", " ").split())
    words = [word for word in normalized.split() if word.lower() not in {"which", "what", "the", "had", "have", "with", "this", "that", "about"}]
    if not words:
        return normalized[:60] or "New chat"
    title = " ".join(words[:5]).title()
    replacements = {
        "Supplier Weakest Fulfillment Reliability": "Supplier Reliability Review",
        "Supplier Fulfillment Reliability": "Supplier Reliability Review",
        "Analyze Suppliers": "Supplier Analysis",
        "Analyze Inventory": "Inventory Analysis",
    }
    return replacements.get(title, title[:60])


def _friendly_citations(result: AgentTurnResult) -> list[ClientCitation]:
    chunks: dict[tuple[str, str | None], dict[str, Any]] = {}
    for tool_result in result.tool_results:
        for chunk in tool_result.output.get("chunks") or []:
            chunks[(str(chunk.get("source_id")), str(chunk.get("chunk_id")))] = chunk

    citations: list[ClientCitation] = []
    for citation in result.citations:
        source_id = str(citation.get("source_id") or "")
        chunk_id = str(citation.get("chunk_id") or "") or None
        claim = citation.get("claim")
        if source_id == "analytics_summary":
            citations.append(ClientCitation(label="Retail analytics", claim=claim))
            continue
        if source_id == "retail_sql":
            citations.append(ClientCitation(label="Retail database", claim=claim))
            continue
        chunk = chunks.get((source_id, chunk_id))
        if chunk:
            page = int(chunk.get("chunk_index") or 0) + 1
            citations.append(
                ClientCitation(
                    label=f"{chunk.get('title') or 'Uploaded document'} - Page {page}",
                    claim=claim,
                    excerpt=_excerpt(str(chunk.get("content") or "")),
                )
            )
    return citations


def _excerpt(content: str) -> str:
    cleaned = " ".join(content.split())
    return cleaned[:240] if len(cleaned) <= 240 else f"{cleaned[:237]}..."


def _to_trace_record(trace_id: str, result: AgentTurnResult) -> TraceRecord:
    route = result.route.category.value if result.route else "blocked"
    model = result.model_selection.model if result.model_selection else "none"
    tool_results = [tool.model_dump(mode="json") for tool in result.tool_results]
    token_event = next((event for event in result.trace.events if event.get("stage") == "generate_answer"), {})
    sql_event = next((event for event in result.trace.events if event.get("stage") == "retail_sql"), {})
    return TraceRecord(
        trace_id=trace_id,
        route=route,
        model=model,
        plan_steps=len(result.execution_plan.steps) if result.execution_plan else 0,
        tools=[tool["tool_name"] for tool in tool_results],
        retrieved=0,
        reranked=0,
        cache_hit=False,
        generation_ms=int(token_event.get("latency_ms") or 0),
        total_ms=sum(int(event.get("latency_ms") or 0) for event in result.trace.events),
        input_tokens=int(token_event.get("input_tokens") or 0),
        output_tokens=int(token_event.get("output_tokens") or 0),
        confidence=result.confidence,
        generated_sql=sql_event.get("normalized_sql") or sql_event.get("generated_sql"),
        events=result.trace.events,
    )
