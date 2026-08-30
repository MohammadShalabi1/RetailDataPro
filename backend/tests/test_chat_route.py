from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.conversations import ClientMessage, ConversationDetail, ConversationMessageResponse, ConversationSummary
from app.services.conversation_service import ConversationNotFoundError, get_conversation_service


class FakeConversationService:
    def __init__(self) -> None:
        self.conversations: dict[str, dict[str, ConversationDetail]] = {}

    def create_conversation(self, client_id: str, title: str | None = None) -> ConversationSummary:
        conversation_id = str(uuid4())
        now = datetime(2026, 8, 30, tzinfo=timezone.utc)
        conversation = ConversationDetail(
            id=conversation_id,
            title=title or "New chat",
            created_at=now,
            updated_at=now,
            last_message_at=now,
            messages=[],
        )
        self.conversations.setdefault(client_id, {})[conversation_id] = conversation
        return ConversationSummary(**conversation.model_dump(exclude={"messages"}))

    def list_conversations(self, client_id: str) -> list[ConversationSummary]:
        return [
            ConversationSummary(**conversation.model_dump(exclude={"messages"}))
            for conversation in self.conversations.get(client_id, {}).values()
        ]

    def get_conversation(self, client_id: str, conversation_id: str) -> ConversationDetail:
        conversation = self.conversations.get(client_id, {}).get(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError
        return conversation

    def get_messages(self, client_id: str, conversation_id: str) -> list[ClientMessage]:
        return self.get_conversation(client_id, conversation_id).messages

    async def send_message(
        self,
        client_id: str,
        conversation_id: str,
        question: str,
        document_source_ids: list[str] | None = None,
    ) -> ConversationMessageResponse:
        del document_source_ids
        conversation = self.get_conversation(client_id, conversation_id)
        now = datetime(2026, 8, 30, tzinfo=timezone.utc)
        if conversation.title == "New chat":
            conversation.title = "Supplier Reliability Review" if "supplier" in question.lower() else question[:60]
        user = ClientMessage(
            id=str(uuid4()),
            conversation_id=conversation_id,
            role="user",
            content=question,
            created_at=now,
            citations=[],
        )
        assistant = ClientMessage(
            id=str(uuid4()),
            conversation_id=conversation_id,
            role="assistant",
            content="Supplier reliability is weakest for the lowest fulfillment performer.",
            created_at=now,
            citations=[{"label": "Retail analytics", "claim": "Supplier reliability comparison"}],
        )
        conversation.messages.extend([user, assistant])
        conversation.last_message_at = now
        conversation.updated_at = now
        return ConversationMessageResponse(conversation_id=conversation_id, message=assistant)


def test_chat_endpoint_returns_client_safe_message_contract() -> None:
    service = FakeConversationService()
    app = create_app()
    app.dependency_overrides[get_conversation_service] = lambda: service

    try:
        response = TestClient(app).post("/api/chat", json={"question": "Which supplier was weakest?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"conversation_id", "message"}
    assert body["message"]["role"] == "assistant"
    assert body["message"]["citations"][0]["label"] == "Retail analytics"
    assert "trace_id" not in str(body)
    assert "multi_source" not in str(body)
    assert "gemini" not in str(body).lower()
    assert "tool_results" not in str(body)


def test_conversation_refresh_restore_contract() -> None:
    service = FakeConversationService()
    app = create_app()
    app.dependency_overrides[get_conversation_service] = lambda: service
    client = TestClient(app)

    try:
        created = client.post("/api/conversations", json={}).json()
        conversation_id = created["id"]
        first = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"message": "Which supplier had the weakest fulfillment reliability?"},
        )
        restored = client.get(f"/api/conversations/{conversation_id}")
        second = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"message": "What should we do next?"},
        )
        restored_again = client.get(f"/api/conversations/{conversation_id}")
    finally:
        app.dependency_overrides.clear()

    assert first.status_code == 200
    assert second.status_code == 200
    assert [message["role"] for message in restored.json()["messages"]] == ["user", "assistant"]
    assert [message["role"] for message in restored_again.json()["messages"]] == ["user", "assistant", "user", "assistant"]


def test_multiple_chat_and_client_scope_isolation() -> None:
    service = FakeConversationService()
    app = create_app()
    app.dependency_overrides[get_conversation_service] = lambda: service
    client = TestClient(app)

    try:
        chat_a = client.post("/api/conversations", json={}, headers={"X-Client-Id": "client-a"}).json()["id"]
        chat_b = client.post("/api/conversations", json={}, headers={"X-Client-Id": "client-a"}).json()["id"]
        client.post(f"/api/conversations/{chat_a}/messages", json={"message": "Analyze suppliers."}, headers={"X-Client-Id": "client-a"})
        client.post(f"/api/conversations/{chat_b}/messages", json={"message": "Analyze inventory."}, headers={"X-Client-Id": "client-a"})
        list_a = client.get("/api/conversations", headers={"X-Client-Id": "client-a"}).json()
        opened_a = client.get(f"/api/conversations/{chat_a}", headers={"X-Client-Id": "client-a"}).json()
        opened_b = client.get(f"/api/conversations/{chat_b}", headers={"X-Client-Id": "client-a"}).json()
        forbidden = client.get(f"/api/conversations/{chat_a}", headers={"X-Client-Id": "client-b"})
    finally:
        app.dependency_overrides.clear()

    assert {conversation["id"] for conversation in list_a} == {chat_a, chat_b}
    assert opened_a["messages"][0]["content"] == "Analyze suppliers."
    assert opened_b["messages"][0]["content"] == "Analyze inventory."
    assert forbidden.status_code == 404
