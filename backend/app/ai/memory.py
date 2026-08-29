from __future__ import annotations

from pydantic import BaseModel, Field


class MessageTurn(BaseModel):
    role: str
    content: str


class ConversationMemory(BaseModel):
    conversation_id: str
    recent_messages: list[MessageTurn] = Field(default_factory=list)
    summary: str | None = None
    selected_source_ids: list[str] = Field(default_factory=list)


class MemoryContext(BaseModel):
    recent_messages: list[MessageTurn]
    summary: str | None
    selected_source_ids: list[str]


class InMemoryConversationMemoryStore:
    def __init__(self) -> None:
        self._items: dict[str, ConversationMemory] = {}

    def get(self, conversation_id: str) -> ConversationMemory:
        return self._items.setdefault(conversation_id, ConversationMemory(conversation_id=conversation_id))

    def save(self, memory: ConversationMemory) -> None:
        self._items[memory.conversation_id] = memory


class ConversationMemoryManager:
    def __init__(self, store: InMemoryConversationMemoryStore | None = None, recent_turn_limit: int = 6) -> None:
        self._store = store or InMemoryConversationMemoryStore()
        self._recent_turn_limit = recent_turn_limit

    def load(self, conversation_id: str | None) -> MemoryContext:
        if conversation_id is None:
            return MemoryContext(recent_messages=[], summary=None, selected_source_ids=[])
        memory = self._store.get(conversation_id)
        return MemoryContext(
            recent_messages=memory.recent_messages[-self._recent_turn_limit :],
            summary=memory.summary,
            selected_source_ids=memory.selected_source_ids,
        )

    def append_turn(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        selected_source_ids: list[str] | None = None,
    ) -> ConversationMemory:
        memory = self._store.get(conversation_id)
        memory.recent_messages.extend(
            [
                MessageTurn(role="user", content=user_message),
                MessageTurn(role="assistant", content=assistant_message),
            ]
        )
        if selected_source_ids:
            memory.selected_source_ids = list(dict.fromkeys([*memory.selected_source_ids, *selected_source_ids]))
        if len(memory.recent_messages) > self._recent_turn_limit:
            older = memory.recent_messages[: -self._recent_turn_limit]
            memory.summary = _compress_summary(memory.summary, older)
            memory.recent_messages = memory.recent_messages[-self._recent_turn_limit :]
        self._store.save(memory)
        return memory


def _compress_summary(existing_summary: str | None, older_messages: list[MessageTurn]) -> str:
    snippets = [message.content for message in older_messages if message.content]
    combined = " ".join([existing_summary or "", *snippets]).strip()
    return combined[:1_000]
