from __future__ import annotations

from pydantic import BaseModel, Field

from app.ai.reranker import RerankedChunk


class BudgetedContext(BaseModel):
    system_instructions: str
    recent_conversation: list[str]
    memory_summary: str | None
    tool_results: list[dict]
    retrieved_evidence: list[RerankedChunk]
    user_question: str
    token_budget: int
    estimated_tokens_used: int
    dropped_chunks: int = 0
    dropped_tokens: int = 0


class ContextBudgeter:
    def __init__(self, token_budget: int = 4_000) -> None:
        self._token_budget = token_budget

    def build(
        self,
        user_question: str,
        system_instructions: str,
        recent_conversation: list[str] | None = None,
        memory_summary: str | None = None,
        tool_results: list[dict] | None = None,
        retrieved_evidence: list[RerankedChunk] | None = None,
    ) -> BudgetedContext:
        recent = recent_conversation or []
        tools = tool_results or []
        evidence = sorted(retrieved_evidence or [], key=lambda chunk: chunk.rerank_score, reverse=True)
        fixed_tokens = (
            _estimate_tokens(user_question)
            + _estimate_tokens(system_instructions)
            + _estimate_tokens(memory_summary or "")
            + sum(_estimate_tokens(item) for item in recent)
            + sum(_estimate_tokens(str(item)) for item in tools)
        )
        remaining = max(0, self._token_budget - fixed_tokens)
        kept: list[RerankedChunk] = []
        dropped_tokens = 0
        for chunk in evidence:
            token_count = _estimate_tokens(chunk.content)
            if token_count <= remaining:
                kept.append(chunk)
                remaining -= token_count
            else:
                dropped_tokens += token_count

        return BudgetedContext(
            system_instructions=system_instructions,
            recent_conversation=recent,
            memory_summary=memory_summary,
            tool_results=tools,
            retrieved_evidence=kept,
            user_question=user_question,
            token_budget=self._token_budget,
            estimated_tokens_used=self._token_budget - remaining,
            dropped_chunks=len(evidence) - len(kept),
            dropped_tokens=dropped_tokens,
        )


def _estimate_tokens(value: str) -> int:
    return max(1, len(value.split()))
