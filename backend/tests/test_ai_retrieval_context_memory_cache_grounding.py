from __future__ import annotations

import pytest
from sqlalchemy.dialects import postgresql

from app.ai.context_budget import ContextBudgeter
from app.ai.grounded_answer import GroundedAnswer, GroundedAnswerGenerator, GroundingEvidence
from app.ai.memory import ConversationMemoryManager, InMemoryConversationMemoryStore
from app.ai.provider import AIProvider
from app.ai.reranker import LexicalCrossEncoderReranker
from app.ai.retrieval import RetrievedChunk, reciprocal_rank_fusion
from app.ai.schemas import AIResponse, StructuredGenerationRequest, TextGenerationRequest
from app.ai.semantic_cache import SemanticCache
from app.models import SourceChunk
from app.repositories.retrieval_repository import RetrievalRepository


class FakeGroundedProvider(AIProvider):
    def __init__(self, answer: GroundedAnswer) -> None:
        self.answer = answer

    async def generate_text(self, request: TextGenerationRequest) -> AIResponse[str]:
        raise NotImplementedError

    async def generate_structured(self, request: StructuredGenerationRequest) -> AIResponse[GroundedAnswer]:
        return AIResponse(content=self.answer, model="fake", provider="fake", latency_ms=1)


def test_source_chunks_include_pgvector_and_lexical_columns() -> None:
    columns = SourceChunk.__table__.columns

    assert "embedding" in columns
    assert "search_vector" in columns


def test_retrieval_statements_compile_for_postgresql() -> None:
    repo = RetrievalRepository(db=None)  # type: ignore[arg-type]

    dense_sql = str(repo.build_dense_statement([0.1] * 768).compile(dialect=postgresql.dialect()))
    lexical_sql = str(repo.build_lexical_statement("supplier reliability").compile(dialect=postgresql.dialect()))

    assert "<=>" in dense_sql
    assert "plainto_tsquery" in lexical_sql
    assert "@@" in lexical_sql


def test_reciprocal_rank_fusion_combines_dense_and_lexical_results() -> None:
    chunk_a = _chunk("a", score=0.1, method="dense")
    chunk_b = _chunk("b", score=0.2, method="lexical")

    result = reciprocal_rank_fusion([chunk_a], [chunk_b, chunk_a], limit=2)

    assert result.dense_count == 1
    assert result.lexical_count == 2
    assert [chunk.chunk_id for chunk in result.chunks] == ["a", "b"]
    assert result.chunks[0].retrieval_method == "hybrid_rrf"


def test_reranker_records_positions_and_scores() -> None:
    chunks = [_chunk("a", content="supplier delays hurt electronics revenue"), _chunk("b", content="unrelated note")]

    reranked = LexicalCrossEncoderReranker().rerank("supplier electronics revenue", chunks, top_k=1)

    assert reranked[0].chunk_id == "a"
    assert reranked[0].reranked_position == 1
    assert reranked[0].rerank_score > 0
    assert reranked[0].initial_rank == 1


def test_context_budget_drops_low_priority_evidence() -> None:
    chunks = [
        _chunk("a", content="important evidence", score=0.9),
        _chunk("b", content="too many tokens for this tiny budget", score=0.1),
    ]
    reranked = LexicalCrossEncoderReranker().rerank("important evidence", chunks, top_k=2)

    context = ContextBudgeter(token_budget=8).build(
        user_question="question",
        system_instructions="system",
        retrieved_evidence=reranked,
    )

    assert context.dropped_chunks >= 1
    assert context.dropped_tokens > 0


def test_memory_uses_recent_turns_plus_summary() -> None:
    manager = ConversationMemoryManager(InMemoryConversationMemoryStore(), recent_turn_limit=2)

    manager.append_turn("conv-1", "Which categories fell?", "Electronics fell.", ["source-1"])
    memory = manager.append_turn("conv-1", "Why did the first one fall?", "Supplier delays.", ["source-2"])
    context = manager.load("conv-1")

    assert memory.summary is not None
    assert [message.content for message in context.recent_messages] == ["Why did the first one fall?", "Supplier delays."]
    assert context.selected_source_ids == ["source-1", "source-2"]


def test_semantic_cache_requires_scope_compatibility_and_confidence() -> None:
    cache = SemanticCache(similarity_threshold=0.9, min_confidence_to_cache=0.7)
    key = cache.make_key("Revenue last month", "retail_analytics", "model-a", "prompt-v1", ["source-1"], "ctx-1")

    assert cache.put(key, [1.0, 0.0], "answer", confidence=0.9) is True
    assert cache.put(key, [1.0, 0.0], "bad", confidence=0.2) is False
    hit, stats = cache.get(key, [0.95, 0.05])
    other_key = cache.make_key("Revenue last month", "retail_analytics", "model-b", "prompt-v1", ["source-1"], "ctx-1")
    miss, miss_stats = cache.get(other_key, [0.95, 0.05])

    assert hit is not None
    assert stats.model_calls_avoided == 1
    assert miss is None
    assert miss_stats.cache_miss is True


@pytest.mark.asyncio
async def test_grounded_answer_requires_evidence() -> None:
    generator = GroundedAnswerGenerator(
        FakeGroundedProvider(GroundedAnswer(answer="Unused", citations=[], confidence=1.0)),
    )

    result = await generator.generate("What happened?", GroundingEvidence())

    assert result.confidence == 0.0
    assert result.citations == []
    assert "No tool results" in result.limitations[0]


@pytest.mark.asyncio
async def test_grounded_answer_rejects_citations_not_in_context() -> None:
    answer = GroundedAnswer(
        answer="Supplier delays hurt revenue.",
        citations=[{"source_id": "missing", "chunk_id": "missing", "claim": "Supplier delays"}],
        confidence=0.8,
    )
    generator = GroundedAnswerGenerator(FakeGroundedProvider(answer))

    result = await generator.generate(
        "Why?",
        GroundingEvidence(retrieved_chunks=[{"source_id": "source-1", "chunk_id": "chunk-1", "content": "Supplier delays"}]),
    )

    assert result.confidence == 0.0
    assert result.citations == []


def _chunk(chunk_id: str, content: str = "content", score: float = 0.1, method: str = "dense") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_id="source-1",
        content=content,
        source_title="Supplier Report",
        initial_rank=1,
        score=score,
        retrieval_method=method,
    )
