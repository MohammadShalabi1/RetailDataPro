from __future__ import annotations

import pytest
from sqlalchemy.dialects import postgresql

from app.ai.context_budget import ContextBudgeter
from app.ai.embedding_provider import EmbeddingResponse
from app.ai.grounded_answer import GroundedAnswer, GroundedAnswerGenerator, GroundingEvidence
from app.ai.hybrid_retrieval import HybridRetrievalService, deduplicate_chunks, normalize_query
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


class FakeEmbeddingProvider:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.queries: list[str] = []

    async def embed_text(self, text: str) -> EmbeddingResponse:
        self.queries.append(text)
        if self.fail:
            from app.ai.errors import AIProviderResponseError

            raise AIProviderResponseError("embedding failed")
        return EmbeddingResponse(embedding=[0.1] * 768, model="fake-embedding", provider="fake", latency_ms=1)

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResponse]:
        return [await self.embed_text(text) for text in texts]


class FakeRetrievalRepository:
    def __init__(self, dense=None, lexical=None, fail_dense: bool = False, fail_lexical: bool = False) -> None:
        self.dense = dense or []
        self.lexical = lexical or []
        self.fail_dense = fail_dense
        self.fail_lexical = fail_lexical
        self.dense_scopes: list[list[str]] = []
        self.lexical_scopes: list[list[str]] = []

    def dense_search(self, query_embedding, limit=20, source_ids=None):
        self.dense_scopes.append(source_ids or [])
        if self.fail_dense:
            raise RuntimeError("dense failed")
        return self.dense[:limit]

    def lexical_search(self, query, limit=20, source_ids=None):
        self.lexical_scopes.append(source_ids or [])
        if self.fail_lexical:
            raise RuntimeError("lexical failed")
        return self.lexical[:limit]


class FailingReranker:
    def rerank(self, query, chunks, top_k=6):
        raise RuntimeError("rerank failed")


def test_source_chunks_include_pgvector_and_lexical_columns() -> None:
    columns = SourceChunk.__table__.columns

    assert "embedding" in columns
    assert "search_vector" in columns


def test_retrieval_statements_compile_for_postgresql() -> None:
    repo = RetrievalRepository(db=None)  # type: ignore[arg-type]

    dense_sql = str(repo.build_dense_statement([0.1] * 768).compile(dialect=postgresql.dialect()))
    dense_scoped_sql = str(repo.build_dense_statement([0.1] * 768, source_ids=["11111111-1111-1111-1111-111111111111"]).compile(dialect=postgresql.dialect()))
    lexical_sql = str(repo.build_lexical_statement("supplier reliability").compile(dialect=postgresql.dialect()))
    lexical_scoped_sql = str(repo.build_lexical_statement("supplier reliability", source_ids=["11111111-1111-1111-1111-111111111111"]).compile(dialect=postgresql.dialect()))

    assert "<=>" in dense_sql
    assert "sources.id" in dense_scoped_sql
    assert "websearch_to_tsquery" in lexical_sql
    assert "@@" in lexical_sql
    assert "sources.id" in lexical_scoped_sql


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


@pytest.mark.asyncio
async def test_hybrid_retrieval_runs_dense_and_lexical_with_same_source_scope() -> None:
    dense = [_chunk("a", "Warehouse disruptions reduced supplier fulfillment.", score=0.1, method="dense")]
    lexical = [_chunk("b", "PO-CE-0821 was delayed by a supplier issue.", score=0.9, method="lexical")]
    repository = FakeRetrievalRepository(dense=dense, lexical=lexical)

    result = await HybridRetrievalService(repository, FakeEmbeddingProvider()).search(
        "Why were deliveries unreliable?",
        ["source-1"],
    )

    assert repository.dense_scopes == [["source-1"]]
    assert repository.lexical_scopes == [["source-1"]]
    assert [chunk.chunk_id for chunk in result.chunks] == ["a", "b"]
    assert result.trace.embedding_model == "fake-embedding"
    assert result.confidence > 0


def test_deduplication_removes_overlapping_chunks_from_same_source_only() -> None:
    first = _chunk("a", "supplier reliability declined for paper towels and trash bags")
    duplicate = _chunk("b", "supplier reliability declined for paper towels and trash bags")
    other_source = first.model_copy(update={"chunk_id": "c", "source_id": "source-2"})

    kept, removed = deduplicate_chunks([first, duplicate, other_source])

    assert [chunk.chunk_id for chunk in kept] == ["a", "c"]
    assert removed == ["b"]


@pytest.mark.asyncio
async def test_embedding_failure_degrades_to_lexical_only() -> None:
    repository = FakeRetrievalRepository(lexical=[_chunk("lex", "PO-CE-0821 supplier issue", score=0.9, method="lexical")])

    result = await HybridRetrievalService(repository, FakeEmbeddingProvider(fail=True)).search("PO-CE-0821", ["source-1"])

    assert [chunk.chunk_id for chunk in result.chunks] == ["lex"]
    assert "Dense embedding retrieval was unavailable" in result.limitations[0]


@pytest.mark.asyncio
async def test_lexical_failure_degrades_to_dense_only() -> None:
    repository = FakeRetrievalRepository(dense=[_chunk("dense", "delivery reliability problem", score=0.1, method="dense")], fail_lexical=True)

    result = await HybridRetrievalService(repository, FakeEmbeddingProvider()).search("Why were deliveries unreliable?", ["source-1"])

    assert [chunk.chunk_id for chunk in result.chunks] == ["dense"]
    assert any("Lexical retrieval was unavailable" in limitation for limitation in result.limitations)


@pytest.mark.asyncio
async def test_reranker_failure_uses_rrf_order() -> None:
    repository = FakeRetrievalRepository(dense=[_chunk("dense", "delivery reliability problem", score=0.1, method="dense")])

    result = await HybridRetrievalService(repository, FakeEmbeddingProvider(), reranker=FailingReranker()).search(
        "Why were deliveries unreliable?",
        ["source-1"],
    )

    assert [chunk.chunk_id for chunk in result.chunks] == ["dense"]
    assert any("Reranking was unavailable" in limitation for limitation in result.limitations)


@pytest.mark.asyncio
async def test_both_retrieval_methods_fail_safely() -> None:
    repository = FakeRetrievalRepository(fail_dense=True, fail_lexical=True)

    result = await HybridRetrievalService(repository, FakeEmbeddingProvider()).search("question", ["source-1"])

    assert result.chunks == []
    assert result.confidence == 0.0
    assert any("No dense or lexical" in limitation for limitation in result.limitations)


def test_query_normalization_preserves_identifiers_and_dates() -> None:
    assert normalize_query("  What happened to PO-CE-0821 in August 2026?  ") == "What happened to PO-CE-0821 in August 2026?"


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


def test_semantic_cache_misses_when_selected_sources_change() -> None:
    cache = SemanticCache(similarity_threshold=0.9, min_confidence_to_cache=0.7)
    key_a = cache.make_key("What does the report say?", "document_search", "model-a", "prompt-v1", ["source-a"], "ctx-a")
    key_b = cache.make_key("What does the report say?", "document_search", "model-a", "prompt-v1", ["source-b"], "ctx-a")

    cache.put(key_a, [1.0, 0.0], "source a answer", confidence=0.9)
    hit, stats = cache.get(key_b, [1.0, 0.0])

    assert hit is None
    assert stats.cache_miss is True


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


@pytest.mark.asyncio
async def test_grounded_answer_accepts_tool_result_citation() -> None:
    answer = GroundedAnswer(
        answer="The best-selling product is supported by analytics output.",
        citations=[{"source_id": "analytics_summary", "claim": "Top product came from analytics_summary."}],
        confidence=0.8,
    )
    generator = GroundedAnswerGenerator(FakeGroundedProvider(answer))

    result = await generator.generate(
        "What is our best selling product?",
        GroundingEvidence(
            tool_results=[
                {
                    "tool_name": "analytics_summary",
                    "status": "success",
                    "output": {"summary_type": "top_products", "data": {"items": []}},
                }
            ]
        ),
    )

    assert result.answer.startswith("The best-selling product")
    assert result.citations[0].source_id == "analytics_summary"


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
