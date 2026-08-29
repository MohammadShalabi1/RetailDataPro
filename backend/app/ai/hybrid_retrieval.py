from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from app.ai.context_budget import ContextBudgeter
from app.ai.embedding_provider import EmbeddingProvider
from app.ai.errors import AIProviderError
from app.ai.reranker import CrossEncoderReranker, LexicalCrossEncoderReranker, RerankedChunk
from app.ai.retrieval import RetrievedChunk, reciprocal_rank_fusion
from app.core.config import Settings, get_settings
from app.repositories.retrieval_repository import RetrievalRepository


class RetrievalTrace(BaseModel):
    query: str
    normalized_query: str
    allowed_source_ids: list[str]
    embedding_model: str | None = None
    dense_candidates: list[dict[str, Any]] = Field(default_factory=list)
    lexical_candidates: list[dict[str, Any]] = Field(default_factory=list)
    rrf_candidates: list[dict[str, Any]] = Field(default_factory=list)
    dedupe: dict[str, list[str]] = Field(default_factory=lambda: {"kept": [], "removed": []})
    reranked: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class HybridSearchResult(BaseModel):
    chunks: list[RerankedChunk]
    trace: RetrievalTrace
    confidence: float
    limitations: list[str] = Field(default_factory=list)


@dataclass
class SourceAuthorizationResult:
    allowed_source_ids: list[str]
    rejected_source_ids: list[str]


class HybridRetrievalService:
    def __init__(
        self,
        repository: RetrievalRepository,
        embedding_provider: EmbeddingProvider,
        settings: Settings | None = None,
        reranker: CrossEncoderReranker | None = None,
        context_budgeter: ContextBudgeter | None = None,
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._settings = settings or get_settings()
        self._reranker = reranker or LexicalCrossEncoderReranker()
        self._context_budgeter = context_budgeter or ContextBudgeter()

    async def search(self, query: str, allowed_source_ids: list[str]) -> HybridSearchResult:
        normalized_query = normalize_query(query)
        trace = RetrievalTrace(query=query, normalized_query=normalized_query, allowed_source_ids=allowed_source_ids)
        dense: list[RetrievedChunk] = []
        lexical: list[RetrievedChunk] = []

        try:
            embedding = await self._embedding_provider.embed_text(normalized_query)
            trace.embedding_model = embedding.model
            dense = self._repository.dense_search(
                embedding.embedding,
                limit=self._settings.rag_dense_top_k,
                source_ids=allowed_source_ids,
            )
            trace.dense_candidates = [_candidate_trace(chunk) for chunk in dense]
        except (AIProviderError, Exception):
            trace.limitations.append("Dense embedding retrieval was unavailable; used lexical retrieval only.")

        try:
            lexical = self._repository.lexical_search(
                normalized_query,
                limit=self._settings.rag_lexical_top_k,
                source_ids=allowed_source_ids,
            )
            trace.lexical_candidates = [_candidate_trace(chunk) for chunk in lexical]
        except Exception:
            trace.limitations.append("Lexical retrieval was unavailable; used dense retrieval only.")

        if not dense and not lexical:
            trace.limitations.append("No dense or lexical retrieval results were available.")
            return HybridSearchResult(chunks=[], trace=trace, confidence=0.0, limitations=trace.limitations)

        fused = reciprocal_rank_fusion(dense, lexical, limit=max(self._settings.rag_dense_top_k, self._settings.rag_lexical_top_k), k=self._settings.rag_rrf_k)
        trace.rrf_candidates = [_candidate_trace(chunk) for chunk in fused.chunks]
        deduped, removed = deduplicate_chunks(fused.chunks)
        trace.dedupe = {"kept": [chunk.chunk_id for chunk in deduped], "removed": removed}

        try:
            reranked = self._reranker.rerank(normalized_query, deduped, top_k=self._settings.rag_rerank_top_k)
        except Exception:
            trace.limitations.append("Reranking was unavailable; used RRF order.")
            reranked = [
                RerankedChunk(**chunk.model_dump(), reranked_position=index, rerank_score=float(chunk.rrf_score or chunk.score or 0.0))
                for index, chunk in enumerate(deduped[: self._settings.rag_rerank_top_k], start=1)
            ]
        reranked = [
            chunk.model_copy(update={"final_rank": index, "reranker_score": chunk.rerank_score})
            for index, chunk in enumerate(reranked, start=1)
        ]
        trace.reranked = [_candidate_trace(chunk) for chunk in reranked]

        budgeted = self._context_budgeter.build(
            user_question=normalized_query,
            system_instructions="Select only the strongest document evidence for grounded answering.",
            retrieved_evidence=reranked,
        )
        selected = budgeted.retrieved_evidence[: self._settings.rag_context_top_k]
        trace.context = {
            "selected_chunks": [chunk.chunk_id for chunk in selected],
            "dropped_chunks": budgeted.dropped_chunks + max(0, len(budgeted.retrieved_evidence) - len(selected)),
            "estimated_tokens": budgeted.estimated_tokens_used,
        }
        confidence = retrieval_confidence(dense, lexical, selected, trace.limitations)
        trace.confidence = confidence
        return HybridSearchResult(chunks=selected, trace=trace, confidence=confidence, limitations=trace.limitations)


def normalize_query(query: str) -> str:
    normalized = unicodedata.normalize("NFKC", query)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def deduplicate_chunks(chunks: list[RetrievedChunk], overlap_threshold: float = 0.86) -> tuple[list[RetrievedChunk], list[str]]:
    kept: list[RetrievedChunk] = []
    removed: list[str] = []
    for chunk in chunks:
        if any(chunk.source_id == other.source_id and _jaccard(chunk.content, other.content) >= overlap_threshold for other in kept):
            removed.append(chunk.chunk_id)
            continue
        kept.append(chunk)
    return kept, removed


def retrieval_confidence(
    dense: list[RetrievedChunk],
    lexical: list[RetrievedChunk],
    selected: list[RetrievedChunk],
    limitations: list[str],
) -> float:
    if not selected:
        return 0.0
    dense_ids = {chunk.chunk_id for chunk in dense[:5]}
    lexical_ids = {chunk.chunk_id for chunk in lexical[:5]}
    agreement = len(dense_ids & lexical_ids)
    score = 0.35 + min(0.25, len(selected) * 0.04) + min(0.25, agreement * 0.08)
    if dense and lexical:
        score += 0.1
    if limitations:
        score -= min(0.25, len(limitations) * 0.1)
    return round(max(0.0, min(0.95, score)), 2)


def _candidate_trace(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "dense_rank": chunk.dense_rank,
        "dense_score": chunk.dense_score,
        "lexical_rank": chunk.lexical_rank,
        "lexical_score": chunk.lexical_score,
        "rrf_score": chunk.rrf_score,
        "reranker_score": getattr(chunk, "rerank_score", None) or chunk.reranker_score,
        "final_rank": chunk.final_rank,
    }


def _jaccard(left: str, right: str) -> float:
    left_terms = set(_terms(left))
    right_terms = set(_terms(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _terms(value: str) -> list[str]:
    return [term for term in re.findall(r"[a-zA-Z0-9_-]+", value.lower()) if term]
