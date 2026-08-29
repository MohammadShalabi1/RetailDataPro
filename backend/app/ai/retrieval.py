from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_id: str
    document_id: str | None = None
    content: str
    source_title: str
    initial_rank: int
    score: float
    retrieval_method: str
    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None
    final_rank: int | None = None
    page_number: int | None = None
    token_count: int | None = None
    metadata: dict = Field(default_factory=dict)


class HybridRetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]
    dense_count: int
    lexical_count: int


def reciprocal_rank_fusion(
    dense_results: list[RetrievedChunk],
    lexical_results: list[RetrievedChunk],
    limit: int = 10,
    k: int = 60,
) -> HybridRetrievalResult:
    scores: dict[str, float] = {}
    chosen: dict[str, RetrievedChunk] = {}

    for method, results in (("dense", dense_results), ("lexical", lexical_results)):
        for index, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (1.0 / (k + index))
            existing = chosen.get(chunk.chunk_id)
            updates = {
                "dense_rank": index if method == "dense" else chunk.dense_rank,
                "dense_score": chunk.score if method == "dense" else chunk.dense_score,
                "lexical_rank": index if method == "lexical" else chunk.lexical_rank,
                "lexical_score": chunk.score if method == "lexical" else chunk.lexical_score,
            }
            if existing is None:
                chosen[chunk.chunk_id] = chunk.model_copy(update=updates)
            else:
                chosen[chunk.chunk_id] = existing.model_copy(update={key: value for key, value in updates.items() if value is not None})

    fused = sorted(chosen.values(), key=lambda chunk: scores[chunk.chunk_id], reverse=True)
    ranked = [
        chunk.model_copy(
            update={
                "initial_rank": rank,
                "score": round(scores[chunk.chunk_id], 6),
                "rrf_score": round(scores[chunk.chunk_id], 6),
                "retrieval_method": "hybrid_rrf",
            }
        )
        for rank, chunk in enumerate(fused[:limit], start=1)
    ]
    return HybridRetrievalResult(chunks=ranked, dense_count=len(dense_results), lexical_count=len(lexical_results))
