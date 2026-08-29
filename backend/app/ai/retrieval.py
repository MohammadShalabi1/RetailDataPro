from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_id: str
    content: str
    source_title: str
    initial_rank: int
    score: float
    retrieval_method: str
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

    for results in (dense_results, lexical_results):
        for index, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (1.0 / (k + index))
            chosen.setdefault(chunk.chunk_id, chunk)

    fused = sorted(chosen.values(), key=lambda chunk: scores[chunk.chunk_id], reverse=True)
    ranked = [
        chunk.model_copy(update={"initial_rank": rank, "score": round(scores[chunk.chunk_id], 6), "retrieval_method": "hybrid_rrf"})
        for rank, chunk in enumerate(fused[:limit], start=1)
    ]
    return HybridRetrievalResult(chunks=ranked, dense_count=len(dense_results), lexical_count=len(lexical_results))
