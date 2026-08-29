from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from app.ai.retrieval import RetrievedChunk


class RerankedChunk(RetrievedChunk):
    reranked_position: int
    rerank_score: float = Field(ge=0.0, le=1.0)


class CrossEncoderReranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int = 6) -> list[RerankedChunk]:
        ...


class LexicalCrossEncoderReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int = 6) -> list[RerankedChunk]:
        query_terms = {term.lower() for term in query.split() if term.strip()}
        scored: list[tuple[float, RetrievedChunk]] = []
        for chunk in chunks:
            content_terms = {term.strip(".,:;!?").lower() for term in chunk.content.split()}
            overlap = len(query_terms & content_terms)
            score = min(1.0, (overlap / max(len(query_terms), 1)) + min(chunk.score, 0.2))
            scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RerankedChunk(**chunk.model_dump(), reranked_position=index, rerank_score=round(score, 4))
            for index, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]
