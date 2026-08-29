from __future__ import annotations

from typing import Any

from sqlalchemy import Select, String, cast, func, select
from sqlalchemy.orm import Session

from app.ai.retrieval import RetrievedChunk
from app.models import Source, SourceChunk


class RetrievalRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def dense_search(self, query_embedding: list[float], limit: int = 20) -> list[RetrievedChunk]:
        return self._fetch_chunks(self.build_dense_statement(query_embedding, limit), "dense")

    def lexical_search(self, query: str, limit: int = 20) -> list[RetrievedChunk]:
        return self._fetch_chunks(self.build_lexical_statement(query, limit), "lexical")

    def build_dense_statement(self, query_embedding: list[float], limit: int = 20) -> Select:
        distance = SourceChunk.embedding.op("<=>")(query_embedding).label("score")
        return (
            select(
                cast(SourceChunk.id, String).label("chunk_id"),
                cast(Source.id, String).label("source_id"),
                SourceChunk.content,
                Source.title.label("source_title"),
                distance,
                SourceChunk.metadata_.label("metadata"),
            )
            .select_from(SourceChunk)
            .join(Source, Source.id == SourceChunk.source_id)
            .where(SourceChunk.embedding.is_not(None))
            .order_by("score")
            .limit(limit)
        )

    def build_lexical_statement(self, query: str, limit: int = 20) -> Select:
        tsquery = func.plainto_tsquery("english", query)
        rank = func.ts_rank(SourceChunk.search_vector, tsquery).label("score")
        return (
            select(
                cast(SourceChunk.id, String).label("chunk_id"),
                cast(Source.id, String).label("source_id"),
                SourceChunk.content,
                Source.title.label("source_title"),
                rank,
                SourceChunk.metadata_.label("metadata"),
            )
            .select_from(SourceChunk)
            .join(Source, Source.id == SourceChunk.source_id)
            .where(SourceChunk.search_vector.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(limit)
        )

    def _fetch_chunks(self, statement: Select, method: str) -> list[RetrievedChunk]:
        rows = [dict(row._mapping) for row in self._db.execute(statement).all()]
        return [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                source_id=row["source_id"],
                content=row["content"],
                source_title=row["source_title"],
                initial_rank=index,
                score=float(row.get("score") or 0.0),
                retrieval_method=method,
                metadata=_dict_value(row.get("metadata")),
            )
            for index, row in enumerate(rows, start=1)
        ]


def _dict_value(value: Any) -> dict:
    return value if isinstance(value, dict) else {}
