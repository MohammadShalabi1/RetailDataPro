from __future__ import annotations

from typing import Any

from sqlalchemy import Select, String, cast, desc, func, select
from sqlalchemy.orm import Session

from app.ai.retrieval import RetrievedChunk
from app.models import Source, SourceChunk


class RetrievalRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def dense_search(self, query_embedding: list[float], limit: int = 20, source_ids: list[str] | None = None) -> list[RetrievedChunk]:
        return self._fetch_chunks(self.build_dense_statement(query_embedding, limit, source_ids), "dense")

    def lexical_search(self, query: str, limit: int = 20, source_ids: list[str] | None = None) -> list[RetrievedChunk]:
        return self._fetch_chunks(self.build_lexical_statement(query, limit, source_ids), "lexical")

    def build_dense_statement(self, query_embedding: list[float], limit: int = 20, source_ids: list[str] | None = None) -> Select:
        distance = SourceChunk.embedding.op("<=>")(query_embedding).label("score")
        return (
            select(
                cast(SourceChunk.id, String).label("chunk_id"),
                cast(Source.id, String).label("source_id"),
                cast(Source.id, String).label("document_id"),
                SourceChunk.content,
                Source.title.label("source_title"),
                distance,
                SourceChunk.metadata_.label("metadata"),
                SourceChunk.token_count,
            )
            .select_from(SourceChunk)
            .join(Source, Source.id == SourceChunk.source_id)
            .where(SourceChunk.embedding.is_not(None), *_source_filters(source_ids))
            .order_by("score")
            .limit(limit)
        )

    def build_lexical_statement(self, query: str, limit: int = 20, source_ids: list[str] | None = None) -> Select:
        tsquery = func.websearch_to_tsquery("english", query)
        rank = func.ts_rank_cd(SourceChunk.search_vector, tsquery).label("score")
        return (
            select(
                cast(SourceChunk.id, String).label("chunk_id"),
                cast(Source.id, String).label("source_id"),
                cast(Source.id, String).label("document_id"),
                SourceChunk.content,
                Source.title.label("source_title"),
                rank,
                SourceChunk.metadata_.label("metadata"),
                SourceChunk.token_count,
            )
            .select_from(SourceChunk)
            .join(Source, Source.id == SourceChunk.source_id)
            .where(SourceChunk.search_vector.op("@@")(tsquery), *_source_filters(source_ids))
            .order_by(desc("score"))
            .limit(limit)
        )

    def _fetch_chunks(self, statement: Select, method: str) -> list[RetrievedChunk]:
        rows = [dict(row._mapping) for row in self._db.execute(statement).all()]
        return [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                source_id=row["source_id"],
                document_id=row.get("document_id"),
                content=row["content"],
                source_title=row["source_title"],
                initial_rank=index,
                score=float(row.get("score") or 0.0),
                retrieval_method=method,
                dense_rank=index if method == "dense" else None,
                dense_score=float(row.get("score") or 0.0) if method == "dense" else None,
                lexical_rank=index if method == "lexical" else None,
                lexical_score=float(row.get("score") or 0.0) if method == "lexical" else None,
                page_number=_page_number(row.get("metadata")),
                token_count=int(row.get("token_count") or 0),
                metadata=_dict_value(row.get("metadata")),
            )
            for index, row in enumerate(rows, start=1)
        ]


def _dict_value(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _source_filters(source_ids: list[str] | None) -> list[Any]:
    filters: list[Any] = [Source.source_type == "document"]
    if source_ids:
        filters.append(cast(Source.id, String).in_(source_ids))
    return filters


def _page_number(metadata: Any) -> int | None:
    data = _dict_value(metadata)
    value = data.get("page_number")
    return int(value) if value is not None else None
