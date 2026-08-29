from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

from sqlalchemy import Select, String, cast, desc, func, literal, or_, select
from sqlalchemy.orm import Session

from app.models import Source, SourceChunk
from app.schemas.documents import DocumentResponse, DocumentSearchResult


class DocumentRepository:
    def __init__(self, db: Session) -> None:
        self._db = db

    def create_document(
        self,
        title: str,
        chunks: list[str],
        uri: str | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> DocumentResponse:
        if uri:
            self._delete_existing_document_versions(uri)
        source = Source(
            title=title,
            source_type="document",
            uri=uri,
            uploaded_at=datetime.now(timezone.utc),
            metadata_={"ingestion": "document_upload", "content_hash": _content_hash(chunks)},
        )
        for index, chunk in enumerate(chunks):
            source.chunks.append(
                SourceChunk(
                    chunk_index=index,
                    content=chunk,
                    token_count=_estimate_tokens(chunk),
                    embedding=embeddings[index] if embeddings and index < len(embeddings) else None,
                    search_vector=func.to_tsvector("english", chunk),
                    metadata_={"content_hash": _content_hash([chunk])},
                )
            )
        self._db.add(source)
        self._db.commit()
        self._db.refresh(source)
        return DocumentResponse(
            source_id=str(source.id),
            title=source.title,
            chunk_count=len(source.chunks),
            uploaded_at=source.uploaded_at,
        )

    def search_documents(self, query: str, limit: int, source_ids: list[str] | None = None) -> list[DocumentSearchResult]:
        return [_to_result(row, index) for index, row in enumerate(self._fetch_all(self.build_search_statement(query, limit, source_ids)), start=1)]

    def get_recent_document_chunks(self, limit: int, source_ids: list[str] | None = None) -> list[DocumentSearchResult]:
        statement = (
            select(
                cast(Source.id, String).label("source_id"),
                cast(SourceChunk.id, String).label("chunk_id"),
                Source.title,
                SourceChunk.chunk_index,
                SourceChunk.content,
                literal(0.0).label("score"),
            )
            .select_from(SourceChunk)
            .join(Source, Source.id == SourceChunk.source_id)
            .where(*self._scope_filters(source_ids))
            .order_by(Source.uploaded_at.desc(), SourceChunk.chunk_index)
            .limit(limit)
        )
        return [_to_result(row, index) for index, row in enumerate(self._fetch_all(statement), start=1)]

    def build_search_statement(self, query: str, limit: int, source_ids: list[str] | None = None) -> Select:
        tsquery = func.plainto_tsquery("english", query)
        rank = func.coalesce(func.ts_rank(SourceChunk.search_vector, tsquery), 0).label("score")
        return (
            select(
                cast(Source.id, String).label("source_id"),
                cast(SourceChunk.id, String).label("chunk_id"),
                Source.title,
                SourceChunk.chunk_index,
                SourceChunk.content,
                rank,
            )
            .select_from(SourceChunk)
            .join(Source, Source.id == SourceChunk.source_id)
            .where(
                *self._scope_filters(source_ids),
                or_(SourceChunk.search_vector.op("@@")(tsquery), SourceChunk.content.ilike(f"%{query}%")),
            )
            .order_by(desc("score"), Source.uploaded_at.desc(), SourceChunk.chunk_index)
            .limit(limit)
        )

    def _fetch_all(self, statement: Select) -> list[dict[str, Any]]:
        return [dict(row._mapping) for row in self._db.execute(statement).all()]

    def authorize_source_ids(self, source_ids: list[str]) -> tuple[list[str], list[str]]:
        parsed = [str(value) for value in source_ids if _parse_uuid(value) is not None]
        if not parsed:
            return [], [value for value in source_ids if _parse_uuid(value) is None]
        rows = self._db.scalars(
            select(cast(Source.id, String)).where(Source.source_type == "document", cast(Source.id, String).in_(parsed))
        ).all()
        allowed = [str(row) for row in rows]
        rejected = [value for value in source_ids if value not in allowed]
        return allowed, rejected

    def _scope_filters(self, source_ids: list[str] | None = None) -> list[Any]:
        filters: list[Any] = [Source.source_type == "document"]
        parsed_ids = [_parse_uuid(value) for value in source_ids or []]
        scoped_ids = [value for value in parsed_ids if value is not None]
        if scoped_ids:
            filters.append(Source.id.in_(scoped_ids))
        return filters

    def _delete_existing_document_versions(self, uri: str) -> None:
        existing_ids = self._db.scalars(select(Source.id).where(Source.source_type == "document", Source.uri == uri)).all()
        if existing_ids:
            self._db.query(Source).filter(Source.id.in_(existing_ids)).delete(synchronize_session=False)


def _to_result(row: dict[str, Any], fallback_rank: int) -> DocumentSearchResult:
    return DocumentSearchResult(
        source_id=row["source_id"],
        chunk_id=row["chunk_id"],
        title=row["title"],
        chunk_index=row["chunk_index"],
        content=row["content"],
        score=float(row.get("score") or (1.0 / fallback_rank)),
    )


def _estimate_tokens(value: str) -> int:
    return max(1, len(value.split()))


def _parse_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


def _content_hash(chunks: list[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(chunks).encode("utf-8")).hexdigest()
