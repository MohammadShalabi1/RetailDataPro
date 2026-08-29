from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=220)
    content: str = Field(min_length=1, max_length=100_000)
    uri: str | None = Field(default=None, max_length=500)


class DocumentResponse(BaseModel):
    source_id: str
    title: str
    chunk_count: int
    uploaded_at: datetime


class DocumentSearchResult(BaseModel):
    source_id: str
    chunk_id: str
    title: str
    chunk_index: int
    content: str
    score: float


class DocumentSearchResponse(BaseModel):
    query: str
    results: list[DocumentSearchResult]
    source_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)
    retrieval_trace: dict = Field(default_factory=dict)
