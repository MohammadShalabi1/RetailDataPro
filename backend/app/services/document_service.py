from __future__ import annotations

from io import BytesIO

from fastapi import Depends
from pypdf import PdfReader
from sqlalchemy.orm import Session

from app.ai.reranker import LexicalCrossEncoderReranker
from app.ai.retrieval import RetrievedChunk, reciprocal_rank_fusion
from app.ai.embedding_provider import EmbeddingProvider, GeminiEmbeddingProvider
from app.ai.errors import AIProviderError
from app.ai.hybrid_retrieval import HybridRetrievalService
from app.database.session import get_db
from app.repositories.document_repository import DocumentRepository
from app.repositories.retrieval_repository import RetrievalRepository
from app.schemas.documents import DocumentCreateRequest, DocumentListItem, DocumentResponse, DocumentSearchResponse, DocumentSearchResult

DEFAULT_CHUNK_WORDS = 180
MAX_SEARCH_LIMIT = 10
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
PDF_MIME_TYPES = {"application/pdf", "application/x-pdf"}
TEXT_EXTENSIONS = (".txt", ".md", ".csv", ".json")


class DocumentService:
    def __init__(
        self,
        repository: DocumentRepository,
        embedding_provider: EmbeddingProvider | None = None,
        retrieval_repository: RetrievalRepository | None = None,
    ) -> None:
        self._repository = repository
        self._embedding_provider = embedding_provider
        self._retrieval_repository = retrieval_repository
        self._reranker = LexicalCrossEncoderReranker()

    def create_document(self, request: DocumentCreateRequest, client_id: str = "single-client") -> DocumentResponse:
        chunks = _chunk_text(request.content)
        return self._repository.create_document(request.title, chunks, request.uri, client_id=client_id)

    async def create_document_async(self, request: DocumentCreateRequest, client_id: str = "single-client") -> DocumentResponse:
        chunks = _chunk_text(request.content)
        embeddings = await self._embed_chunks(chunks)
        return self._repository.create_document(request.title, chunks, request.uri, embeddings, client_id)

    async def create_document_from_upload(
        self,
        title: str,
        filename: str,
        content_type: str | None,
        data: bytes,
        client_id: str = "single-client",
    ) -> DocumentResponse:
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("Document uploads must be 10 MB or smaller.")

        text = _extract_uploaded_text(filename, content_type, data)
        if not text.strip():
            raise ValueError("No readable text was found in the uploaded document.")

        chunks = _chunk_text(text)
        embeddings = await self._embed_chunks(chunks)
        return self._repository.create_document(title, chunks, filename, embeddings, client_id)

    def list_documents(self, client_id: str = "single-client") -> list[DocumentListItem]:
        return [DocumentListItem.model_validate(row) for row in self._repository.list_documents(client_id)]

    def search_documents(
        self,
        query: str,
        limit: int = 5,
        source_ids: list[str] | None = None,
        client_id: str = "single-client",
    ) -> DocumentSearchResponse:
        resolved_limit = min(max(limit, 1), MAX_SEARCH_LIMIT)
        scope = source_ids or self._repository.get_authorized_document_source_ids(client_id)
        if self._embedding_provider is not None and self._retrieval_repository is not None:
            raise RuntimeError("Use search_documents_async when hybrid retrieval is configured.")
        lexical_results = self._repository.search_documents(query, resolved_limit, scope, client_id)
        hybrid = reciprocal_rank_fusion([], [_to_retrieved_chunk(result, "lexical") for result in lexical_results], limit=resolved_limit)
        results = [_from_reranked_chunk(chunk) for chunk in self._reranker.rerank(query, hybrid.chunks, top_k=resolved_limit)]
        if not results:
            results = self._repository.get_recent_document_chunks(resolved_limit, scope, client_id)
        return DocumentSearchResponse(query=query, results=results, source_ids=scope, confidence=0.45 if results else 0.0)

    async def search_documents_async(
        self,
        query: str,
        limit: int = 5,
        source_ids: list[str] | None = None,
        client_id: str = "single-client",
    ) -> DocumentSearchResponse:
        resolved_limit = min(max(limit, 1), MAX_SEARCH_LIMIT)
        requested_scope = source_ids or self._repository.get_authorized_document_source_ids(client_id)
        if not requested_scope:
            return DocumentSearchResponse(
                query=query,
                results=[],
                source_ids=[],
                limitations=["No client documents are available yet."],
            )
        allowed_scope, rejected_scope = self._repository.authorize_source_ids(requested_scope, client_id)
        if rejected_scope or not allowed_scope:
            return DocumentSearchResponse(
                query=query,
                results=[],
                source_ids=allowed_scope,
                limitations=["One or more requested document sources were not authorized."],
            )
        if self._embedding_provider is None or self._retrieval_repository is None:
            return self.search_documents(query, resolved_limit, allowed_scope)

        retrieval = HybridRetrievalService(self._retrieval_repository, self._embedding_provider)
        result = await retrieval.search(query, allowed_scope)
        rows = [
            DocumentSearchResult(
                source_id=chunk.source_id,
                chunk_id=chunk.chunk_id,
                title=chunk.source_title,
                chunk_index=int(chunk.metadata.get("chunk_index", 0)),
                content=chunk.content,
                score=chunk.rerank_score,
            )
            for chunk in result.chunks
        ]
        return DocumentSearchResponse(
            query=query,
            results=rows,
            source_ids=allowed_scope,
            confidence=result.confidence,
            limitations=result.limitations,
            retrieval_trace=result.trace.model_dump(mode="json"),
        )

    async def _embed_chunks(self, chunks: list[str]) -> list[list[float]] | None:
        if self._embedding_provider is None:
            return None
        try:
            responses = await self._embedding_provider.embed_batch(chunks)
        except AIProviderError:
            return None
        return [response.embedding for response in responses]


def get_document_service(db: Session = Depends(get_db)) -> DocumentService:
    return DocumentService(DocumentRepository(db), GeminiEmbeddingProvider(), RetrievalRepository(db))


def _chunk_text(content: str, chunk_words: int = DEFAULT_CHUNK_WORDS) -> list[str]:
    words = content.split()
    if not words:
        return [content]
    return [" ".join(words[index : index + chunk_words]) for index in range(0, len(words), chunk_words)]


def _extract_uploaded_text(filename: str, content_type: str | None, data: bytes) -> str:
    lower_name = filename.lower()
    if content_type in PDF_MIME_TYPES or lower_name.endswith(".pdf"):
        return _extract_pdf_text(data)
    if (content_type or "").startswith("text/") or lower_name.endswith(TEXT_EXTENSIONS):
        return data.decode("utf-8", errors="replace")
    raise ValueError("Supported uploads are PDF, TXT, Markdown, CSV, and JSON files.")


def _extract_pdf_text(data: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise ValueError("The uploaded PDF could not be parsed.") from exc
    return "\n\n".join(page.strip() for page in pages if page.strip())


def _to_retrieved_chunk(result: DocumentSearchResult, method: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=result.chunk_id,
        source_id=result.source_id,
        content=result.content,
        source_title=result.title,
        initial_rank=result.chunk_index + 1,
        score=result.score,
        retrieval_method=method,
        metadata={"chunk_index": result.chunk_index},
    )


def _from_reranked_chunk(chunk: RetrievedChunk) -> DocumentSearchResult:
    return DocumentSearchResult(
        source_id=chunk.source_id,
        chunk_id=chunk.chunk_id,
        title=chunk.source_title,
        chunk_index=int(chunk.metadata.get("chunk_index", 0)),
        content=chunk.content,
        score=chunk.score,
    )
