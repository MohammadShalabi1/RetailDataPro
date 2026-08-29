from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.documents import DocumentCreateRequest, DocumentResponse, DocumentSearchResponse, DocumentSearchResult
from app.services.document_service import DocumentService, get_document_service
from app.tools.gateway import authorize_and_execute_tool
from app.tools.schemas import ToolExecutionContext, ToolExecutionRequest, ToolStatus
from app.ai.embedding_provider import EmbeddingResponse


class FakeDocumentRepository:
    def __init__(self) -> None:
        self.created: list[tuple[str, list[str], str | None]] = []
        self.recent_calls: list[int] = []
        self.search_scopes: list[list[str]] = []

    def create_document(self, title: str, chunks: list[str], uri: str | None = None, embeddings=None) -> DocumentResponse:
        self.created.append((title, chunks, uri))
        self.embeddings = embeddings
        return DocumentResponse(
            source_id="src_1",
            title=title,
            chunk_count=len(chunks),
            uploaded_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
        )

    def search_documents(self, query: str, limit: int, source_ids: list[str] | None = None) -> list[DocumentSearchResult]:
        self.search_scopes.append(source_ids or [])
        return [
            DocumentSearchResult(
                source_id="src_1",
                chunk_id="chk_1",
                title="Supplier report",
                chunk_index=0,
                content="Supplier lead times improved while margin pressure remained visible.",
                score=0.88,
            )
        ][:limit]

    def get_recent_document_chunks(self, limit: int, source_ids: list[str] | None = None) -> list[DocumentSearchResult]:
        self.recent_calls.append(limit)
        return [
            DocumentSearchResult(
                source_id="src_recent",
                chunk_id="chk_recent",
                title="Latest PDF",
                chunk_index=0,
                content="The latest uploaded PDF says supplier risk is improving.",
                score=0.0,
            )
        ][:limit]

    def authorize_source_ids(self, source_ids: list[str]) -> tuple[list[str], list[str]]:
        return source_ids, []


class EmptySearchDocumentRepository(FakeDocumentRepository):
    def search_documents(self, query: str, limit: int, source_ids: list[str] | None = None) -> list[DocumentSearchResult]:
        self.search_scopes.append(source_ids or [])
        return []


class FakeDocumentService:
    def create_document(self, request: DocumentCreateRequest) -> DocumentResponse:
        return DocumentResponse(
            source_id="src_route",
            title=request.title,
            chunk_count=2,
            uploaded_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
        )

    async def create_document_async(self, request: DocumentCreateRequest) -> DocumentResponse:
        return self.create_document(request)

    async def create_document_from_upload(
        self,
        title: str,
        filename: str,
        content_type: str | None,
        data: bytes,
    ) -> DocumentResponse:
        assert filename
        assert content_type
        assert data
        return DocumentResponse(
            source_id="src_upload",
            title=title,
            chunk_count=3,
            uploaded_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
        )

    async def search_documents_async(self, query: str, limit: int = 5, source_ids: list[str] | None = None) -> DocumentSearchResponse:
        return self.search_documents(query, limit, source_ids)

    def search_documents(self, query: str, limit: int = 5, source_ids: list[str] | None = None) -> DocumentSearchResponse:
        return DocumentSearchResponse(
            query=query,
            source_ids=source_ids or [],
            results=[
                DocumentSearchResult(
                    source_id="src_route",
                    chunk_id="chk_route",
                    title="Supplier report",
                    chunk_index=0,
                    content="The supplier report says replenishment risk is moderate.",
                    score=0.91,
                )
            ],
            confidence=0.75,
            retrieval_trace={"normalized_query": query},
        )


class FakeEmbeddingProvider:
    async def embed_text(self, text: str) -> EmbeddingResponse:
        return EmbeddingResponse(embedding=[0.2] * 768, model="fake-embedding", provider="fake", latency_ms=1)

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResponse]:
        return [await self.embed_text(text) for text in texts]


def test_document_service_chunks_and_stores_content() -> None:
    repository = FakeDocumentRepository()
    service = DocumentService(repository)  # type: ignore[arg-type]
    content = " ".join(f"word{index}" for index in range(370))

    response = service.create_document(DocumentCreateRequest(title="Long report", content=content))

    assert response.chunk_count == 3
    assert repository.created[0][0] == "Long report"
    assert [len(chunk.split()) for chunk in repository.created[0][1]] == [180, 180, 10]


@pytest.mark.asyncio
async def test_document_service_accepts_text_file_upload() -> None:
    repository = FakeDocumentRepository()
    service = DocumentService(repository)  # type: ignore[arg-type]

    response = await service.create_document_from_upload(
        title="Supplier notes",
        filename="supplier-notes.txt",
        content_type="text/plain",
        data=b"Supplier lead times improved in August.",
    )

    assert response.title == "Supplier notes"
    assert repository.created[0][2] == "supplier-notes.txt"
    assert repository.created[0][1] == ["Supplier lead times improved in August."]


@pytest.mark.asyncio
async def test_document_service_rejects_unsupported_upload_type() -> None:
    service = DocumentService(FakeDocumentRepository())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Supported uploads"):
        await service.create_document_from_upload(
            title="Image",
            filename="image.png",
            content_type="image/png",
            data=b"not text",
        )


@pytest.mark.asyncio
async def test_document_service_embeds_chunks_when_provider_is_configured() -> None:
    repository = FakeDocumentRepository()
    service = DocumentService(repository, embedding_provider=FakeEmbeddingProvider())  # type: ignore[arg-type]

    response = await service.create_document_async(DocumentCreateRequest(title="Embedded", content="one two three"))

    assert response.chunk_count == 1
    assert repository.embeddings == [[0.2] * 768]


def test_document_service_uses_recent_chunks_when_query_has_no_keyword_match() -> None:
    repository = EmptySearchDocumentRepository()
    service = DocumentService(repository)  # type: ignore[arg-type]

    response = service.search_documents("What does this PDF say?", limit=5)

    assert repository.recent_calls == [5]
    assert response.results[0].title == "Latest PDF"
    assert "supplier risk" in response.results[0].content


def test_document_service_passes_source_scope_to_retrieval() -> None:
    repository = FakeDocumentRepository()
    service = DocumentService(repository)  # type: ignore[arg-type]

    response = service.search_documents("Which supplier had weakest fulfillment reliability?", limit=5, source_ids=["source-a"])

    assert repository.search_scopes == [["source-a"]]
    assert response.source_ids == ["source-a"]


def test_document_routes_accept_upload_and_search_without_live_database() -> None:
    app = create_app()
    app.dependency_overrides[get_document_service] = lambda: FakeDocumentService()
    client = TestClient(app)

    try:
        upload_response = client.post(
            "/api/documents",
            json={"title": "Supplier report", "content": "Supplier lead time and margin notes."},
        )
        search_response = client.get("/api/documents/search?q=supplier%20report&limit=3&source_ids=src_route")
    finally:
        app.dependency_overrides.clear()

    assert upload_response.status_code == 200
    assert upload_response.json()["title"] == "Supplier report"
    assert upload_response.json()["chunk_count"] == 2
    assert search_response.status_code == 200
    assert search_response.json()["results"][0]["content"].startswith("The supplier report says")


def test_document_upload_route_accepts_multipart_pdf_without_live_database() -> None:
    app = create_app()
    app.dependency_overrides[get_document_service] = lambda: FakeDocumentService()
    client = TestClient(app)

    try:
        response = client.post(
            "/api/documents/upload",
            data={"title": "Supplier PDF"},
            files={"file": ("supplier-report.pdf", b"%PDF fake bytes", "application/pdf")},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["source_id"] == "src_upload"
    assert response.json()["title"] == "Supplier PDF"
    assert response.json()["chunk_count"] == 3


@pytest.mark.asyncio
async def test_document_search_tool_returns_typed_chunks_through_gateway() -> None:
    result = await authorize_and_execute_tool(
        ToolExecutionRequest(tool_name="document_search", input={"question": "What does the supplier report say?"}),
        ToolExecutionContext(user_role="analyst", document_service=FakeDocumentService()),
    )

    assert result.status == ToolStatus.success
    assert result.authorized is True
    assert result.error_code is None
    assert result.output["query"] == "What does the supplier report say?"
    assert result.output["chunks"][0]["title"] == "Supplier report"
    assert result.output["confidence"] == 0.75
    assert result.output["retrieval_trace"]["normalized_query"] == "What does the supplier report say?"
