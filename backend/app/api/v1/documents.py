from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from app.core.security import ClientContext, get_client_context
from app.schemas.documents import DocumentCreateRequest, DocumentListItem, DocumentResponse, DocumentSearchResponse
from app.services.document_service import DocumentService, get_document_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentListItem])
def list_documents(
    client: ClientContext = Depends(get_client_context),
    document_service: DocumentService = Depends(get_document_service),
) -> list[DocumentListItem]:
    return document_service.list_documents(client.client_id)


@router.post("", response_model=DocumentResponse)
async def create_document(
    request: DocumentCreateRequest,
    client: ClientContext = Depends(get_client_context),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    return await document_service.create_document_async(request, client.client_id)


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    title: str = Form(min_length=1, max_length=220),
    file: UploadFile = File(...),
    client: ClientContext = Depends(get_client_context),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    data = await file.read()
    try:
        return await document_service.create_document_from_upload(title, file.filename or title, file.content_type, data, client.client_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/search", response_model=DocumentSearchResponse)
async def search_documents(
    q: str = Query(min_length=1, max_length=500),
    limit: int = Query(default=5, ge=1, le=10),
    source_ids: list[str] = Query(default_factory=list),
    client: ClientContext = Depends(get_client_context),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentSearchResponse:
    return await document_service.search_documents_async(q, limit, source_ids, client.client_id)
