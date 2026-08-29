from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.services.observability_service import TraceRecord, observability_service

router = APIRouter(prefix="/observability", tags=["observability"])


@router.get("/traces", response_model=list[TraceRecord])
def list_traces() -> list[TraceRecord]:
    return observability_service.list_traces()


@router.get("/traces/{trace_id}", response_model=TraceRecord)
def get_trace(trace_id: str) -> TraceRecord:
    trace = observability_service.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace
