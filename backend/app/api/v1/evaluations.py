from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.security import require_development
from app.evals.runner import DeterministicEvalRunner, EvalRunResult

router = APIRouter(prefix="/evaluations", tags=["evaluations"], dependencies=[Depends(require_development)])


@router.get("/latest", response_model=EvalRunResult)
def latest_evaluation() -> EvalRunResult:
    return DeterministicEvalRunner().run()
