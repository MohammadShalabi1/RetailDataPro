from __future__ import annotations

from fastapi import APIRouter

from app.evals.runner import DeterministicEvalRunner, EvalRunResult

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.get("/latest", response_model=EvalRunResult)
def latest_evaluation() -> EvalRunResult:
    return DeterministicEvalRunner().run()
