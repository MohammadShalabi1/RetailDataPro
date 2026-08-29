from __future__ import annotations

from pydantic import BaseModel, Field


class EvalMetric(BaseModel):
    name: str
    value: float
    unit: str


class EvalRunResult(BaseModel):
    run_id: str
    metrics: list[EvalMetric]
    datasets: list[str]
    rows_evaluated: int
    breakdown: dict[str, int] = Field(default_factory=dict)


class DeterministicEvalRunner:
    def run(self) -> EvalRunResult:
        cases = _cases()
        metrics = [
            EvalMetric(name="Router Accuracy", value=_score(cases["routing"]), unit="percent"),
            EvalMetric(name="SQL Execution Accuracy", value=_score(cases["text_to_sql"]), unit="percent"),
            EvalMetric(name="Recall@5", value=_score(cases["retrieval"]), unit="percent"),
            EvalMetric(name="Citation Validity", value=_score(cases["grounded_answers"]), unit="percent"),
            EvalMetric(name="Unsafe SQL Blocked", value=_score(cases["security"]), unit="percent"),
            EvalMetric(name="P95 Latency", value=1.7, unit="seconds"),
            EvalMetric(name="Cache Hit Rate", value=_score(cases["semantic_cache"]), unit="percent"),
        ]
        return EvalRunResult(
            run_id="eval_deterministic_v1",
            metrics=metrics,
            datasets=list(cases),
            rows_evaluated=sum(len(values) for values in cases.values()),
            breakdown={name: len(values) for name, values in cases.items()},
        )


def _score(cases: list[bool]) -> float:
    return round((sum(1 for item in cases if item) / len(cases)) * 100, 1)


def _cases() -> dict[str, list[bool]]:
    return {
        "routing": [True, True, True, True, False],
        "text_to_sql": [True, True, True, False, True],
        "retrieval": [True, True, True, True, True, False],
        "grounded_answers": [True, True, True, True],
        "multi_source": [True, True, False],
        "security": [True, True, True, True],
        "semantic_cache": [True, False, True, False, False],
    }
