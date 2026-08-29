from __future__ import annotations

from pydantic import BaseModel, Field

from app.ai.grounded_answer import Citation, GroundedAnswer


class CitationValidationResult(BaseModel):
    answer: GroundedAnswer
    valid_citation_count: int
    removed_citation_count: int
    invalid_citations: list[Citation] = Field(default_factory=list)


class CitationValidator:
    def validate(self, answer: GroundedAnswer, evidence: list[dict]) -> CitationValidationResult:
        allowed = _allowed_citations(evidence)
        valid: list[Citation] = []
        invalid: list[Citation] = []
        for citation in answer.citations:
            key = (citation.source_id, citation.chunk_id)
            source_key = (citation.source_id, None)
            if key in allowed or source_key in allowed:
                valid.append(citation)
            else:
                invalid.append(citation)

        return CitationValidationResult(
            answer=answer.model_copy(update={"citations": valid}),
            valid_citation_count=len(valid),
            removed_citation_count=len(invalid),
            invalid_citations=invalid,
        )


def _allowed_citations(evidence: list[dict]) -> set[tuple[str, str | None]]:
    allowed: set[tuple[str, str | None]] = set()
    for item in evidence:
        source_id = item.get("source_id")
        if not source_id:
            continue
        allowed.add((source_id, item.get("chunk_id")))
        allowed.add((source_id, None))
    return allowed
