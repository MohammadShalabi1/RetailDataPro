from __future__ import annotations

from pydantic import BaseModel, Field

from app.ai.provider import AIProvider
from app.ai.schemas import StructuredGenerationRequest


class Citation(BaseModel):
    source_id: str
    chunk_id: str | None = None
    claim: str = Field(min_length=1, max_length=500)


class GroundedAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=4_000)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    limitations: list[str] = Field(default_factory=list)


class GroundingEvidence(BaseModel):
    tool_results: list[dict] = Field(default_factory=list)
    retrieved_chunks: list[dict] = Field(default_factory=list)
    source_metadata: list[dict] = Field(default_factory=list)


class GroundedAnswerGenerator:
    def __init__(self, ai_provider: AIProvider, prompt_version: str = "grounded-answer-v1") -> None:
        self._ai_provider = ai_provider
        self.prompt_version = prompt_version

    async def generate(self, question: str, evidence: GroundingEvidence, model: str | None = None) -> GroundedAnswer:
        if not evidence.tool_results and not evidence.retrieved_chunks:
            return GroundedAnswer(
                answer="I do not have enough verified evidence to answer that.",
                citations=[],
                confidence=0.0,
                limitations=["No tool results or retrieved evidence were provided."],
            )

        response = await self._ai_provider.generate_structured(
            StructuredGenerationRequest(
                prompt=_prompt(question, evidence, self.prompt_version),
                response_model=GroundedAnswer,
                model=model,
            )
        )
        answer = response.content
        allowed_citations = _allowed_citations(evidence)
        invalid = [
            citation
            for citation in answer.citations
            if (citation.source_id, citation.chunk_id) not in allowed_citations and (citation.source_id, None) not in allowed_citations
        ]
        if invalid:
            return GroundedAnswer(
                answer="I do not have enough verified evidence to answer that with valid citations.",
                citations=[],
                confidence=0.0,
                limitations=["The model returned citations that were not present in the provided evidence."],
            )
        return answer


def _prompt(question: str, evidence: GroundingEvidence, prompt_version: str) -> str:
    return (
        f"Prompt version: {prompt_version}\n"
        "Answer only from provided tool results and retrieved chunks. "
        "Citations must map to the supplied source_id/chunk_id values. "
        "Admit when evidence is insufficient.\n"
        f"Evidence: {evidence.model_dump(mode='json')}\n"
        f"Question: {question}"
    )


def _allowed_citations(evidence: GroundingEvidence) -> set[tuple[str, str | None]]:
    allowed: set[tuple[str, str | None]] = set()
    for chunk in evidence.retrieved_chunks:
        source_id = chunk.get("source_id")
        chunk_id = chunk.get("chunk_id")
        if source_id:
            allowed.add((source_id, chunk_id))
    for item in evidence.source_metadata:
        source_id = item.get("source_id")
        if source_id:
            allowed.add((source_id, None))
    return allowed
