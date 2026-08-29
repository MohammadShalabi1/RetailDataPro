from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from app.ai.errors import AIProviderError
from app.ai.provider import AIProvider
from app.ai.schemas import AIResponse, StructuredGenerationRequest

T = TypeVar("T", bound=BaseModel)


class StructuredOutputRepairer:
    def __init__(self, ai_provider: AIProvider) -> None:
        self._ai_provider = ai_provider

    async def generate_with_one_repair(
        self,
        request: StructuredGenerationRequest[type[T]],
        fallback: T,
    ) -> tuple[AIResponse[T], bool]:
        try:
            return await self._ai_provider.generate_structured(request), False
        except AIProviderError:
            repair_prompt = (
                "The previous structured response was invalid. Return exactly one valid object "
                f"for this schema: {request.response_model.__name__}.\nOriginal prompt:\n{request.prompt}"
            )
            try:
                return await self._ai_provider.generate_structured(
                    StructuredGenerationRequest(prompt=repair_prompt, response_model=request.response_model, model=request.model)
                ), True
            except AIProviderError:
                return AIResponse(content=fallback, model=request.model or "fallback", provider="safe_fallback", latency_ms=0), True
