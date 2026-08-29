from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar

from app.ai.errors import AIProviderError, AIProviderTimeoutError
from app.ai.provider import AIProvider
from app.ai.schemas import AIResponse, StructuredGenerationRequest, TextGenerationRequest

T = TypeVar("T")


@dataclass
class ModelFallbackTrace:
    events: list[dict] = field(default_factory=list)

    def record(self, event: dict) -> None:
        self.events.append(event)


class FallbackAIProvider(AIProvider):
    def __init__(self, primary: AIProvider, fallback: AIProvider, fallback_model: str) -> None:
        self._primary = primary
        self._fallback = fallback
        self._fallback_model = fallback_model
        self.trace = ModelFallbackTrace()

    async def generate_text(self, request: TextGenerationRequest) -> AIResponse[str]:
        try:
            return await self._primary.generate_text(request)
        except AIProviderTimeoutError as exc:
            return await self._fallback_text(request, "timeout", exc)
        except AIProviderError as exc:
            return await self._fallback_text(request, "provider_error", exc)

    async def generate_structured(self, request: StructuredGenerationRequest[type[T]]) -> AIResponse[T]:
        try:
            return await self._primary.generate_structured(request)
        except AIProviderTimeoutError as exc:
            return await self._fallback_structured(request, "timeout", exc)
        except AIProviderError as exc:
            return await self._fallback_structured(request, "provider_error", exc)

    async def _fallback_text(self, request: TextGenerationRequest, reason: str, exc: Exception) -> AIResponse[str]:
        self.trace.record({"primary_model": request.model, "fallback_model": self._fallback_model, "failure_reason": reason})
        return await self._fallback.generate_text(TextGenerationRequest(prompt=request.prompt, model=self._fallback_model))

    async def _fallback_structured(
        self,
        request: StructuredGenerationRequest[type[T]],
        reason: str,
        exc: Exception,
    ) -> AIResponse[T]:
        self.trace.record({"primary_model": request.model, "fallback_model": self._fallback_model, "failure_reason": reason})
        return await self._fallback.generate_structured(
            StructuredGenerationRequest(prompt=request.prompt, response_model=request.response_model, model=self._fallback_model)
        )
