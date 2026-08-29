from __future__ import annotations

from abc import ABC, abstractmethod

from app.ai.schemas import AIResponse, StructuredGenerationRequest, StructuredModelT, TextGenerationRequest


class AIProvider(ABC):
    @abstractmethod
    async def generate_text(self, request: TextGenerationRequest) -> AIResponse[str]:
        raise NotImplementedError

    @abstractmethod
    async def generate_structured(
        self,
        request: StructuredGenerationRequest[StructuredModelT],
    ) -> AIResponse[StructuredModelT]:
        raise NotImplementedError
