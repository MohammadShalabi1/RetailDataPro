from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from app.ai.errors import AIConfigurationError, AIProviderResponseError, AIProviderTimeoutError
from app.core.config import Settings, get_settings

SleepFn = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class EmbeddingResponse:
    embedding: list[float]
    model: str
    provider: str
    latency_ms: int


class EmbeddingProvider(Protocol):
    async def embed_text(self, text: str) -> EmbeddingResponse:
        ...

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResponse]:
        ...


class GeminiEmbeddingProvider:
    provider_name = "gemini"

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        sleep: SleepFn = asyncio.sleep,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._sleep = sleep

    async def embed_text(self, text: str) -> EmbeddingResponse:
        responses = await self.embed_batch([text])
        return responses[0]

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResponse]:
        if not texts:
            return []
        started_at = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self._embed_content, texts),
                timeout=self._settings.ai_provider_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise AIProviderTimeoutError("Gemini embedding request timed out.") from exc
        except Exception as exc:
            if isinstance(exc, AIConfigurationError):
                raise
            raise AIProviderResponseError("Gemini embedding request failed.") from exc

        vectors = _extract_embeddings(response)
        if len(vectors) != len(texts):
            raise AIProviderResponseError("Gemini returned an unexpected embedding count.")
        for vector in vectors:
            _validate_embedding(vector, self._settings.embedding_dimension)

        latency_ms = max(0, round((time.perf_counter() - started_at) * 1000))
        return [
            EmbeddingResponse(
                embedding=vector,
                model=self._settings.gemini_embedding_model,
                provider=self.provider_name,
                latency_ms=latency_ms,
            )
            for vector in vectors
        ]

    def _embed_content(self, texts: list[str]) -> Any:
        client = self._get_client()
        return client.models.embed_content(
            model=self._settings.gemini_embedding_model,
            contents=texts,
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._settings.gemini_api_key:
            raise AIConfigurationError("GEMINI_API_KEY is required to create the Gemini embedding client.")
        try:
            from google import genai
        except ImportError as exc:
            raise AIConfigurationError("google-genai is not installed.") from exc
        self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client


def _extract_embeddings(response: Any) -> list[list[float]]:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None and isinstance(response, dict):
        embeddings = response.get("embeddings")
    if embeddings is None:
        embedding = getattr(response, "embedding", None)
        if embedding is not None:
            embeddings = [embedding]
    vectors: list[list[float]] = []
    for item in embeddings or []:
        values = getattr(item, "values", None)
        if values is None and isinstance(item, dict):
            values = item.get("values") or item.get("embedding")
        if values is None and isinstance(item, list):
            values = item
        vectors.append([float(value) for value in values or []])
    return vectors


def _validate_embedding(vector: list[float], dimensions: int) -> None:
    if len(vector) != dimensions:
        raise AIProviderResponseError(f"Embedding dimension mismatch: expected {dimensions}, got {len(vector)}.")
