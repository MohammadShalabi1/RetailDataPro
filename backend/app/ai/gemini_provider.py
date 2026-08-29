from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from app.ai.errors import (
    AIConfigurationError,
    AIProviderError,
    AIProviderResponseError,
    AIProviderTimeoutError,
    AIProviderTransientError,
    StructuredOutputValidationError,
)
from app.ai.provider import AIProvider
from app.ai.schemas import AIResponse, StructuredGenerationRequest, StructuredModelT, TextGenerationRequest
from app.core.config import Settings, get_settings

SleepFn = Callable[[float], Awaitable[None]]


class GeminiProvider(AIProvider):
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

    async def generate_text(self, request: TextGenerationRequest) -> AIResponse[str]:
        model = request.model or self._settings.gemini_text_model
        started_at = time.perf_counter()
        response = await self._run_with_timeout_and_retries(
            lambda: self._generate_content(
                model=model,
                prompt=request.prompt,
                config=self._build_text_config(),
            )
        )
        text = _extract_text(response)
        if not text:
            raise AIProviderResponseError("Gemini returned an empty text response.")
        return self._build_response(text, model, started_at, response)

    async def generate_structured(
        self,
        request: StructuredGenerationRequest[StructuredModelT],
    ) -> AIResponse[StructuredModelT]:
        model = request.model or self._settings.gemini_structured_model
        started_at = time.perf_counter()
        response = await self._run_with_timeout_and_retries(
            lambda: self._generate_content(
                model=model,
                prompt=request.prompt,
                config=self._build_structured_config(request.response_model),
            )
        )
        text = _extract_text(response)
        if not text:
            raise AIProviderResponseError("Gemini returned an empty structured response.")

        try:
            parsed = request.response_model.model_validate_json(text)
        except ValidationError as exc:
            raise StructuredOutputValidationError("Gemini structured output failed validation.") from exc

        return self._build_response(parsed, model, started_at, response)

    async def _run_with_timeout_and_retries(self, operation: Callable[[], Any]) -> Any:
        attempts = self._settings.ai_provider_max_retries + 1
        last_transient: BaseException | None = None

        for attempt in range(attempts):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(operation),
                    timeout=self._settings.ai_provider_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise AIProviderTimeoutError("Gemini request timed out.") from exc
            except Exception as exc:
                if isinstance(exc, AIProviderError):
                    raise
                if not _is_transient_error(exc):
                    raise AIProviderResponseError("Gemini request failed.") from exc
                last_transient = exc
                if attempt < attempts - 1:
                    await self._sleep(2**attempt * 0.25)

        raise AIProviderTransientError("Gemini transient failure remained after retries.") from last_transient

    def _generate_content(self, model: str, prompt: str, config: Any) -> Any:
        client = self._get_client()
        return client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self._settings.gemini_api_key:
            raise AIConfigurationError("GEMINI_API_KEY is required to create the Gemini client.")

        try:
            from google import genai
        except ImportError as exc:
            raise AIConfigurationError("google-genai is not installed.") from exc

        self._client = genai.Client(api_key=self._settings.gemini_api_key)
        return self._client

    def _build_text_config(self) -> Any:
        return _make_generate_content_config()

    def _build_structured_config(self, response_model: type[StructuredModelT]) -> Any:
        return _make_generate_content_config(
            response_mime_type="application/json",
            response_json_schema=response_model.model_json_schema(),
        )

    def _build_response(self, content: Any, model: str, started_at: float, provider_response: Any) -> AIResponse:
        usage = _extract_usage_metadata(provider_response)
        return AIResponse(
            content=content,
            model=model,
            provider=self.provider_name,
            latency_ms=max(0, round((time.perf_counter() - started_at) * 1000)),
            input_tokens=_usage_int(usage, "prompt_token_count"),
            output_tokens=_usage_int(usage, "candidates_token_count"),
            raw_usage_metadata=usage,
        )


def _make_generate_content_config(**kwargs: Any) -> Any:
    try:
        from google.genai import types
    except ImportError:
        return kwargs
    return types.GenerateContentConfig(**kwargs)


def _extract_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text is not None:
        return text
    if isinstance(response, dict):
        return str(response.get("text") or "")
    return ""


def _extract_usage_metadata(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage_metadata")
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()

    keys = ["prompt_token_count", "candidates_token_count", "total_token_count"]
    return {key: getattr(usage, key) for key in keys if getattr(usage, key, None) is not None}


def _usage_int(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key)
    return int(value or 0)


def _is_transient_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    return isinstance(exc, (ConnectionError, TimeoutError))
