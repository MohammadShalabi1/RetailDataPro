from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from app.ai.errors import (
    AIConfigurationError,
    AIProviderResponseError,
    AIProviderTimeoutError,
    AIProviderTransientError,
    StructuredOutputValidationError,
)
from app.ai.gemini_provider import GeminiProvider
from app.ai.schemas import StructuredGenerationRequest, TextGenerationRequest
from app.core.config import Settings


class DemoStructuredResponse(BaseModel):
    answer: str
    score: int


@dataclass
class FakeUsage:
    prompt_token_count: int
    candidates_token_count: int
    total_token_count: int


@dataclass
class FakeResponse:
    text: str
    usage_metadata: FakeUsage | dict | None = None


class FakeModels:
    def __init__(self, responses=None, errors=None) -> None:
        self.responses = list(responses or [])
        self.errors = list(errors or [])
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses=None, errors=None) -> None:
        self.models = FakeModels(responses=responses, errors=errors)


class TransientProviderError(Exception):
    status_code = 503


class NonTransientProviderError(Exception):
    status_code = 400


@pytest.mark.asyncio
async def test_generate_text_success_returns_metadata() -> None:
    client = FakeClient(
        responses=[
            FakeResponse(
                text="Retail revenue is stable.",
                usage_metadata=FakeUsage(
                    prompt_token_count=12,
                    candidates_token_count=7,
                    total_token_count=19,
                ),
            )
        ]
    )
    provider = GeminiProvider(settings=_settings(), client=client, sleep=_no_sleep)

    response = await provider.generate_text(TextGenerationRequest(prompt="Summarize revenue."))

    assert response.content == "Retail revenue is stable."
    assert response.provider == "gemini"
    assert response.model == "gemini-test-text"
    assert response.input_tokens == 12
    assert response.output_tokens == 7
    assert response.latency_ms >= 0
    assert "temperature" not in _config_values(client.models.calls[0]["config"])


@pytest.mark.asyncio
async def test_generate_structured_success_validates_with_pydantic() -> None:
    client = FakeClient(responses=[FakeResponse(text='{"answer":"yes","score":91}')])
    provider = GeminiProvider(settings=_settings(), client=client, sleep=_no_sleep)

    response = await provider.generate_structured(
        StructuredGenerationRequest(
            prompt="Return a score.",
            response_model=DemoStructuredResponse,
        )
    )

    assert response.content == DemoStructuredResponse(answer="yes", score=91)
    assert response.model == "gemini-test-structured"
    config_values = _config_values(client.models.calls[0]["config"])
    assert config_values["response_mime_type"] == "application/json"
    assert "temperature" not in config_values


@pytest.mark.asyncio
async def test_generate_structured_invalid_json_raises_validation_error() -> None:
    client = FakeClient(responses=[FakeResponse(text='{"answer":"missing score"}')])
    provider = GeminiProvider(settings=_settings(), client=client, sleep=_no_sleep)

    with pytest.raises(StructuredOutputValidationError):
        await provider.generate_structured(
            StructuredGenerationRequest(
                prompt="Return a score.",
                response_model=DemoStructuredResponse,
            )
        )


@pytest.mark.asyncio
async def test_timeout_is_converted_to_internal_timeout_error() -> None:
    client = FakeClient(responses=[FakeResponse(text="too late")])
    provider = GeminiProvider(
        settings=_settings(timeout=0.001),
        client=client,
        sleep=_no_sleep,
    )

    with pytest.raises(AIProviderTimeoutError):
        await provider.generate_text(TextGenerationRequest(prompt="Slow call."))


@pytest.mark.asyncio
async def test_transient_error_retries_then_succeeds() -> None:
    client = FakeClient(
        responses=[FakeResponse(text="Recovered.")],
        errors=[TransientProviderError("temporary")],
    )
    provider = GeminiProvider(settings=_settings(max_retries=1), client=client, sleep=_no_sleep)

    response = await provider.generate_text(TextGenerationRequest(prompt="Try once."))

    assert response.content == "Recovered."
    assert len(client.models.calls) == 2


@pytest.mark.asyncio
async def test_transient_error_after_retries_raises_internal_error() -> None:
    client = FakeClient(errors=[TransientProviderError("temporary"), TransientProviderError("temporary")])
    provider = GeminiProvider(settings=_settings(max_retries=1), client=client, sleep=_no_sleep)

    with pytest.raises(AIProviderTransientError):
        await provider.generate_text(TextGenerationRequest(prompt="Try twice."))


@pytest.mark.asyncio
async def test_non_transient_error_is_not_retried() -> None:
    client = FakeClient(errors=[NonTransientProviderError("bad request")])
    provider = GeminiProvider(settings=_settings(max_retries=2), client=client, sleep=_no_sleep)

    with pytest.raises(AIProviderResponseError):
        await provider.generate_text(TextGenerationRequest(prompt="Bad request."))

    assert len(client.models.calls) == 1


@pytest.mark.asyncio
async def test_missing_api_key_raises_configuration_error_without_fake_client() -> None:
    provider = GeminiProvider(settings=_settings(api_key=None), sleep=_no_sleep)

    with pytest.raises(AIConfigurationError):
        await provider.generate_text(TextGenerationRequest(prompt="Hello."))


def _settings(
    api_key: str | None = "fake-key",
    timeout: float = 5.0,
    max_retries: int = 2,
) -> Settings:
    return Settings(
        gemini_api_key=api_key,
        gemini_text_model="gemini-test-text",
        gemini_structured_model="gemini-test-structured",
        ai_provider_timeout_seconds=timeout,
        ai_provider_max_retries=max_retries,
    )


def _config_values(config) -> dict:
    if isinstance(config, dict):
        return config
    if hasattr(config, "model_dump"):
        return config.model_dump(exclude_none=True)
    return {}


async def _no_sleep(seconds: float) -> None:
    await asyncio.sleep(0)
