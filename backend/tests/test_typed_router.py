from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ai.errors import AIProviderResponseError
from app.ai.provider import AIProvider
from app.ai.router import ModelRoute, RouteCategory, RouteReason, TypedRouter, deterministic_route
from app.ai.schemas import AIResponse, StructuredGenerationRequest, TextGenerationRequest


class FakeProvider(AIProvider):
    def __init__(self, outcomes: list) -> None:
        self.outcomes = outcomes
        self.structured_requests: list[StructuredGenerationRequest] = []

    async def generate_text(self, request: TextGenerationRequest) -> AIResponse[str]:
        raise NotImplementedError

    async def generate_structured(self, request: StructuredGenerationRequest) -> AIResponse:
        self.structured_requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, dict):
            content = request.response_model.model_validate(outcome)
        else:
            content = outcome
        return AIResponse(
            content=content,
            model="configured-structured-model",
            provider="fake",
            latency_ms=12,
            input_tokens=10,
            output_tokens=5,
        )


@pytest.mark.asyncio
async def test_valid_structured_route_returns_directly() -> None:
    provider = FakeProvider(
        [
            {
                "category": "retail_analytics",
                "confidence": 0.93,
                "reason_code": "retail_metric",
            }
        ]
    )
    router = TypedRouter(provider)

    route = await router.select_route("What was revenue last month?")

    assert route == ModelRoute(
        category=RouteCategory.retail_analytics,
        confidence=0.93,
        reason_code=RouteReason.retail_metric,
    )
    assert len(provider.structured_requests) == 1
    assert provider.structured_requests[0].model is None


@pytest.mark.asyncio
async def test_invalid_confidence_triggers_one_repair_attempt() -> None:
    provider = FakeProvider(
        [
            {"category": "retail_analytics", "confidence": 8.7, "reason_code": "retail_metric"},
            {"category": "retail_analytics", "confidence": 0.87, "reason_code": "retail_metric"},
        ]
    )
    router = TypedRouter(provider)

    route = await router.select_route("Show sales trends.")

    assert route.confidence == 0.87
    assert len(provider.structured_requests) == 2


@pytest.mark.asyncio
async def test_invalid_reason_code_triggers_repair_attempt() -> None:
    provider = FakeProvider(
        [
            {"category": "document_search", "confidence": 0.8, "reason_code": "because_i_think_so"},
            {"category": "document_search", "confidence": 0.78, "reason_code": "document_reference"},
        ]
    )
    router = TypedRouter(provider)

    route = await router.select_route("What does the supplier report say?")

    assert route.category == RouteCategory.document_search
    assert route.reason_code == RouteReason.document_reference
    assert len(provider.structured_requests) == 2


@pytest.mark.asyncio
async def test_invalid_repair_response_uses_deterministic_fallback() -> None:
    provider = FakeProvider(
        [
            {"category": "retail_analytics", "confidence": 2.0, "reason_code": "retail_metric"},
            {"category": "retail_analytics", "confidence": 3.0, "reason_code": "retail_metric"},
        ]
    )
    router = TypedRouter(provider)

    route = await router.select_route("Which categories declined according to the supplier report?")

    assert route.category == RouteCategory.multi_source
    assert route.reason_code == RouteReason.fallback_keyword
    assert len(provider.structured_requests) == 2


@pytest.mark.asyncio
async def test_provider_errors_use_deterministic_fallback() -> None:
    provider = FakeProvider([AIProviderResponseError("bad"), AIProviderResponseError("bad again")])
    router = TypedRouter(provider)

    route = await router.select_route("Open this URL and compare it to sales.")

    assert route.category == RouteCategory.multi_source
    assert route.reason_code == RouteReason.fallback_keyword


@pytest.mark.asyncio
async def test_trace_collector_receives_route_metadata() -> None:
    provider = FakeProvider(
        [
            {
                "category": "website_search",
                "confidence": 0.76,
                "reason_code": "web_reference",
            }
        ]
    )
    router = TypedRouter(provider)
    trace: list[dict] = []

    await router.select_route("Check https://example.com", trace=trace)

    assert trace == [
        {
            "stage": "route",
            "category": "website_search",
            "reason_code": "web_reference",
            "confidence": 0.76,
            "repair_used": False,
            "fallback_used": False,
            "provider": "fake",
            "model": "configured-structured-model",
            "latency_ms": 12,
            "input_tokens": 10,
            "output_tokens": 5,
        }
    ]


def test_deterministic_route_detects_mixed_sources() -> None:
    route = deterministic_route("Which categories declined, and does the supplier report explain why?")

    assert route.category == RouteCategory.multi_source
    assert route.reason_code == RouteReason.fallback_keyword


@pytest.mark.asyncio
async def test_model_multi_source_route_is_corrected_for_pure_retail_question() -> None:
    provider = FakeProvider(
        [
            {
                "category": "multi_source",
                "confidence": 0.94,
                "reason_code": "mixed_sources",
            }
        ]
    )
    router = TypedRouter(provider)

    route = await router.select_route("What was revenue last month?")

    assert route.category == RouteCategory.retail_analytics
    assert route.reason_code == RouteReason.retail_metric
    assert route.confidence == 0.78


def test_model_route_validates_confidence_instead_of_clamping() -> None:
    with pytest.raises(ValidationError):
        ModelRoute(category="retail_analytics", confidence=8.7, reason_code="retail_metric")


def test_route_reason_is_typed() -> None:
    with pytest.raises(ValidationError):
        ModelRoute(category="retail_analytics", confidence=0.8, reason_code="database probably lol")
