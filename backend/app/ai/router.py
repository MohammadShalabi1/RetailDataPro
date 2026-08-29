from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.ai.errors import AIProviderError
from app.ai.provider import AIProvider
from app.ai.schemas import AIResponse, StructuredGenerationRequest


class RouteCategory(str, Enum):
    conversation = "conversation"
    retail_analytics = "retail_analytics"
    document_search = "document_search"
    website_search = "website_search"
    multi_source = "multi_source"


class RouteReason(str, Enum):
    general_conversation = "general_conversation"
    retail_metric = "retail_metric"
    retail_entity = "retail_entity"
    document_reference = "document_reference"
    web_reference = "web_reference"
    mixed_sources = "mixed_sources"
    fallback_keyword = "fallback_keyword"


class ModelRoute(BaseModel):
    category: RouteCategory
    confidence: float = Field(ge=0.0, le=1.0)
    reason_code: RouteReason


class TypedRouter:
    def __init__(self, ai_provider: AIProvider) -> None:
        self._ai_provider = ai_provider

    async def select_route(self, question: str, trace: Any | None = None) -> ModelRoute:
        repair_used = False
        fallback_used = False
        provider_metadata: dict[str, Any] = {}

        try:
            response = await self._ai_provider.generate_structured(
                StructuredGenerationRequest(
                    prompt=build_route_prompt(question),
                    response_model=ModelRoute,
                )
            )
            route = _validate_route(response.content)
            provider_metadata = _response_metadata(response)
        except (AIProviderError, ValidationError, TypeError, ValueError):
            repair_used = True
            try:
                response = await self._ai_provider.generate_structured(
                    StructuredGenerationRequest(
                        prompt=build_route_repair_prompt(question),
                        response_model=ModelRoute,
                    )
                )
                route = _validate_route(response.content)
                provider_metadata = _response_metadata(response)
            except (AIProviderError, ValidationError, TypeError, ValueError):
                fallback_used = True
                route = deterministic_route(question)

        _record_trace(
            trace,
            {
                "stage": "route",
                "category": route.category.value,
                "reason_code": route.reason_code.value,
                "confidence": route.confidence,
                "repair_used": repair_used,
                "fallback_used": fallback_used,
                **provider_metadata,
            },
        )
        return route


def build_route_prompt(question: str) -> str:
    return (
        "Classify the user question for the RetailData-Pro backend.\n"
        "Return only valid JSON matching this schema:\n"
        "{category: conversation|retail_analytics|document_search|website_search|multi_source, "
        "confidence: number from 0.0 to 1.0, "
        "reason_code: general_conversation|retail_metric|retail_entity|document_reference|web_reference|mixed_sources}.\n"
        "Use retail_analytics for deterministic retail metrics, sales, inventory, customer, product, category, "
        "or supplier analytics. Do not use a SQL route because text-to-SQL is not implemented yet.\n"
        "Use multi_source when the question needs both retail analytics and document or web evidence.\n"
        f"Question: {question}"
    )


def build_route_repair_prompt(question: str) -> str:
    return (
        "The prior routing response was invalid. Repair it by returning only valid JSON.\n"
        "Allowed categories: conversation, retail_analytics, document_search, website_search, multi_source.\n"
        "Allowed reason_code values: general_conversation, retail_metric, retail_entity, document_reference, "
        "web_reference, mixed_sources.\n"
        "Confidence must be between 0.0 and 1.0. Do not include explanations outside JSON.\n"
        f"Question: {question}"
    )


def deterministic_route(question: str) -> ModelRoute:
    normalized = question.lower()
    has_retail = _contains_any(normalized, RETAIL_ANALYTICS_KEYWORDS)
    has_document = _contains_any(normalized, DOCUMENT_KEYWORDS)
    has_web = _contains_any(normalized, WEBSITE_KEYWORDS)

    if has_retail and (has_document or has_web):
        return ModelRoute(category=RouteCategory.multi_source, confidence=0.62, reason_code=RouteReason.fallback_keyword)
    if has_web:
        return ModelRoute(
            category=RouteCategory.website_search,
            confidence=0.58,
            reason_code=RouteReason.fallback_keyword,
        )
    if has_document:
        return ModelRoute(
            category=RouteCategory.document_search,
            confidence=0.58,
            reason_code=RouteReason.fallback_keyword,
        )
    if has_retail:
        return ModelRoute(
            category=RouteCategory.retail_analytics,
            confidence=0.6,
            reason_code=RouteReason.fallback_keyword,
        )
    return ModelRoute(
        category=RouteCategory.conversation,
        confidence=0.5,
        reason_code=RouteReason.fallback_keyword,
    )


RETAIL_ANALYTICS_KEYWORDS = {
    "revenue",
    "sales",
    "trend",
    "customer",
    "customers",
    "product",
    "products",
    "category",
    "categories",
    "inventory",
    "stock",
    "supplier",
    "suppliers",
    "order",
    "orders",
    "margin",
    "top",
    "decline",
    "declined",
    "growth",
    "performance",
}

DOCUMENT_KEYWORDS = {
    "document",
    "documents",
    "pdf",
    "report",
    "reports",
    "source",
    "sources",
    "brief",
    "supplier report",
    "citation",
    "file",
}

WEBSITE_KEYWORDS = {
    "website",
    "web",
    "url",
    "http://",
    "https://",
    "page",
    "site",
    "online",
}


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _validate_route(value: Any) -> ModelRoute:
    return ModelRoute.model_validate(value)


def _response_metadata(response: AIResponse[Any]) -> dict[str, Any]:
    return {
        "provider": response.provider,
        "model": response.model,
        "latency_ms": response.latency_ms,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
    }


def _record_trace(trace: Any | None, event: dict[str, Any]) -> None:
    if trace is None:
        return
    if hasattr(trace, "append"):
        trace.append(event)
        return
    if hasattr(trace, "record"):
        trace.record(event)
