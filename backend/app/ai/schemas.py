from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)
ContentT = TypeVar("ContentT")


@dataclass(frozen=True)
class TextGenerationRequest:
    prompt: str
    model: str | None = None


@dataclass(frozen=True)
class StructuredGenerationRequest(Generic[StructuredModelT]):
    prompt: str
    response_model: type[StructuredModelT]
    model: str | None = None


@dataclass(frozen=True)
class AIResponse(Generic[ContentT]):
    content: ContentT
    model: str
    provider: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    raw_usage_metadata: dict[str, Any] = field(default_factory=dict)
