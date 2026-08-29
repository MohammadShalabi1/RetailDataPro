from __future__ import annotations

import math
import re
import time
from pydantic import BaseModel, ConfigDict, Field


class SemanticCacheKey(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    normalized_query: str
    route: str
    model_version: str
    prompt_version: str
    selected_sources: tuple[str, ...] = ()
    context_version: str


class SemanticCacheEntry(BaseModel):
    key: SemanticCacheKey
    embedding: list[float]
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: float = Field(default_factory=time.time)


class SemanticCacheStats(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    cache_hit: bool
    cache_miss: bool
    latency_saved_ms: int = 0
    model_calls_avoided: int = 0


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.92, min_confidence_to_cache: float = 0.7) -> None:
        self._entries: list[SemanticCacheEntry] = []
        self._similarity_threshold = similarity_threshold
        self._min_confidence_to_cache = min_confidence_to_cache

    def make_key(
        self,
        query: str,
        route: str,
        model_version: str,
        prompt_version: str,
        selected_sources: list[str] | None,
        context_version: str,
    ) -> SemanticCacheKey:
        return SemanticCacheKey(
            normalized_query=_normalize(query),
            route=route,
            model_version=model_version,
            prompt_version=prompt_version,
            selected_sources=tuple(sorted(selected_sources or [])),
            context_version=context_version,
        )

    def get(self, key: SemanticCacheKey, embedding: list[float]) -> tuple[SemanticCacheEntry | None, SemanticCacheStats]:
        compatible = [entry for entry in self._entries if _scope_compatible(entry.key, key)]
        best = max(compatible, key=lambda entry: _cosine_similarity(entry.embedding, embedding), default=None)
        if best is not None and _cosine_similarity(best.embedding, embedding) >= self._similarity_threshold:
            return best, SemanticCacheStats(cache_hit=True, cache_miss=False, latency_saved_ms=1, model_calls_avoided=1)
        return None, SemanticCacheStats(cache_hit=False, cache_miss=True)

    def put(
        self,
        key: SemanticCacheKey,
        embedding: list[float],
        answer: str,
        confidence: float,
        security_sensitive: bool = False,
        failed_answer: bool = False,
    ) -> bool:
        if security_sensitive or failed_answer or confidence < self._min_confidence_to_cache:
            return False
        self._entries.append(SemanticCacheEntry(key=key, embedding=embedding, answer=answer, confidence=confidence))
        return True


def _normalize(query: str) -> str:
    return re.sub(r"\s+", " ", query.lower()).strip()


def _scope_compatible(left: SemanticCacheKey, right: SemanticCacheKey) -> bool:
    return (
        left.route == right.route
        and left.model_version == right.model_version
        and left.prompt_version == right.prompt_version
        and left.selected_sources == right.selected_sources
        and left.context_version == right.context_version
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
