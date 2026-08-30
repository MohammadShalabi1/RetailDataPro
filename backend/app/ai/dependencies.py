from __future__ import annotations

from app.ai.gemini_provider import GeminiProvider
from app.ai.provider import AIProvider


def get_ai_provider() -> AIProvider:
    return GeminiProvider()
