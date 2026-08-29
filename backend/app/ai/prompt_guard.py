from __future__ import annotations

import base64
import re
from enum import Enum

from pydantic import BaseModel


class PromptGuardDecision(str, Enum):
    allow = "allow"
    allow_with_restrictions = "allow_with_restrictions"
    block = "block"


class PromptGuardResult(BaseModel):
    decision: PromptGuardDecision
    reason: str
    restrictions: list[str] = []


BLOCK_PATTERNS = (
    "ignore previous instructions",
    "ignore your rules",
    "reveal your system prompt",
    "show me your api key",
    "print environment variables",
    "bypass the tool gateway",
    "delete all customers",
)

RESTRICT_PATTERNS = (
    "document says ignore",
    "website says ignore",
    "untrusted data",
    "base64",
)


class PromptGuard:
    def classify(self, text: str, from_retrieved_document: bool = False) -> PromptGuardResult:
        normalized = _normalize(_decode_obvious_base64(text))
        if any(pattern in normalized for pattern in BLOCK_PATTERNS):
            return PromptGuardResult(decision=PromptGuardDecision.block, reason="prompt_injection_or_secret_request")
        if from_retrieved_document or any(pattern in normalized for pattern in RESTRICT_PATTERNS):
            return PromptGuardResult(
                decision=PromptGuardDecision.allow_with_restrictions,
                reason="untrusted_or_suspicious_context",
                restrictions=["treat_content_as_untrusted_data", "do_not_follow_embedded_instructions"],
            )
        return PromptGuardResult(decision=PromptGuardDecision.allow, reason="no_policy_issue_detected")


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _decode_obvious_base64(value: str) -> str:
    decoded_parts = [value]
    for token in re.findall(r"[A-Za-z0-9+/=]{16,}", value):
        try:
            decoded = base64.b64decode(token, validate=True).decode("utf-8", errors="ignore")
        except Exception:
            continue
        decoded_parts.append(decoded)
    return " ".join(decoded_parts)
