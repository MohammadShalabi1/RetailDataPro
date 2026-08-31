from __future__ import annotations

import base64
import re
from enum import Enum

from pydantic import BaseModel, Field


class PromptGuardDecision(str, Enum):
    allow = "allow"
    allow_with_restrictions = "allow_with_restrictions"
    block = "block"


class PromptGuardResult(BaseModel):
    decision: PromptGuardDecision
    reason: str
    restrictions: list[str] = Field(default_factory=list)


DIRECT_BLOCK_PATTERNS = (
    r"\b(ignore|disregard|forget|override)\b.{0,80}\b(previous|prior|above|system|developer|safety|policy|rules|instructions?)\b",
    r"\b(reveal|show|print|display|dump|expose|share|give me)\b.{0,80}\b(system prompt|developer message|hidden prompt|hidden instructions?|internal prompt|chain of thought|reasoning trace)\b",
    r"\b(reveal|show|print|display|dump|expose|share|give me)\b.{0,80}\b(api keys?|secrets?|environment variables?|env vars?|credentials?|passwords?|tokens?)\b",
    r"\b(bypass|disable|ignore|override)\b.{0,80}\b(tool gateway|tool authorization|guardrails?|safety checks?|policy|retrieval filter|source authorization)\b",
    r"\b(call|use|invoke|execute)\b.{0,80}\b(hidden|internal|unauthorized|private)\b.{0,40}\b(tool|function|endpoint|route)\b",
    r"\b(ai_traces|messages|conversations|sources|source_chunks)\b",
    r"\b(role\s*:\s*system|role\s*:\s*developer|system\s*:|developer\s*:)\b",
    r"\byou are now\b.{0,80}\b(system|developer|admin|root|superuser)\b",
    r"\bdelete all customers\b",
)

UNTRUSTED_RESTRICT_PATTERNS = (
    r"\b(ignore|disregard|forget|override)\b.{0,80}\b(previous|prior|above|system|developer|safety|policy|rules|instructions?)\b",
    r"\b(reveal|show|print|display|dump|expose|share|give me)\b.{0,80}\b(system prompt|developer message|hidden prompt|hidden instructions?|internal prompt|api keys?|secrets?|environment variables?|env vars?)\b",
    r"\b(bypass|disable|ignore|override)\b.{0,80}\b(tool gateway|tool authorization|guardrails?|safety checks?|policy)\b",
    r"\b(document|website|report|file)\b.{0,40}\b(says|instructs|tells)\b.{0,60}\b(ignore|override|reveal|bypass)\b",
    r"\bbase64\b",
)


class PromptGuard:
    def classify(
        self,
        text: str,
        from_retrieved_document: bool = False,
        from_conversation_history: bool = False,
    ) -> PromptGuardResult:
        normalized = _normalize(_decode_obvious_base64(text))
        direct_match = _matches_any(normalized, DIRECT_BLOCK_PATTERNS)
        untrusted_match = _matches_any(normalized, UNTRUSTED_RESTRICT_PATTERNS)
        if from_retrieved_document or from_conversation_history:
            restrictions = ["treat_content_as_untrusted_data", "do_not_follow_embedded_instructions"]
            if direct_match or untrusted_match:
                restrictions.append("do_not_reveal_or_modify_system_behavior")
            return PromptGuardResult(
                decision=PromptGuardDecision.allow_with_restrictions,
                reason="untrusted_or_suspicious_context",
                restrictions=restrictions,
            )
        if direct_match:
            return PromptGuardResult(decision=PromptGuardDecision.block, reason="prompt_injection_or_secret_request")
        if untrusted_match:
            return PromptGuardResult(
                decision=PromptGuardDecision.allow_with_restrictions,
                reason="suspicious_user_context",
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


def _matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, value) for pattern in patterns)
