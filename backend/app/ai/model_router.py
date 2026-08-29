from __future__ import annotations

from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings

TOKENS_PER_MILLION = Decimal("1000000")


class ModelTask(str, Enum):
    simple_classification = "simple_classification"
    structured_generation = "structured_generation"
    analytics_answer = "analytics_answer"
    multi_source_synthesis = "multi_source_synthesis"


class ModelSelectionReason(str, Enum):
    fast_low_cost = "fast_low_cost"
    structured_reliability = "structured_reliability"
    balanced_answer = "balanced_answer"
    stronger_synthesis = "stronger_synthesis"


class ModelPricing(BaseModel):
    input_cost_per_1m_tokens: Decimal = Field(ge=Decimal("0"))
    output_cost_per_1m_tokens: Decimal = Field(ge=Decimal("0"))


class ModelSelection(BaseModel):
    task: ModelTask
    model: str
    reason: ModelSelectionReason
    estimated_input_tokens: int = Field(ge=0)
    estimated_output_tokens: int = Field(ge=0)
    estimated_input_cost_usd: Decimal | None
    estimated_output_cost_usd: Decimal | None
    estimated_total_cost_usd: Decimal | None
    cost_estimate_available: bool

    def to_trace_metadata(self) -> dict:
        return {
            "task": self.task.value,
            "model": self.model,
            "reason": self.reason.value,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_input_cost_usd": _decimal_to_string(self.estimated_input_cost_usd),
            "estimated_output_cost_usd": _decimal_to_string(self.estimated_output_cost_usd),
            "estimated_total_cost_usd": _decimal_to_string(self.estimated_total_cost_usd),
            "cost_estimate_available": self.cost_estimate_available,
        }


class ModelRouter:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._pricing_by_model = build_model_pricing_registry(self._settings)

    def select_model(
        self,
        task: ModelTask,
        estimated_input_tokens: int = 0,
        estimated_output_tokens: int = 0,
    ) -> ModelSelection:
        if estimated_input_tokens < 0 or estimated_output_tokens < 0:
            raise ValueError("estimated token counts must be non-negative")

        model, reason = self._model_for_task(task)
        pricing = self._pricing_by_model.get(model)
        input_cost, output_cost, total_cost = _estimate_costs(
            pricing=pricing,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
        )
        return ModelSelection(
            task=task,
            model=model,
            reason=reason,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            estimated_input_cost_usd=input_cost,
            estimated_output_cost_usd=output_cost,
            estimated_total_cost_usd=total_cost,
            cost_estimate_available=pricing is not None,
        )

    def _model_for_task(self, task: ModelTask) -> tuple[str, ModelSelectionReason]:
        if task is ModelTask.simple_classification:
            return self._settings.gemini_text_model, ModelSelectionReason.fast_low_cost
        if task is ModelTask.structured_generation:
            return self._settings.gemini_structured_model, ModelSelectionReason.structured_reliability
        if task is ModelTask.analytics_answer:
            return self._settings.gemini_text_model, ModelSelectionReason.balanced_answer
        if task is ModelTask.multi_source_synthesis:
            return self._settings.gemini_structured_model, ModelSelectionReason.stronger_synthesis
        raise ValueError(f"Unsupported model task: {task}")


def build_model_pricing_registry(settings: Settings) -> dict[str, ModelPricing]:
    return {
        "gemini-3.5-flash-lite": ModelPricing(
            input_cost_per_1m_tokens=settings.gemini_35_flash_lite_input_cost_per_1m,
            output_cost_per_1m_tokens=settings.gemini_35_flash_lite_output_cost_per_1m,
        ),
        "gemini-3.6-flash": ModelPricing(
            input_cost_per_1m_tokens=settings.gemini_36_flash_input_cost_per_1m,
            output_cost_per_1m_tokens=settings.gemini_36_flash_output_cost_per_1m,
        ),
    }


def _estimate_costs(
    pricing: ModelPricing | None,
    estimated_input_tokens: int,
    estimated_output_tokens: int,
) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
    if pricing is None:
        return None, None, None

    input_cost = Decimal(estimated_input_tokens) / TOKENS_PER_MILLION * pricing.input_cost_per_1m_tokens
    output_cost = Decimal(estimated_output_tokens) / TOKENS_PER_MILLION * pricing.output_cost_per_1m_tokens
    return input_cost, output_cost, input_cost + output_cost


def _decimal_to_string(value: Decimal | None) -> str | None:
    if value is None:
        return None
    if value == Decimal("0"):
        return "0.00"
    if abs(value) < Decimal("1"):
        return format(value.quantize(Decimal("0.00001")), "f")
    normalized = value.normalize()
    if normalized == normalized.to_integral():
        return format(normalized.quantize(Decimal("0.00")), "f")
    return format(normalized, "f")
