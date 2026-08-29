from decimal import Decimal

import pytest

from app.ai.model_router import (
    ModelRouter,
    ModelSelectionReason,
    ModelTask,
    build_model_pricing_registry,
)
from app.core.config import Settings


def test_each_task_maps_to_expected_configured_model() -> None:
    router = ModelRouter(settings=_settings())

    assert router.select_model(ModelTask.simple_classification).model == "gemini-3.5-flash-lite"
    assert router.select_model(ModelTask.analytics_answer).model == "gemini-3.5-flash-lite"
    assert router.select_model(ModelTask.structured_generation).model == "gemini-3.6-flash"
    assert router.select_model(ModelTask.multi_source_synthesis).model == "gemini-3.6-flash"


def test_selection_reasons_are_task_specific() -> None:
    router = ModelRouter(settings=_settings())

    assert router.select_model(ModelTask.simple_classification).reason == ModelSelectionReason.fast_low_cost
    assert router.select_model(ModelTask.structured_generation).reason == ModelSelectionReason.structured_reliability
    assert router.select_model(ModelTask.analytics_answer).reason == ModelSelectionReason.balanced_answer
    assert router.select_model(ModelTask.multi_source_synthesis).reason == ModelSelectionReason.stronger_synthesis


def test_settings_changes_alter_selected_models_without_code_changes() -> None:
    router = ModelRouter(
        settings=_settings(
            gemini_text_model="gemini-custom-fast",
            gemini_structured_model="gemini-custom-structured",
        )
    )

    assert router.select_model(ModelTask.simple_classification).model == "gemini-custom-fast"
    assert router.select_model(ModelTask.structured_generation).model == "gemini-custom-structured"


def test_model_specific_pricing_registry_uses_settings_values() -> None:
    registry = build_model_pricing_registry(
        _settings(
            flash_lite_input=Decimal("0.11"),
            flash_lite_output=Decimal("0.22"),
            flash_input=Decimal("0.33"),
            flash_output=Decimal("0.44"),
        )
    )

    assert registry["gemini-3.5-flash-lite"].input_cost_per_1m_tokens == Decimal("0.11")
    assert registry["gemini-3.5-flash-lite"].output_cost_per_1m_tokens == Decimal("0.22")
    assert registry["gemini-3.6-flash"].input_cost_per_1m_tokens == Decimal("0.33")
    assert registry["gemini-3.6-flash"].output_cost_per_1m_tokens == Decimal("0.44")


def test_cost_estimate_uses_decimal_split_input_output_and_total() -> None:
    router = ModelRouter(settings=_settings())

    selection = router.select_model(
        ModelTask.simple_classification,
        estimated_input_tokens=500,
        estimated_output_tokens=100,
    )

    assert selection.estimated_input_cost_usd == Decimal("0.00015")
    assert selection.estimated_output_cost_usd == Decimal("0.00025")
    assert selection.estimated_total_cost_usd == Decimal("0.00040")
    assert selection.cost_estimate_available is True


def test_unknown_model_pricing_returns_cost_unavailable_not_fake_zero() -> None:
    router = ModelRouter(settings=_settings(gemini_text_model="gemini-3.7-flash"))

    selection = router.select_model(
        ModelTask.simple_classification,
        estimated_input_tokens=500,
        estimated_output_tokens=100,
    )

    assert selection.model == "gemini-3.7-flash"
    assert selection.cost_estimate_available is False
    assert selection.estimated_input_cost_usd is None
    assert selection.estimated_output_cost_usd is None
    assert selection.estimated_total_cost_usd is None


def test_zero_token_estimates_return_zero_cost_when_pricing_is_known() -> None:
    router = ModelRouter(settings=_settings())

    selection = router.select_model(ModelTask.structured_generation)

    assert selection.cost_estimate_available is True
    assert selection.estimated_input_cost_usd == Decimal("0.00")
    assert selection.estimated_output_cost_usd == Decimal("0.00")
    assert selection.estimated_total_cost_usd == Decimal("0.00")


def test_negative_token_estimates_are_rejected() -> None:
    router = ModelRouter(settings=_settings())

    with pytest.raises(ValueError, match="non-negative"):
        router.select_model(ModelTask.simple_classification, estimated_input_tokens=-1)


def test_trace_metadata_contains_model_cost_and_reason_fields() -> None:
    router = ModelRouter(settings=_settings())

    selection = router.select_model(
        ModelTask.simple_classification,
        estimated_input_tokens=500,
        estimated_output_tokens=100,
    )

    assert selection.to_trace_metadata() == {
        "task": "simple_classification",
        "model": "gemini-3.5-flash-lite",
        "reason": "fast_low_cost",
        "estimated_input_tokens": 500,
        "estimated_output_tokens": 100,
        "estimated_input_cost_usd": "0.00015",
        "estimated_output_cost_usd": "0.00025",
        "estimated_total_cost_usd": "0.00040",
        "cost_estimate_available": True,
    }


def _settings(
    gemini_text_model: str = "gemini-3.5-flash-lite",
    gemini_structured_model: str = "gemini-3.6-flash",
    flash_lite_input: Decimal = Decimal("0.30"),
    flash_lite_output: Decimal = Decimal("2.50"),
    flash_input: Decimal = Decimal("1.50"),
    flash_output: Decimal = Decimal("7.50"),
) -> Settings:
    return Settings(
        gemini_text_model=gemini_text_model,
        gemini_structured_model=gemini_structured_model,
        gemini_35_flash_lite_input_cost_per_1m=flash_lite_input,
        gemini_35_flash_lite_output_cost_per_1m=flash_lite_output,
        gemini_36_flash_input_cost_per_1m=flash_input,
        gemini_36_flash_output_cost_per_1m=flash_output,
    )
