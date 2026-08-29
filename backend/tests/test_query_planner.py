from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ai.model_router import ModelTask
from app.ai.planner import ExecutionPlan, PlanStep, PlanStepKind, QueryPlanner
from app.ai.router import ModelRoute, RouteCategory, RouteReason
from app.tools.schemas import ToolName


def test_conversation_plan_has_no_tool_step() -> None:
    plan = QueryPlanner().create_plan("hello", _route(RouteCategory.conversation, RouteReason.general_conversation))

    assert plan.route == RouteCategory.conversation
    assert plan.model_task == ModelTask.analytics_answer
    assert plan.requires_synthesis is False
    assert plan.tool_steps == []


def test_retail_analytics_uses_analytics_summary_not_retail_sql() -> None:
    plan = QueryPlanner().create_plan("What was revenue?", _route(RouteCategory.retail_analytics, RouteReason.retail_metric))

    assert plan.steps[0].tool_name == ToolName.analytics_summary
    assert all(step.tool_name != ToolName.retail_sql for step in plan.steps)


def test_multi_source_plan_is_bounded_and_uses_approved_tools() -> None:
    plan = QueryPlanner().create_plan(
        "Compare category revenue with the supplier report.",
        _route(RouteCategory.multi_source, RouteReason.mixed_sources),
    )

    assert len(plan.steps) <= 3
    assert [step.tool_name for step in plan.tool_steps] == [ToolName.analytics_summary, ToolName.document_search]
    assert plan.model_task == ModelTask.multi_source_synthesis


def test_website_multi_source_plan_uses_website_search() -> None:
    plan = QueryPlanner().create_plan(
        "Compare revenue with this website.",
        _route(RouteCategory.multi_source, RouteReason.mixed_sources),
    )

    assert [step.tool_name for step in plan.tool_steps] == [ToolName.analytics_summary, ToolName.website_search]


def test_execution_plan_rejects_too_many_steps() -> None:
    with pytest.raises(ValidationError):
        ExecutionPlan(
            route=RouteCategory.multi_source,
            model_task=ModelTask.multi_source_synthesis,
            requires_synthesis=True,
            steps=[
                PlanStep(id=f"step_{index}", kind=PlanStepKind.answer, goal="Do bounded work.")
                for index in range(4)
            ],
        )


def _route(category: RouteCategory, reason: RouteReason) -> ModelRoute:
    return ModelRoute(category=category, confidence=0.82, reason_code=reason)
