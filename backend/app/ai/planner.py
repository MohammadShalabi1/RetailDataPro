from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ai.model_router import ModelTask
from app.ai.router import ModelRoute, RouteCategory
from app.tools.schemas import ToolName

MAX_PLAN_STEPS = 3


class PlanStepKind(str, Enum):
    answer = "answer"
    tool = "tool"


class PlanStep(BaseModel):
    id: str
    kind: PlanStepKind
    goal: str = Field(min_length=1, max_length=300)
    tool_name: ToolName | None = None
    required: bool = True
    depends_on: list[str] = Field(default_factory=list, max_length=MAX_PLAN_STEPS)

    @field_validator("depends_on")
    @classmethod
    def unique_dependencies(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("depends_on must not contain duplicate step ids")
        return value


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    route: RouteCategory
    model_task: ModelTask
    steps: list[PlanStep] = Field(min_length=1, max_length=MAX_PLAN_STEPS)
    requires_synthesis: bool

    @field_validator("steps")
    @classmethod
    def validate_steps(cls, value: list[PlanStep]) -> list[PlanStep]:
        step_ids = [step.id for step in value]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("plan step ids must be unique")
        known_ids = set(step_ids)
        for step in value:
            if step.kind is PlanStepKind.tool and step.tool_name is None:
                raise ValueError("tool steps require a tool_name")
            missing_dependencies = set(step.depends_on) - known_ids
            if missing_dependencies:
                raise ValueError("plan step dependencies must reference existing steps")
        return value

    @property
    def tool_steps(self) -> list[PlanStep]:
        return [step for step in self.steps if step.kind is PlanStepKind.tool]


class QueryPlanner:
    def create_plan(self, question: str, route: ModelRoute) -> ExecutionPlan:
        if route.category is RouteCategory.conversation:
            return ExecutionPlan(
                route=route.category,
                model_task=ModelTask.analytics_answer,
                requires_synthesis=False,
                steps=[
                    PlanStep(
                        id="answer_conversation",
                        kind=PlanStepKind.answer,
                        goal="Answer the user conversationally without claiming tool access.",
                        required=True,
                    )
                ],
            )

        if route.category is RouteCategory.multi_source:
            return self._multi_source_plan(question, route.category)

        if route.category is RouteCategory.retail_analytics:
            return ExecutionPlan(
                route=route.category,
                model_task=ModelTask.analytics_answer,
                requires_synthesis=True,
                steps=[
                    PlanStep(
                        id="retail_analytics",
                        kind=PlanStepKind.tool,
                        tool_name=ToolName.analytics_summary,
                        goal="Get deterministic retail analytics for the user question.",
                        required=True,
                    )
                ],
            )

        if route.category is RouteCategory.document_search:
            return ExecutionPlan(
                route=route.category,
                model_task=ModelTask.structured_generation,
                requires_synthesis=True,
                steps=[
                    PlanStep(
                        id="document_evidence",
                        kind=PlanStepKind.tool,
                        tool_name=ToolName.document_search,
                        goal="Retrieve document evidence relevant to the user question.",
                        required=True,
                    )
                ],
            )

        return ExecutionPlan(
            route=route.category,
            model_task=ModelTask.structured_generation,
            requires_synthesis=True,
            steps=[
                PlanStep(
                    id="website_evidence",
                    kind=PlanStepKind.tool,
                    tool_name=ToolName.website_search,
                    goal="Retrieve website evidence relevant to the user question.",
                    required=True,
                )
            ],
        )

    def _multi_source_plan(self, question: str, route: RouteCategory) -> ExecutionPlan:
        normalized = question.lower()
        steps = [
            PlanStep(
                id="retail_analytics",
                kind=PlanStepKind.tool,
                tool_name=ToolName.analytics_summary,
                goal="Get deterministic retail analytics for the user question.",
                required=True,
            )
        ]

        if any(term in normalized for term in ("website", "url", "web page", "current site")):
            steps.append(
                PlanStep(
                    id="website_evidence",
                    kind=PlanStepKind.tool,
                    tool_name=ToolName.website_search,
                    goal="Retrieve website evidence relevant to the user question.",
                    required=True,
                )
            )
        else:
            steps.append(
                PlanStep(
                    id="document_evidence",
                    kind=PlanStepKind.tool,
                    tool_name=ToolName.document_search,
                    goal="Retrieve document evidence relevant to the user question.",
                    required=True,
                )
            )

        return ExecutionPlan(
            route=route,
            model_task=ModelTask.multi_source_synthesis,
            requires_synthesis=True,
            steps=steps[:MAX_PLAN_STEPS],
        )
