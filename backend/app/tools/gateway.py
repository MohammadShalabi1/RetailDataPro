from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import ValidationError

from app.tools.registry import ToolRegistry, build_default_tool_registry
from app.tools.schemas import ToolExecutionContext, ToolExecutionRequest, ToolExecutionResult, ToolName, ToolStatus


async def authorize_and_execute_tool(
    request: ToolExecutionRequest,
    context: ToolExecutionContext,
    registry: ToolRegistry | None = None,
) -> ToolExecutionResult:
    started_at = time.perf_counter()
    active_registry = registry or build_default_tool_registry()
    try:
        tool_name = ToolName(request.tool_name)
    except ValueError:
        result = _result(request.tool_name, ToolStatus.rejected, started_at, authorized=False, error_code="unknown_tool")
        _record_trace(context.trace, result)
        return result

    definition = active_registry.get(tool_name)

    if definition is None:
        result = _result(request.tool_name, ToolStatus.rejected, started_at, authorized=False, error_code="unknown_tool")
        _record_trace(context.trace, result)
        return result

    authorized = context.user_role in definition.allowed_roles
    if not authorized:
        result = _result(request.tool_name, ToolStatus.rejected, started_at, authorized=False, error_code="unauthorized")
        _record_trace(context.trace, result)
        return result

    if not definition.is_available or definition.executor is None:
        result = _result(request.tool_name, ToolStatus.unavailable, started_at, authorized=True, error_code="tool_unavailable")
        _record_trace(context.trace, result)
        return result

    try:
        parsed_input = definition.input_model.model_validate(request.input)
    except ValidationError:
        result = _result(request.tool_name, ToolStatus.validation_error, started_at, authorized=True, error_code="invalid_input")
        _record_trace(context.trace, result)
        return result

    try:
        output = await asyncio.wait_for(
            definition.executor(parsed_input, context),
            timeout=definition.timeout_seconds,
        )
    except asyncio.TimeoutError:
        result = _result(request.tool_name, ToolStatus.timeout, started_at, authorized=True, error_code="tool_timeout")
        _record_trace(context.trace, result)
        return result
    except Exception:
        result = _result(request.tool_name, ToolStatus.execution_error, started_at, authorized=True, error_code="tool_error")
        _record_trace(context.trace, result)
        return result

    result = _result(
        request.tool_name,
        ToolStatus.success,
        started_at,
        authorized=True,
        output=output.model_dump(mode="json") if hasattr(output, "model_dump") else dict(output),
    )
    _record_trace(context.trace, result)
    return result


def _result(
    tool_name: str,
    status: ToolStatus,
    started_at: float,
    authorized: bool,
    output: dict[str, Any] | None = None,
    error_code: str | None = None,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_name=tool_name,
        status=status,
        output=output or {},
        latency_ms=max(0, round((time.perf_counter() - started_at) * 1000)),
        authorized=authorized,
        error_code=error_code,
    )


def _record_trace(trace: Any | None, result: ToolExecutionResult) -> None:
    if trace is None:
        return
    event = {"stage": "tool_gateway", **result.to_trace_metadata()}
    if hasattr(trace, "append"):
        trace.append(event)
        return
    if hasattr(trace, "record"):
        trace.record(event)
