"""Build and validate author-facing computation workflow declarations."""

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from .types import IterativeWorkflow, StepDefinition, SteppedWorkflow, StepResult


@dataclass
class _LocalStep:
    fn: Callable
    input_fn: Callable = None
    name: str = None


@dataclass
class _RemoteStep:
    fn: Callable


@dataclass
class _SiteOutputStep:
    fn: Callable
    name: str = None


PayloadT = TypeVar("PayloadT")
StateT = TypeVar("StateT")


@dataclass(frozen=True)
class _StatefulValue(Generic[PayloadT, StateT]):
    payload: PayloadT
    state: StateT


_POSITIONAL_PARAMETER_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)
_LEGACY_INJECTED_PARAMETER_NAMES = {
    "computation_parameters": "parameters",
    "params": "parameters",
    "local_state": "state",
    "remote_state": "state",
}
_FRAMEWORK_INJECTED_PARAMETER_NAMES = {
    "state",
    "parameters",
    "data_dir",
    "output_dir",
    "artifact_dir",
    "cache_dir",
    "logger",
}


def local_step(fn: Callable, input_fn: Callable = None, name: str = None) -> _LocalStep:
    """Declare a site-side computation step and optional input loader."""
    return _LocalStep(fn=fn, input_fn=input_fn, name=name)


def remote_step(fn: Callable) -> _RemoteStep:
    """Declare a server-side aggregation step."""
    return _RemoteStep(fn=fn)


def site_output_step(fn: Callable, name: str = None) -> _SiteOutputStep:
    """Declare the final site-side output step."""
    return _SiteOutputStep(fn=fn, name=name)


def with_state(payload: PayloadT, state: StateT) -> _StatefulValue[PayloadT, StateT]:
    """Return a transport payload together with persistent local or remote state."""
    return _StatefulValue(payload=payload, state=state)


def stepped_workflow(*steps) -> SteppedWorkflow:
    """Build a validated workflow from a known sequence of distinct steps."""
    if not steps:
        raise ValueError("stepped_workflow requires at least one step")

    step_definitions = []
    inferred_local_state_type = None
    index = 0

    while index < len(steps):
        step = steps[index]
        if isinstance(step, _LocalStep):
            remote = (
                steps[index + 1]
                if index + 1 < len(steps) and isinstance(steps[index + 1], _RemoteStep)
                else None
            )
            step_definition = _build_local_definition(step, remote)
            if remote is None:
                raise ValueError(
                    f"local_step '{step_definition.name}' must be immediately followed "
                    "by remote_step"
                )
            step_definitions.append(step_definition)
            inferred_local_state_type = (
                inferred_local_state_type or _infer_state_input_type(step.fn)
            )
            index += 2
            continue
        if isinstance(step, _SiteOutputStep):
            step_definition = _build_site_output_definition(step)
            if index != len(steps) - 1:
                raise ValueError(
                    f"site_output_step '{step_definition.name}' must be the final workflow step"
                )
            step_definitions.append(step_definition)
            inferred_local_state_type = (
                inferred_local_state_type or _infer_state_input_type(step.fn)
            )
            index += 1
            continue
        if isinstance(step, _RemoteStep):
            raise ValueError("remote_step must follow a local_step immediately")
        raise ValueError(f"Unsupported workflow step: {step!r}")

    _validate_unique_step_names(step_definitions)
    return SteppedWorkflow(
        steps=step_definitions,
        local_state_type=inferred_local_state_type,
    )


def iterative_workflow(
    local,
    remote,
    output,
    *,
    stop_when: Callable = None,
    max_iterations: int = 50,
) -> IterativeWorkflow:
    """Build a repeated local/remote workflow with optional convergence."""
    if not isinstance(local, _LocalStep):
        raise TypeError("iterative_workflow first step must be local_step(...)")
    if not isinstance(remote, _RemoteStep):
        raise TypeError("iterative_workflow second step must be remote_step(...)")
    if not isinstance(output, _SiteOutputStep):
        raise TypeError("iterative_workflow third step must be site_output_step(...)")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("iterative_workflow max_iterations must be a positive integer")

    iteration_step = _build_local_definition(
        local,
        remote,
        input_on_first_call_only=True,
    )
    output_step = _build_site_output_definition(output)
    _validate_unique_step_names([iteration_step, output_step])

    stop_fn = None
    if stop_when is not None:
        _validate_author_callable(stop_when, role="stop_when", expects_payload=True)
        stop_fn = _wrap_stop_fn(stop_when)

    return IterativeWorkflow(
        iteration_step=iteration_step,
        output_step=output_step,
        stop_fn=stop_fn,
        max_iterations=max_iterations,
        local_state_type=(
            _infer_state_input_type(local.fn) or _infer_state_input_type(output.fn)
        ),
    )


def get_task_names(workflow) -> List[str]:
    """Return the site task names generated by a workflow."""
    if isinstance(workflow, SteppedWorkflow):
        return [step.name for step in workflow.steps]
    if isinstance(workflow, IterativeWorkflow):
        return [workflow.iteration_step.name, workflow.output_step.name]
    raise TypeError(f"Unsupported workflow type: {type(workflow)!r}")


def _build_local_definition(
    step: _LocalStep,
    remote: _RemoteStep = None,
    *,
    input_on_first_call_only: bool = False,
) -> StepDefinition:
    _validate_author_callable(step.fn, role="local_step", expects_payload=True)
    if step.input_fn is not None:
        _validate_author_callable(step.input_fn, role="input_fn", expects_payload=False)
    if remote is not None:
        _validate_author_callable(remote.fn, role="remote_step", expects_payload=True)

    return StepDefinition(
        name=_resolve_step_name(step.name, step.fn, role="local_step"),
        local_fn=_wrap_local_fn(
            step.fn,
            step.input_fn,
            input_on_first_call_only=input_on_first_call_only,
        ),
        remote_fn=_wrap_remote_fn(remote.fn) if remote else None,
        local_input_type=(
            _infer_first_input_type(step.fn)
            if input_on_first_call_only or step.input_fn is None
            else None
        ),
        remote_site_result_type=_infer_site_result_type(remote.fn) if remote else None,
    )


def _build_site_output_definition(step: _SiteOutputStep) -> StepDefinition:
    _validate_author_callable(step.fn, role="site_output_step", expects_payload=True)
    return StepDefinition(
        name=_resolve_step_name(step.name, step.fn, role="site_output_step"),
        local_fn=_wrap_site_output_fn(step.fn),
        local_input_type=_infer_first_input_type(step.fn),
        is_site_output=True,
    )


def _wrap_local_fn(
    fn: Callable,
    input_fn: Callable = None,
    *,
    input_on_first_call_only: bool = False,
) -> Callable:
    def wrapped(incoming_payload, computation_parameters, local_state, runtime):
        input_state = None
        use_input_fn = input_fn is not None and (
            not input_on_first_call_only or runtime.current_round == 0
        )
        if use_input_fn:
            inputs = _invoke_author_callable(
                input_fn,
                computation_parameters,
                runtime,
                state=local_state,
                expects_payload=False,
            )
            if input_on_first_call_only and isinstance(inputs, _StatefulValue):
                input_state = inputs.state
                inputs = inputs.payload
            result = _invoke_author_callable(
                fn,
                computation_parameters,
                runtime,
                payload=inputs,
                state=local_state if local_state is not None else input_state,
                expects_payload=True,
            )
        else:
            result = _invoke_author_callable(
                fn,
                computation_parameters,
                runtime,
                payload=incoming_payload,
                state=local_state,
                expects_payload=True,
            )
        step_result = _to_step_result(result, state_field="local_state")
        if (
            input_state is not None
            and local_state is None
            and step_result.local_state is None
        ):
            step_result.local_state = input_state
        return step_result

    return wrapped


def _wrap_remote_fn(fn: Callable) -> Callable:
    def wrapped(site_results, computation_parameters, remote_state, runtime):
        site_arg = (
            list(site_results.values())
            if _expects_site_result_list(fn)
            else site_results
        )
        result = _invoke_author_callable(
            fn,
            computation_parameters,
            runtime,
            payload=site_arg,
            state=remote_state,
            expects_payload=True,
        )
        return _to_step_result(result, state_field="remote_state")

    return wrapped


def _wrap_site_output_fn(fn: Callable) -> Callable:
    def wrapped(incoming_payload, computation_parameters, local_state, runtime):
        result = _invoke_author_callable(
            fn,
            computation_parameters,
            runtime,
            payload=incoming_payload,
            state=local_state,
            expects_payload=True,
        )
        if isinstance(result, StepResult):
            return result
        if result is None:
            return StepResult()
        if not isinstance(result, Mapping):
            raise TypeError(
                f"site_output_step function '{_function_name(fn)}' must return a "
                "filename-to-value mapping or None"
            )
        return StepResult(outputs=dict(result))

    return wrapped


def _wrap_stop_fn(fn: Callable) -> Callable:
    def wrapped(payload, computation_parameters, remote_state, runtime):
        result = _invoke_author_callable(
            fn,
            computation_parameters,
            runtime,
            payload=payload,
            state=remote_state,
            expects_payload=True,
        )
        if (
            not isinstance(result, bool)
            and result.__class__.__module__.split(".")[0] == "numpy"
            and result.__class__.__name__ == "bool_"
        ):
            result = bool(result)
        if not isinstance(result, bool):
            raise TypeError(
                f"stop_when function '{_function_name(fn)}' must return bool, "
                f"not {type(result).__name__}"
            )
        return result

    return wrapped


def _invoke_author_callable(
    fn: Callable,
    computation_parameters: Dict[str, Any],
    runtime,
    payload: Any = None,
    state: Any = None,
    expects_payload: bool = True,
) -> Any:
    payload_parameter = _payload_parameter(fn) if expects_payload else None
    args = [payload] if payload_parameter is not None else []
    kwargs = {}
    framework_values = {
        "parameters": computation_parameters,
        "data_dir": runtime.data_dir,
        "output_dir": runtime.output_dir,
        "artifact_dir": getattr(runtime, "artifact_dir", ""),
        "cache_dir": getattr(runtime, "cache_dir", ""),
        "logger": runtime.logger,
    }

    for parameter in _injected_parameters(fn, expects_payload):
        if parameter.name == "state":
            if state is None:
                if parameter.default is inspect.Parameter.empty:
                    raise TypeError(
                        f"Cannot call {_function_name(fn)}: parameter 'state' was requested, "
                        "but no state is available"
                    )
                continue
            kwargs[parameter.name] = state
        elif parameter.name in framework_values:
            kwargs[parameter.name] = framework_values[parameter.name]
        elif parameter.name in computation_parameters:
            kwargs[parameter.name] = computation_parameters[parameter.name]
        elif parameter.default is inspect.Parameter.empty:
            raise TypeError(
                f"Cannot call {_function_name(fn)}: required parameter '{parameter.name}' "
                "is not a framework value and is missing from computation parameters"
            )

    return fn(*args, **kwargs)


def _validate_author_callable(fn: Callable, role: str, expects_payload: bool) -> None:
    if not callable(fn):
        raise TypeError(f"{role} fn must be callable; got {type(fn).__name__}")
    if inspect.iscoroutinefunction(fn):
        raise TypeError(f"{role} function '{_function_name(fn)}' must be synchronous")

    parameters = list(inspect.signature(fn).parameters.values())
    for parameter in parameters:
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"{role} function '{_function_name(fn)}' cannot use *args or **kwargs"
            )
        if parameter.name in _LEGACY_INJECTED_PARAMETER_NAMES:
            replacement = _LEGACY_INJECTED_PARAMETER_NAMES[parameter.name]
            raise TypeError(
                f"{role} function '{_function_name(fn)}' uses unsupported parameter "
                f"'{parameter.name}'; use '{replacement}'"
            )
        if parameter.name == "runtime":
            raise TypeError(
                f"{role} function '{_function_name(fn)}' cannot request framework runtime internals"
            )

    payload_parameter = _payload_parameter(fn) if expects_payload else None
    if (
        payload_parameter is not None
        and payload_parameter.name in _FRAMEWORK_INJECTED_PARAMETER_NAMES
    ):
        raise TypeError(
            f"{role} function '{_function_name(fn)}' uses reserved injected parameter "
            f"'{payload_parameter.name}' as its payload; declare a payload first or make "
            f"'{payload_parameter.name}' keyword-only"
        )

    for parameter in _injected_parameters(fn, expects_payload):
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"{role} function '{_function_name(fn)}' parameter '{parameter.name}' "
                "must be keyword-addressable"
            )


def _payload_parameter(fn: Callable):
    parameters = list(inspect.signature(fn).parameters.values())
    if parameters and parameters[0].kind in _POSITIONAL_PARAMETER_KINDS:
        return parameters[0]
    return None


def _injected_parameters(fn: Callable, expects_payload: bool):
    parameters = list(inspect.signature(fn).parameters.values())
    if expects_payload and _payload_parameter(fn) is not None:
        return parameters[1:]
    return parameters


def _to_step_result(value: Any, state_field: str) -> StepResult:
    if isinstance(value, StepResult):
        return value
    if isinstance(value, _StatefulValue):
        return StepResult(payload=value.payload, **{state_field: value.state})
    return StepResult(payload=value)


def _function_name(fn: Callable) -> str:
    return getattr(fn, "__name__", fn.__class__.__name__)


def _resolve_step_name(name: str, fn: Callable, role: str) -> str:
    if name is None:
        return _function_name(fn)
    if not isinstance(name, str) or not name.strip():
        raise TypeError(f"{role} name must be a non-empty string")
    return name


def _validate_unique_step_names(step_definitions: List[StepDefinition]) -> None:
    seen_names = set()
    for step_definition in step_definitions:
        if step_definition.name in seen_names:
            raise ValueError(
                f"Workflow step name '{step_definition.name}' is used more than once. "
                "Pass a unique name=... when reusing a local or output function."
            )
        seen_names.add(step_definition.name)


def _infer_first_input_type(fn: Callable):
    payload_parameter = _payload_parameter(fn)
    if payload_parameter is None:
        return None
    type_hints = get_type_hints(fn)
    return type_hints.get(payload_parameter.name)


def _infer_site_result_type(fn: Callable):
    first_type = _infer_first_input_type(fn)
    origin = get_origin(first_type)
    args = get_args(first_type)
    if origin in (dict, Dict) and len(args) >= 2:
        return args[1]
    if origin in (list, List) and args:
        return args[0]
    return None


def _infer_state_input_type(fn: Callable):
    type_hints = get_type_hints(fn)
    for parameter in _injected_parameters(fn, expects_payload=True):
        if parameter.name == "state":
            return type_hints.get(parameter.name)
    return None


def _expects_site_result_list(fn: Callable) -> bool:
    first_type = _infer_first_input_type(fn)
    return get_origin(first_type) in (list, List)
