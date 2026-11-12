"""Define internal workflow and runtime value objects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from .artifacts import (
    DEFAULT_ARTIFACT_RETRIES,
    DEFAULT_ARTIFACT_TIMEOUT,
    DEFAULT_MAX_ARTIFACT_BYTES,
    DEFAULT_MAX_ARTIFACT_TOTAL_BYTES,
)
from .serialization import DEFAULT_MAX_INLINE_ARRAY_BYTES

ITERATION_INDEX_KEY = "__neuroflame_iteration__"
ITERATION_STOP_KEY = "__neuroflame_iteration_stop__"


@dataclass
class RuntimeContext:
    """Runtime services and paths available to an author function."""

    fl_ctx: Any
    data_dir: str
    output_dir: str
    artifact_dir: str
    cache_dir: str
    current_round: int
    logger: Any = None
    max_inline_array_bytes: int = DEFAULT_MAX_INLINE_ARRAY_BYTES


@dataclass
class StepResult:
    """Normalized payload, state, and output returned by a wrapped step."""

    payload: Any = None
    local_state: Any = None
    remote_state: Any = None
    outputs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepDefinition:
    """Internal executable definition of one site-side workflow step."""

    name: str
    local_fn: Callable[
        [Any, Dict[str, Any], Dict[str, Any], RuntimeContext], StepResult
    ]
    remote_fn: Optional[
        Callable[
            [Dict[str, Any], Dict[str, Any], Dict[str, Any], RuntimeContext], StepResult
        ]
    ] = None
    local_input_type: Optional[type] = None
    remote_site_result_type: Optional[type] = None
    is_site_output: bool = False


@dataclass
class SteppedWorkflow:
    """Known sequence of distinct local, remote, and output steps."""

    steps: List[StepDefinition]
    local_state_type: Optional[type] = None


@dataclass
class IterativeWorkflow:
    """Repeated local and remote update pair followed by site output."""

    iteration_step: StepDefinition
    output_step: StepDefinition
    stop_fn: Optional[Callable[[Any, Dict[str, Any], Any, RuntimeContext], bool]] = None
    max_iterations: int = 50
    local_state_type: Optional[type] = None


WorkflowDefinition = Union[SteppedWorkflow, IterativeWorkflow]


@dataclass(init=False)
class ComputationSpec:
    """Top-level author configuration for a computation workflow."""

    workflow: WorkflowDefinition
    codecs: Dict[type, Any] = field(default_factory=dict)
    max_inline_array_bytes: int = DEFAULT_MAX_INLINE_ARRAY_BYTES
    max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES
    max_artifact_total_bytes: int = DEFAULT_MAX_ARTIFACT_TOTAL_BYTES
    artifact_timeout: float = DEFAULT_ARTIFACT_TIMEOUT
    artifact_retries: int = DEFAULT_ARTIFACT_RETRIES

    def __init__(
        self,
        workflow: WorkflowDefinition,
        *,
        codecs: Optional[Mapping] = None,
        max_inline_array_bytes: int = DEFAULT_MAX_INLINE_ARRAY_BYTES,
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        max_artifact_total_bytes: int = DEFAULT_MAX_ARTIFACT_TOTAL_BYTES,
        artifact_timeout: float = DEFAULT_ARTIFACT_TIMEOUT,
        artifact_retries: int = DEFAULT_ARTIFACT_RETRIES,
    ):
        """Validate and store workflow and serialization options."""
        if not isinstance(workflow, (SteppedWorkflow, IterativeWorkflow)):
            raise TypeError(
                "ComputationSpec workflow must be created by stepped_workflow(...) "
                "or iterative_workflow(...)"
            )
        if codecs is not None and not isinstance(codecs, Mapping):
            raise TypeError("ComputationSpec codecs must be a type-to-codec mapping")

        resolved_codecs = dict(codecs or {})
        for value_type, codec in resolved_codecs.items():
            if not isinstance(value_type, type):
                raise TypeError("ComputationSpec codec keys must be Python types")
            if not callable(getattr(codec, "encode", None)) or not callable(
                getattr(codec, "decode", None)
            ):
                raise TypeError(
                    f"Codec for {value_type.__name__} must define callable encode() and decode()"
                )

        if isinstance(max_inline_array_bytes, bool) or not isinstance(
            max_inline_array_bytes, int
        ):
            raise TypeError("max_inline_array_bytes must be an integer byte count")
        if max_inline_array_bytes < 0:
            raise ValueError("max_inline_array_bytes cannot be negative")

        for field_name, field_value in (
            ("max_artifact_bytes", max_artifact_bytes),
            ("max_artifact_total_bytes", max_artifact_total_bytes),
            ("artifact_retries", artifact_retries),
        ):
            if isinstance(field_value, bool) or not isinstance(field_value, int):
                raise TypeError(f"{field_name} must be an integer")
            if field_value < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if max_artifact_total_bytes < max_artifact_bytes:
            raise ValueError(
                "max_artifact_total_bytes cannot be smaller than max_artifact_bytes"
            )
        if isinstance(artifact_timeout, bool) or not isinstance(
            artifact_timeout, (int, float)
        ):
            raise TypeError("artifact_timeout must be a number")
        if artifact_timeout <= 0:
            raise ValueError("artifact_timeout must be positive")

        self.workflow = workflow
        self.codecs = resolved_codecs
        self.max_inline_array_bytes = max_inline_array_bytes
        self.max_artifact_bytes = max_artifact_bytes
        self.max_artifact_total_bytes = max_artifact_total_bytes
        self.artifact_timeout = float(artifact_timeout)
        self.artifact_retries = artifact_retries
