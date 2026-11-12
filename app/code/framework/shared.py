"""Build shared runtime values used by framework components."""

import json
import os
from typing import Any, Dict

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext

from .logger import create_computation_logger
from .paths import (
    get_data_directory_path,
    get_output_directory_path,
    get_parameters_file_path,
)
from .types import ComputationSpec, RuntimeContext


def load_computation_parameters(fl_ctx: FLContext) -> Dict[str, Any]:
    """Load computation parameters from the resolved JSON file."""
    with open(
        get_parameters_file_path(fl_ctx), "r", encoding="utf-8"
    ) as parameters_file:
        return json.load(parameters_file)


def resolve_data_directory(fl_ctx: FLContext) -> str:
    """Resolve the site-local data directory."""
    return get_data_directory_path(fl_ctx)


def resolve_output_directory(fl_ctx: FLContext) -> str:
    """Resolve the current participant's output directory."""
    return get_output_directory_path(fl_ctx)


def resolve_site_name(site_id: str, parameters: Dict[str, Any]) -> str:
    """Map an internal site identifier to its configured display name."""
    site_id_name_map = parameters.get("site_id_name_map", {})
    return site_id_name_map.get(site_id, site_id)


def build_runtime_context(
    spec: ComputationSpec,
    fl_ctx: FLContext,
    current_round: int,
    parameters: Dict[str, Any],
    logger_suffix: str,
) -> RuntimeContext:
    """Build the paths, logger, and metadata injected into author functions."""
    output_dir = resolve_output_directory(fl_ctx)
    data_dir = ""

    try:
        data_dir = resolve_data_directory(fl_ctx)
    except Exception:
        data_dir = ""

    client_name = (
        fl_ctx.get_prop(FLContextKey.CLIENT_NAME, default="aggregator") or "aggregator"
    )
    logger = create_computation_logger(
        output_dir,
        f"{client_name}{logger_suffix}",
        parameters,
    )
    artifact_dir = os.path.join(
        output_dir,
        ".artifacts",
        "author",
        f"round-{current_round}",
    )
    cache_dir = os.path.join(output_dir, ".artifacts", "cache")
    os.makedirs(artifact_dir, mode=0o700, exist_ok=True)
    os.chmod(artifact_dir, 0o700)
    os.makedirs(cache_dir, mode=0o700, exist_ok=True)
    os.chmod(cache_dir, 0o700)

    return RuntimeContext(
        fl_ctx=fl_ctx,
        data_dir=data_dir,
        output_dir=output_dir,
        artifact_dir=artifact_dir,
        cache_dir=cache_dir,
        current_round=current_round,
        logger=logger,
        max_inline_array_bytes=spec.max_inline_array_bytes,
    )
