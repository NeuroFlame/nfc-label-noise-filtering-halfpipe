"""Create an NVFlare job bundle from the computation application."""

import json
import os
import shutil
import sys
from typing import Any, Dict


def generate_job_meta(min_clients: int) -> Dict[str, Any]:
    """Build job metadata for deployment to all participants."""
    return {
        "resource_spec": {},
        "min_clients": min_clients,
        "deploy_map": {"app": ["@ALL"]},
    }


def create_job(app_path: str, job_path: str, min_clients: int) -> None:
    """Copy an application into a configured NVFlare job directory."""
    if not os.path.isdir(app_path):
        raise FileNotFoundError(f"Source app path '{app_path}' does not exist.")

    if type(min_clients) is not int or min_clients < 1:
        raise ValueError("Participant count must be a positive integer")

    # Prepare the destination path for the app
    job_app_path = os.path.join(job_path, "app")
    os.makedirs(job_app_path, exist_ok=True)

    # Copy the app directory
    shutil.copytree(
        app_path,
        job_app_path,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    update_client_tasks_from_spec(job_app_path)
    update_controller_participant_count(job_app_path, min_clients)

    # Generate and write job_meta to meta.json
    job_meta = generate_job_meta(min_clients)
    with open(os.path.join(job_path, "meta.json"), "w") as meta_file:
        json.dump(job_meta, meta_file, indent=2)


# Example usage:
# create_job('/path/to/app_folder', '/path/to/job_folder', min_clients=2)


def update_client_tasks_from_spec(app_path: str) -> None:
    """Populate client task names from the application's computation spec."""
    code_path = os.path.join(app_path, "code")
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, code_path)
    try:
        from computation.spec import SPEC
        from framework.workflow import get_task_names
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        if sys.path and sys.path[0] == code_path:
            sys.path.pop(0)

    config_path = os.path.join(app_path, "config", "config_fed_client.json")
    with open(config_path, "r+") as config_file:
        config = json.load(config_file)
        if "executors" in config and config["executors"]:
            config["executors"][0]["tasks"] = get_task_names(SPEC.workflow)
        config_file.seek(0)
        json.dump(config, config_file, indent=2)
        config_file.truncate()


def update_controller_participant_count(app_path: str, min_clients: int) -> None:
    """Require every provisioned participant to finish each federated step."""
    config_path = os.path.join(app_path, "config", "config_fed_server.json")
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    controllers = [
        workflow
        for workflow in config["workflows"]
        if workflow.get("path") == "runtime.controller.RuntimeController"
    ]
    if len(controllers) != 1:
        raise ValueError("Expected one computation runtime controller")
    # Job admission alone does not prevent NVFlare from advancing after a quorum.
    controllers[0].setdefault("args", {})["min_clients"] = min_clients
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
