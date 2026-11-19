"""Build and publish a versioned NeuroFLAME computation image."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

if __package__:
    from .neuroflame_manifest import REQUIRED_IMAGE_KEYS, SEMVER, load_manifest
else:
    from neuroflame_manifest import REQUIRED_IMAGE_KEYS, SEMVER, load_manifest

REVISION = re.compile(r"[0-9a-f]{7,64}")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
MANIFEST_DIGEST = re.compile(rf"Digest:\s+({DIGEST.pattern})")
DEFAULT_PLATFORMS = "linux/amd64,linux/arm64"


def _run(command: list[str], repository: Path, *, capture_output: bool = False) -> str:
    """Run a command in the computation repository."""
    completed = subprocess.run(
        command,
        cwd=repository,
        check=True,
        text=True,
        capture_output=capture_output,
    )
    return completed.stdout


def _read_nvflare_version(repository: Path) -> str:
    """Read the exact NVFlare pin from requirements.txt."""
    matches = re.findall(
        r"(?im)^nvflare==([^\s#]+)\s*$",
        (repository / "requirements.txt").read_text(encoding="utf-8"),
    )
    if len(matches) != 1 or not SEMVER.fullmatch(matches[0]):
        raise ValueError("requirements.txt must contain one exact nvflare==X.Y.Z pin")
    return matches[0]


def _get_revision(repository: Path, requested_revision: str | None) -> str:
    """Return HEAD and reject an override that names another checkout."""
    head = _run(["git", "rev-parse", "HEAD"], repository, capture_output=True).strip()
    if not REVISION.fullmatch(head):
        raise ValueError(f"Invalid Git HEAD revision: {head!r}")
    if requested_revision is None:
        return head
    if not REVISION.fullmatch(requested_revision):
        raise ValueError(f"Invalid Git revision: {requested_revision!r}")
    resolved = _run(
        ["git", "rev-parse", "--verify", f"{requested_revision}^{{commit}}"],
        repository,
        capture_output=True,
    ).strip()
    if resolved != head:
        raise ValueError("Requested image revision does not match the current Git HEAD")
    return head


def _ensure_tracked_files_clean(repository: Path) -> None:
    """Reject publication when the labeled revision does not describe the build."""
    status = _run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        repository,
        capture_output=True,
    )
    if status:
        raise ValueError(
            "Refusing to publish when tracked or non-ignored untracked files "
            "are not represented by the labeled revision"
        )


def build_labels(
    repository: Path, revision: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Build the canonical labels and return them with image configuration."""
    manifest = load_manifest(repository)
    config = {key: manifest["image"][key].strip() for key in REQUIRED_IMAGE_KEYS}
    labels = {
        "org.opencontainers.image.title": config["title"],
        "org.opencontainers.image.version": manifest["computation"]["version"],
        "org.opencontainers.image.revision": revision,
        "org.opencontainers.image.source": config["source"],
        "org.neuroflame.computation-api.version": manifest["compatibility"][
            "computationApiVersion"
        ],
        "org.neuroflame.boilerplate.version": manifest["compatibility"][
            "boilerplateVersion"
        ],
        "org.neuroflame.nvflare.version": _read_nvflare_version(repository),
    }
    return labels, config


def _image_tags(config: dict[str, str], version: str, revision: str) -> list[str]:
    """Return floating, release, and revision image references."""
    repository = config["repository"]
    prefix = config["tagPrefix"]
    return [
        f"{repository}:{config['floatingTag']}",
        f"{repository}:{prefix}{version}",
        f"{repository}:{prefix}{revision}",
    ]


def _verify_pushed_platforms(tag: str, platforms: list[str], repository: Path) -> None:
    """Confirm the pushed manifest list covers every requested platform."""
    output = _run(
        ["docker", "buildx", "imagetools", "inspect", tag],
        repository,
        capture_output=True,
    )
    missing = [platform for platform in platforms if platform not in output]
    if missing:
        raise RuntimeError(
            f"Pushed manifest for {tag} is missing platform(s): {', '.join(missing)}"
        )


def _manifest_digest(tag: str, repository: Path) -> str:
    """Return the digest of the pushed (possibly multi-platform) manifest."""
    output = _run(
        ["docker", "buildx", "imagetools", "inspect", tag],
        repository,
        capture_output=True,
    )
    match = MANIFEST_DIGEST.search(output)
    if not match:
        raise RuntimeError("docker buildx imagetools inspect did not report a digest")
    return match.group(1)


ARCH_ALIASES = {"x86_64": "amd64", "aarch64": "arm64"}


def _host_platform(repository: Path) -> str:
    """Return the Docker daemon's own OS/architecture as a buildx platform."""
    raw = _run(
        ["docker", "info", "--format", "{{.OSType}}/{{.Architecture}}"],
        repository,
        capture_output=True,
    ).strip()
    os_type, _, arch = raw.partition("/")
    return f"{os_type}/{ARCH_ALIASES.get(arch, arch)}"


def publish(
    repository: Path,
    *,
    revision: str | None,
    platform: str,
    push: bool,
    local: bool,
) -> None:
    """Build, inspect, tag, and optionally push a computation image.

    `platform` may name several comma-separated platforms (the default is
    both `linux/amd64` and `linux/arm64`). `buildx` can only `--push` a
    multi-platform result straight to a registry; it cannot `--load` one
    into the local image store for inspection. `--local` and `--no-push`
    builds, which do load locally, therefore require a single platform.
    """
    resolved_revision = _get_revision(repository, revision)
    if push:
        _ensure_tracked_files_clean(repository)
    labels, config = build_labels(repository, resolved_revision)
    tags = _image_tags(
        config, labels["org.opencontainers.image.version"], resolved_revision
    )
    platforms = [name.strip() for name in platform.split(",") if name.strip()]
    if not platforms:
        raise ValueError("--platform must name at least one platform")
    if not push and len(platforms) > 1:
        raise ValueError(
            "--local and --no-push build a single platform so it can be "
            "loaded and inspected locally; pass one --platform for those, "
            "e.g. --platform linux/arm64"
        )

    build_command = [
        "docker",
        "buildx",
        "build",
        "--platform",
        platform,
        "-f",
        "Dockerfile-prod",
    ]
    for key, value in labels.items():
        build_command.extend(["--label", f"{key}={value}"])
    for tag in tags:
        build_command.extend(["-t", tag])
    build_command.append("--push" if push else "--load")
    build_command.append(".")
    _run(build_command, repository)

    if push:
        _verify_pushed_platforms(tags[0], platforms, repository)
        digest = _manifest_digest(tags[0], repository)
        print(f"Published multi-platform image: {config['repository']}@{digest}")
        print(f"Platforms: {', '.join(platforms)}")
        return

    inspected = _run(
        ["docker", "image", "inspect", tags[0], "--format", "{{json .Config.Labels}}"],
        repository,
        capture_output=True,
    )
    actual_labels = json.loads(inspected)
    if any(actual_labels.get(key) != value for key, value in labels.items()):
        raise RuntimeError("Built image does not contain the required metadata labels")

    if local:
        image_id = _run(
            ["docker", "image", "inspect", tags[0], "--format", "{{.Id}}"],
            repository,
            capture_output=True,
        ).strip()
        if not DIGEST.fullmatch(image_id):
            raise RuntimeError(f"Docker returned an invalid local image ID: {image_id}")
        print(f"NeuroFLAME local image: {tags[0]}")
        print(f"Local image ID: {image_id}")
        return

    print(f"Built and validated {tags[0]} for {platform} without pushing")


def _build_parser() -> argparse.ArgumentParser:
    """Create the publisher command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", nargs="?", help="explicit Git revision")
    parser.add_argument(
        "--platform",
        default=None,
        help=(
            "target container platform(s), comma-separated for a multi-arch "
            f"push (default: {DEFAULT_PLATFORMS}); --local and --no-push "
            "default to the host's own platform instead, since buildx can "
            "only load a single platform into the local image store"
        ),
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--no-push", action="store_true", help="build and validate without pushing"
    )
    destination.add_argument(
        "--local",
        action="store_true",
        help="build labeled local tags for NeuroFLAME development",
    )
    return parser


def main() -> None:
    """Run the computation image publisher."""
    args = _build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    push = not args.no_push and not args.local
    platform = args.platform
    if platform is None:
        platform = DEFAULT_PLATFORMS if push else _host_platform(repository)
    publish(
        repository,
        revision=args.revision,
        platform=platform,
        push=push,
        local=args.local,
    )


if __name__ == "__main__":
    main()
