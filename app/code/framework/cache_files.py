"""Provide safe paths for author-managed, run-scoped cache files."""

import os
import re

_CACHE_FILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def cache_file(name: str, *, cache_dir: str) -> str:
    """Return a path for a named file inside the injected run cache directory."""
    if not isinstance(name, str) or not _CACHE_FILE_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Cache filename must contain only letters, numbers, '.', '_', or '-' "
            "and cannot contain path separators"
        )
    if not isinstance(cache_dir, str) or not cache_dir:
        raise ValueError("cache_dir must be a non-empty path")
    return os.path.join(cache_dir, name)
