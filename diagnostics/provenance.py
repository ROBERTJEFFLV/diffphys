from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def canonical_config_hash(config: Mapping[str, Any]) -> str:
    """Hash a resolved configuration mapping with stable JSON normalization."""

    normalized = {
        str(key): _json_value(value)
        for key, value in sorted(config.items(), key=lambda item: str(item[0]))
        if not str(key).endswith("_overridden")
    }
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def git_commit(repo_root: Path) -> str:
    """Return the commit hash, or an explicit non-git marker."""

    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable:not-a-git-repository"


def scenario_uid(seed: int, sample_index: int) -> str:
    return f"seed-{int(seed)}-sample-{int(sample_index):06d}"
