from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _temporary_path(destination: Path, *, mode: str, encoding: str | None = None) -> tuple[Any, Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode=mode,
        encoding=encoding,
        suffix=destination.suffix,
        prefix=f".{destination.name}.",
        dir=destination.parent,
        delete=False,
    )
    return handle, Path(handle.name)


def atomic_write_dataframe(frame: pd.DataFrame, path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    handle, temporary = _temporary_path(destination, mode="wb")
    handle.close()
    try:
        suffix = destination.suffix.lower()
        if suffix == ".csv":
            frame.to_csv(temporary, index=False)
        elif suffix in {".parquet", ".pq"}:
            frame.to_parquet(temporary, index=False)
        else:
            raise ValueError(f"Unsupported dataframe format: {suffix}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
        # Some mounted workspaces can leave a second, suffixed copy of a large
        # NamedTemporaryFile during atomic replacement. It is never a measured
        # artifact and must not be mistaken for one by provenance collection.
        for orphan in destination.parent.glob(f".{destination.name}.*"):
            if orphan.is_file():
                orphan.unlink(missing_ok=True)


def atomic_write_text(text: str, path: str | os.PathLike[str]) -> None:
    destination = Path(path)
    handle, temporary = _temporary_path(destination, mode="w", encoding="utf-8")
    try:
        with handle:
            handle.write(text)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: Any, path: str | os.PathLike[str]) -> None:
    atomic_write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), path)


def append_blocker(path: str | os.PathLike[str], title: str, details: str) -> None:
    destination = Path(path)
    existing = destination.read_text(encoding="utf-8") if destination.exists() else "# Blockers\n\n"
    section = f"## {title}\n\n{details.strip()}\n\n"
    if section not in existing:
        atomic_write_text(existing + section, destination)


def artifact_fingerprint(
    inputs: Sequence[str | os.PathLike[str]],
    *,
    parameters: Mapping[str, Any],
    code_paths: Sequence[str | os.PathLike[str]] = (),
) -> dict[str, Any]:
    input_hashes = {
        str(Path(path).resolve()): sha256_file(path)
        for path in inputs
        if Path(path).is_file()
    }
    code_hashes = {
        str(Path(path).resolve()): sha256_file(path)
        for path in code_paths
        if Path(path).is_file()
    }
    payload: dict[str, Any] = {
        "inputs": input_hashes,
        "code": code_hashes,
        "parameters": dict(parameters),
    }
    payload["pipeline_fingerprint"] = stable_json_hash(payload)
    return payload


def scenario_bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 4000,
    seed: int = 1007,
    statistic: str = "mean",
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, array.size, size=(n_bootstrap, array.size), endpoint=False)]
    if statistic == "mean":
        estimates = sampled.mean(axis=1)
    elif statistic == "median":
        estimates = np.median(sampled, axis=1)
    else:
        raise ValueError(f"Unsupported statistic: {statistic}")
    alpha = 1.0 - confidence
    low, high = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def first_existing_column(frame: pd.DataFrame, aliases: Iterable[str], *, required: bool = True) -> str | None:
    normalized = {str(column).strip().lower(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias.strip().lower() in normalized:
            return normalized[alias.strip().lower()]
    if required:
        raise KeyError(f"None of the required columns exists: {list(aliases)}")
    return None


def read_table(path: str | os.PathLike[str], **kwargs: Any) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source, **kwargs)
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source, **kwargs)
    raise ValueError(f"Unsupported table format: {source}")


def summarize_numeric(values: Sequence[float] | np.ndarray, *, seed: int = 1007) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {key: (0 if key == "n" else float("nan")) for key in (
            "n", "mean", "median", "std", "q05", "q25", "q75", "q95",
            "bootstrap_ci_low", "bootstrap_ci_high",
        )}
    ci_low, ci_high = scenario_bootstrap_ci(array, seed=seed)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
    }
