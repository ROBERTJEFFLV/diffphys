from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "diffphys_continuity_cadence_complete_20260806.zip"
PACKAGE_ROOT = "diffphys_continuity_cadence_complete_20260806"
INDEX_NAME = "BUNDLE_INDEX_20260806.csv"

PROJECT_DIRECTORIES = (
    "checkpoints",
    "configs",
    "cuda_ext",
    "diagnostic_inputs",
    "diagnostics",
    "matlab_l2f",
    "reference",
    "runs",
    "tests",
    "tools",
    "物理配置",
)

REPORT_DIRECTORIES = (
    "reports/continuity_cadence_9p6m_20260804",
    "reports/continuity_cadence_cuda_smoke_20260804",
    "reports/continuity_cadence_internal_eval_20260804",
    "reports/continuity_cadence_analysis_20260804",
    "reports/continuity_cadence_formal_analysis_20260804",
    "reports/continuity_cadence_label_parity_20260804",
    "reports/continuity_cadence_cpu_confirmation_20260804",
    "reports/continuity_cadence_matlab_20260804",
    "reports/next_stage_retain_bank",
)

REPORT_FILES = (
    "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt",
)

EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".torch_extensions",
    "__pycache__",
    "tmp",
}
EXCLUDED_SUFFIXES = {".zip", ".pyc", ".pyo"}
EXCLUDED_ROOT_NAMES = {
    INDEX_NAME,
    DEFAULT_OUTPUT.name,
    f"{DEFAULT_OUTPUT.name}.sha256",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _excluded(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return path.name in {".DS_Store", "Thumbs.db"}


def _collect_path(path: Path, destination: set[Path]) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        if not _excluded(path):
            destination.add(path.resolve())
        return
    for candidate in path.rglob("*"):
        if candidate.is_file() and not _excluded(candidate):
            destination.add(candidate.resolve())


def _category(relative: Path) -> str:
    parts = relative.parts
    if parts[0] == "reports":
        if relative.suffix.lower() in {".pt", ".mat"}:
            return "report_checkpoint_or_model"
        return "report"
    if parts[0] == "checkpoints":
        return "checkpoint"
    if parts[0] == "diagnostic_inputs":
        return "diagnostic_input"
    if parts[0] == "tests":
        return "test"
    if parts[0] == "configs":
        return "config"
    return "project_source_or_asset"


def _payload_files() -> list[Path]:
    files: set[Path] = set()
    for path in ROOT.iterdir():
        if path.is_file() and path.name not in EXCLUDED_ROOT_NAMES and not _excluded(path):
            files.add(path.resolve())
    for relative in PROJECT_DIRECTORIES:
        _collect_path(ROOT / relative, files)
    for relative in REPORT_DIRECTORIES:
        _collect_path(ROOT / relative, files)
    for relative in REPORT_FILES:
        _collect_path(ROOT / relative, files)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def _write_index(files: list[Path], path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for file_path in files:
        relative = file_path.relative_to(ROOT)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "category": _category(relative),
                "size_bytes": file_path.stat().st_size,
                "sha256": _sha256(file_path),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the complete continuity/cadence deliverable.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    checksum_path = output.with_name(f"{output.name}.sha256")
    if (output.exists() or checksum_path.exists()) and not args.force:
        raise FileExistsError(f"refusing to overwrite existing deliverable: {output}")
    if output.parent != ROOT:
        output.parent.mkdir(parents=True, exist_ok=True)

    files = _payload_files()
    index_path = ROOT / INDEX_NAME
    rows = _write_index(files, index_path)
    payload_bytes = sum(int(row["size_bytes"]) for row in rows)

    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=compression,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for file_path in files:
            relative = file_path.relative_to(ROOT).as_posix()
            archive.write(file_path, f"{PACKAGE_ROOT}/{relative}")
        archive.write(index_path, f"{PACKAGE_ROOT}/{INDEX_NAME}")

    with zipfile.ZipFile(output, mode="r") as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"ZIP CRC verification failed: {corrupt}")
        names = archive.namelist()
    expected_entries = len(files) + 1
    if len(names) != expected_entries or len(set(names)) != expected_entries:
        raise RuntimeError(
            f"ZIP entry count mismatch: {len(names)} entries, expected {expected_entries}"
        )
    if any(name.lower().endswith(".zip") for name in names):
        raise RuntimeError("nested ZIP unexpectedly entered the deliverable")

    archive_sha256 = _sha256(output)
    checksum_path.write_text(f"{archive_sha256}  {output.name}\n", encoding="ascii")
    result = {
        "archive": str(output),
        "archive_sha256": archive_sha256,
        "archive_bytes": output.stat().st_size,
        "payload_files": len(files),
        "zip_entries": len(names),
        "payload_bytes": payload_bytes,
        "index": str(index_path),
        "checksum_file": str(checksum_path),
        "crc_verified": True,
        "nested_zip_count": 0,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
