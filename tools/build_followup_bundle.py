from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_continuity_cadence_bundle as base


base.DEFAULT_OUTPUT = (
    base.ROOT / "diffphys_continuity_cadence_followup_validated_20260806.zip"
)
base.PACKAGE_ROOT = "diffphys_continuity_cadence_followup_validated_20260806"
base.REPORT_DIRECTORIES = base.REPORT_DIRECTORIES + (
    "reports/continuity_cadence_mechanism_audit_20260806",
    "reports/physical_fit_sampler_audit_20260806",
    "reports/size_causality_audit_20260806",
    "reports/physical_failure_axes_audit_20260806",
    "reports/retain_bank_compatibility_audit_20260806",
    "reports/arm_d_cpu_semantic_smoke_20260806",
    "reports/arm_d_cpu_schedule_smoke_20260806",
)
base.REPORT_FILES = base.REPORT_FILES + (
    "reports/EXPERIMENT_DECISIONS_20260806.md",
    "reports/EXPERIMENT_DECISIONS_20260806_ZH.md",
    "reports/PREREGISTRATION_ARM_D_20260806.md",
)
base.EXCLUDED_ROOT_NAMES = {
    base.INDEX_NAME,
    base.DEFAULT_OUTPUT.name,
    f"{base.DEFAULT_OUTPUT.name}.sha256",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and fully verify the continuity/cadence follow-up bundle."
    )
    parser.add_argument("--output", type=Path, default=base.DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    checksum_path = output.with_name(f"{output.name}.sha256")
    if (output.exists() or checksum_path.exists()) and not args.force:
        raise FileExistsError(f"refusing to overwrite existing deliverable: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    index_path = base.ROOT / base.INDEX_NAME
    excluded_outputs = {output, checksum_path, index_path.resolve()}
    files = [path for path in base._payload_files() if path not in excluded_outputs]
    rows: list[dict[str, object]] = []
    for file_path in files:
        relative = file_path.relative_to(base.ROOT)
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "category": base._category(relative),
                "size_bytes": file_path.stat().st_size,
                "sha256": base._sha256(file_path),
            }
        )

    index_text = io.StringIO(newline="")
    writer = csv.DictWriter(index_text, fieldnames=tuple(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    index_bytes = index_text.getvalue().encode("utf-8")

    temp_handle = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    )
    temp_output = Path(temp_handle.name)
    temp_handle.close()
    try:
        with zipfile.ZipFile(
            temp_output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for file_path in files:
                relative = file_path.relative_to(base.ROOT).as_posix()
                archive.write(file_path, f"{base.PACKAGE_ROOT}/{relative}")
            archive.writestr(
                f"{base.PACKAGE_ROOT}/{base.INDEX_NAME}", index_bytes
            )

        expected_by_path = {str(row["relative_path"]): row for row in rows}
        with zipfile.ZipFile(temp_output, mode="r") as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise RuntimeError(f"ZIP CRC verification failed: {corrupt}")
            names = archive.namelist()
            expected_entries = len(files) + 1
            if len(names) != expected_entries or len(set(names)) != expected_entries:
                raise RuntimeError(
                    f"ZIP entry count mismatch: {len(names)} entries, "
                    f"expected {expected_entries}"
                )
            if any(name.lower().endswith(".zip") for name in names):
                raise RuntimeError("nested ZIP unexpectedly entered the deliverable")

            archived_index = archive.read(
                f"{base.PACKAGE_ROOT}/{base.INDEX_NAME}"
            )
            if archived_index != index_bytes:
                raise RuntimeError("archived bundle index differs from generated index")
            archived_rows = list(
                csv.DictReader(io.StringIO(archived_index.decode("utf-8")))
            )
            if len(archived_rows) != len(expected_by_path):
                raise RuntimeError("archived bundle index row count mismatch")

            prefix = f"{base.PACKAGE_ROOT}/"
            payload_names = {
                name[len(prefix) :]: name
                for name in names
                if name != f"{prefix}{base.INDEX_NAME}"
            }
            if set(payload_names) != set(expected_by_path):
                raise RuntimeError("archived payload paths differ from bundle index")
            for relative, row in expected_by_path.items():
                info = archive.getinfo(payload_names[relative])
                if info.file_size != int(row["size_bytes"]):
                    raise RuntimeError(f"archived size mismatch: {relative}")
                digest = base.hashlib.sha256()
                with archive.open(info) as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != row["sha256"]:
                    raise RuntimeError(f"archived SHA-256 mismatch: {relative}")

        os.replace(temp_output, output)
    finally:
        if temp_output.exists():
            temp_output.unlink()

    index_temp_handle = tempfile.NamedTemporaryFile(
        prefix=f".{base.INDEX_NAME}.", suffix=".tmp", dir=base.ROOT, delete=False
    )
    index_temp = Path(index_temp_handle.name)
    try:
        index_temp_handle.write(index_bytes)
        index_temp_handle.flush()
        os.fsync(index_temp_handle.fileno())
    finally:
        index_temp_handle.close()
    os.replace(index_temp, index_path)

    archive_sha256 = base._sha256(output)
    checksum_path.write_text(
        f"{archive_sha256}  {output.name}\n", encoding="ascii"
    )
    result = {
        "archive": str(output),
        "archive_sha256": archive_sha256,
        "archive_bytes": output.stat().st_size,
        "payload_files": len(files),
        "zip_entries": len(files) + 1,
        "payload_bytes": sum(int(row["size_bytes"]) for row in rows),
        "index": str(index_path),
        "index_rows": len(rows),
        "checksum_file": str(checksum_path),
        "crc_verified": True,
        "index_hashes_verified": True,
        "nested_zip_count": 0,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
