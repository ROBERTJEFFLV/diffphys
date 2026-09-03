from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import platform
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retain_bank import (
    PHYSICAL_FIT_SAMPLER_SOURCE_METADATA_KEY,
    load_retain_bank,
    physical_fit_sampler_source_sha256,
    rigid_body_inertia_violation_mask,
    validate_retain_bank_for_sampler,
)


DEFAULT_BANK = ROOT / "reports/next_stage_retain_bank/retain_bank.pt"
DEFAULT_OUTPUT = ROOT / "reports/retain_bank_compatibility_audit_20260806"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit retain-bank sampler compatibility.")
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--retain-fraction", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 0.0 <= args.retain_fraction <= 1.0:
        raise ValueError("--retain-fraction must be in [0, 1]")

    bank_path = args.bank.resolve()
    bank = load_retain_bank(bank_path)
    violations = int(rigid_body_inertia_violation_mask(bank).sum().item())
    violation_fraction = violations / len(bank)
    physical_fit_rejected = False
    rejection_reason = ""
    try:
        validate_retain_bank_for_sampler(bank, "physical-fit")
    except ValueError as error:
        physical_fit_rejected = True
        rejection_reason = str(error)
    validate_retain_bank_for_sampler(bank, "physical")

    current_sampler_sha = physical_fit_sampler_source_sha256()
    relabeled_bank = replace(
        bank,
        metadata={
            **bank.metadata,
            "broad_sampler": "physical-fit",
            PHYSICAL_FIT_SAMPLER_SOURCE_METADATA_KEY: current_sampler_sha,
        },
    )
    relabeled_physical_fit_rejected = False
    relabeled_rejection_reason = ""
    try:
        validate_retain_bank_for_sampler(relabeled_bank, "physical-fit")
    except ValueError as error:
        relabeled_physical_fit_rejected = True
        relabeled_rejection_reason = str(error)

    non_finite_float_fields = sorted(
        name
        for name, value in bank.state.items()
        if (torch.is_floating_point(value) or torch.is_complex(value))
        and not bool(torch.isfinite(value).all().item())
    )

    row = {
        "bank_samples": len(bank),
        "metadata_broad_sampler": bank.metadata.get("broad_sampler", ""),
        "metadata_physical_fit_sampler_source_sha256": bank.metadata.get(
            PHYSICAL_FIT_SAMPLER_SOURCE_METADATA_KEY, ""
        ),
        "current_physical_fit_sampler_source_sha256": physical_fit_sampler_source_sha256(),
        "sampler_source_sha256_matches": int(
            bank.metadata.get(PHYSICAL_FIT_SAMPLER_SOURCE_METADATA_KEY)
            == physical_fit_sampler_source_sha256()
        ),
        "rigid_body_inertia_violations": violations,
        "rigid_body_inertia_violation_fraction": violation_fraction,
        "configured_retain_fraction": args.retain_fraction,
        "estimated_invalid_fraction_of_all_resets_if_mixed": args.retain_fraction * violation_fraction,
        "historical_physical_accepted": 1,
        "physical_fit_rejected": int(physical_fit_rejected),
        "physical_fit_rejection_reason": rejection_reason,
        "diagnostic_relabel_with_current_sha_rejected": int(
            relabeled_physical_fit_rejected
        ),
        "diagnostic_relabel_rejection_reason": relabeled_rejection_reason,
        "non_finite_floating_state_field_count": len(non_finite_float_fields),
        "non_finite_floating_state_fields": ";".join(non_finite_float_fields),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "SUMMARY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)
    sampler_metadata = bank.metadata.get(PHYSICAL_FIT_SAMPLER_SOURCE_METADATA_KEY)
    (output_dir / "SUMMARY.md").write_text(
        f"""# Retain-bank compatibility audit

The audited bank contains {len(bank):,} scenarios and declares
`broad_sampler={bank.metadata.get('broad_sampler')!r}`. It is accepted only by the
historical `physical` path and is rejected by the `physical-fit` guard. Its
metadata sampler-source SHA is {sampler_metadata!r}; the current required
`env_l2f.py` SHA-256 is `{current_sampler_sha}`.

Independently of the provenance rejection, {violations}/{len(bank)}
({violation_fraction:.3%}) samples violate a necessary principal-inertia
condition. At the configured retain fraction {args.retain_fraction:.3f}, blindly
mixing this bank would expose approximately
{args.retain_fraction * violation_fraction:.3%} of all reset slots to such
inertias. All serialized floating state fields finite: **{not non_finite_float_fields}**.

For a guard stress test only, the in-memory metadata was relabeled as
`physical-fit` and assigned the current source SHA; it was still rejected:
`{relabeled_rejection_reason}`. This relabel is not evidence of sampler origin
and is never written back to the bank.

This report shows that the current legacy bank cannot enter a `physical-fit`
training run. It does **not** validate a replacement bank. A replacement must be
freshly generated by the current sampler, retain the exact source SHA, and pass
all finite-value, strict-positive, motor-time and inertia checks before use by
`physical-fit` training.
""",
        encoding="utf-8",
    )
    provenance = {
        "bank": bank_path.relative_to(ROOT).as_posix(),
        "bank_sha256": _sha256(bank_path),
        "tool": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "tool_sha256": _sha256(Path(__file__).resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "current_physical_fit_sampler_source_sha256": current_sampler_sha,
        "source_hashes": {
            "env_l2f.py": _sha256(ROOT / "env_l2f.py"),
            "retain_bank.py": _sha256(ROOT / "retain_bank.py"),
            "tools/build_retain_bank.py": _sha256(ROOT / "tools/build_retain_bank.py"),
        },
    }
    (output_dir / "RUN_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), **row}, indent=2, ensure_ascii=False))
    if not physical_fit_rejected:
        raise RuntimeError("legacy retain bank was not rejected for physical-fit")
    if violations > 0 and not relabeled_physical_fit_rejected:
        raise RuntimeError("invalid diagnostic relabel was not rejected for physical-fit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
