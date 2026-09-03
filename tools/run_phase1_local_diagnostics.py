from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.closed_loop_spectrum import ArnoldiConfig, arnoldi_spectrum, spectrum_to_frames
from diagnostics.formal_rollout import load_q2_policy, select_state
from diagnostics.phase1_fast_common import (
    atomic_write_dataframe,
    atomic_write_json,
    sha256_file,
    summarize_numeric,
)
from diagnostics.q2_local_analysis import (
    Q2ClosedLoopMap,
    capture_q2_snapshots,
    run_integral_sensitivity,
    snapshot_payload,
    snapshots_from_payload,
)
from diagnostics.scenarios import load_matlab_scenarios, read_scenario_rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _checkpoint(checkpoints_csv: Path, label: str) -> Path:
    row = next(item for item in _read_csv(checkpoints_csv) if item["label"] == label)
    path = Path(row["checkpoint_path"])
    if sha256_file(path) != row["checkpoint_sha256"]:
        raise RuntimeError("checkpoint SHA256 mismatch")
    return path


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=path.suffix, dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def capture(args: argparse.Namespace, checkpoint: Path, plan: pd.DataFrame) -> Path:
    output = args.report_dir / "decisive_snapshots.pt"
    if output.exists() and not args.force:
        return output
    scenario_rows = read_scenario_rows(args.scenario_csv)
    ids, full_state = load_matlab_scenarios(args.scenario_csv, device=args.device, dtype=torch.float32)
    del ids
    uid_to_index = {row["scenario_uid"]: index for index, row in enumerate(scenario_rows)}
    selected_uids = list(dict.fromkeys(plan["scenario_uid"].astype(str)))
    indices = torch.tensor([uid_to_index[uid] for uid in selected_uids], device=args.device)
    state = select_state(full_state, indices)
    policy, _ = load_q2_policy(checkpoint, device=args.device, dtype=torch.float32)
    snapshots = capture_q2_snapshots(
        policy,
        state,
        selected_uids,
        plan,
        backend="cuda" if torch.device(args.device).type == "cuda" else "torch",
    )
    _atomic_torch_save(snapshot_payload(snapshots), output)
    return output


def integral(args: argparse.Namespace, checkpoint: Path, snapshot_path: Path) -> None:
    snapshots = snapshots_from_payload(torch.load(snapshot_path, map_location="cpu", weights_only=False))
    policy, _ = load_q2_policy(checkpoint, device="cpu", dtype=torch.float64)
    started = time.perf_counter()
    frame = run_integral_sensitivity(policy, snapshots)
    atomic_write_dataframe(frame, args.report_dir / "integral_sensitivity.csv")
    scenario_metrics = [
        "effective_normalized_sensitivity",
        "effective_cancellation_fraction",
        "effective_actual_update_wrench_cosine",
        "instantaneous_world_force_cosine",
        "actuator_headroom_fraction",
        "jacobian_validation_pass",
        "motor_switch_nonsmooth",
    ]
    scenario_summary = (
        frame.groupby(["failure_group", "scenario_uid"], as_index=False)[scenario_metrics]
        .mean(numeric_only=True)
    )
    atomic_write_dataframe(
        scenario_summary,
        args.report_dir / "integral_sensitivity_scenario_summary.csv",
    )
    group_records: list[dict[str, object]] = []
    for group_index, (failure_group, group) in enumerate(
        scenario_summary.groupby("failure_group", sort=True)
    ):
        for metric in scenario_metrics:
            group_records.append({
                "failure_group": failure_group,
                "metric": metric,
                **summarize_numeric(group[metric].to_numpy(), seed=1007 + group_index),
            })
    atomic_write_dataframe(
        pd.DataFrame(group_records),
        args.report_dir / "integral_sensitivity_group_summary.csv",
    )
    mechanism_summary = (
        frame.groupby(["failure_group", "integral_mechanism"], dropna=False)
        .agg(snapshot_count=("scenario_uid", "size"), scenario_count=("scenario_uid", "nunique"))
        .reset_index()
    )
    atomic_write_dataframe(
        mechanism_summary,
        args.report_dir / "integral_sensitivity_mechanism_counts.csv",
    )
    atomic_write_json(
        {"elapsed_s": time.perf_counter() - started, "snapshot_count": len(snapshots)},
        args.report_dir / "INTEGRAL_SENSITIVITY_RUN.json",
    )


def _spectrum_subset(snapshots, *, allowed_groups: set[str], per_group: int):
    selected = []
    for group, group_snapshots in pd.Series(snapshots).groupby(
        [snapshot.metadata.get("failure_group", "") for snapshot in snapshots]
    ):
        candidates = list(group_snapshots)
        if group not in allowed_groups:
            continue
        if group == "position-only":
            preferred = [item for item in candidates if item.metadata.get("selection_reason") == "late_position_peak"]
        elif group in {"omega-related", "dynamic-hard"}:
            preferred = [item for item in candidates if item.metadata.get("selection_reason") == "late_omega_peak"]
        elif group == "success":
            preferred = [item for item in candidates if item.metadata.get("selection_reason") == "late_position_peak"]
        else:
            preferred = []
        selected.extend(sorted(preferred, key=lambda item: (item.scenario_uid, item.step))[:per_group])
    return selected


def spectrum(args: argparse.Namespace, checkpoint: Path, snapshot_path: Path) -> None:
    snapshots = snapshots_from_payload(torch.load(snapshot_path, map_location="cpu", weights_only=False))
    allowed_groups = set(args.spectrum_groups.split(","))
    snapshots = _spectrum_subset(snapshots, allowed_groups=allowed_groups, per_group=args.per_spectrum_group)
    policy, _ = load_q2_policy(checkpoint, device="cpu", dtype=torch.float64)
    spectra = []
    modes = []
    validations = []
    started = time.perf_counter()
    for snapshot_index, snapshot in enumerate(snapshots):
        local_map = Q2ClosedLoopMap(policy, snapshot)
        metadata = {
            "snapshot_id": f"{snapshot.scenario_uid}:{snapshot.step}:{snapshot.metadata.get('selection_reason', '')}",
            "step": snapshot.step,
            "failure_group": snapshot.metadata.get("failure_group", ""),
            "selection_reason": snapshot.metadata.get("selection_reason", ""),
            "integral_clamped": bool((snapshot.integral.abs() >= 0.5 - 1e-7).any()),
        }
        spectrum_options = {
            "raw": None,
            "yaw_projected": local_map.yaw_projector(),
        }
        for kind in args.spectrum_kinds.split(","):
            projector = spectrum_options[kind]
            try:
                result = arnoldi_spectrum(
                    step_map=local_map,
                    state=local_map.base_vector,
                    projector=projector,
                    config=ArnoldiConfig(
                        krylov_dim=args.krylov_dim,
                        num_eigenvalues=args.num_eigenvalues,
                        seed=1007 + snapshot_index,
                        residual_tolerance=args.residual_tolerance,
                    ),
                    nonsmooth=local_map.nonsmooth,
                )
                spectrum_frame, mode_frame, validation_frame = spectrum_to_frames(
                    result,
                    scenario_uid=snapshot.scenario_uid,
                    checkpoint=args.checkpoint_label,
                    spectrum_kind=kind,
                    dt=0.01,
                    layout=local_map.layout,
                    metadata=metadata,
                )
                spectra.append(spectrum_frame)
                modes.append(mode_frame)
                validations.append(validation_frame)
            except Exception as error:
                validations.append(pd.DataFrame([{
                    "scenario_uid": snapshot.scenario_uid,
                    "checkpoint": args.checkpoint_label,
                    "spectrum_kind": kind,
                    "pass": False,
                    "error": repr(error),
                    **metadata,
                }]))
    spectrum_frame = pd.concat(spectra, ignore_index=True) if spectra else pd.DataFrame()
    mode_frame = pd.concat(modes, ignore_index=True) if modes else pd.DataFrame()
    validation_frame = pd.concat(validations, ignore_index=True)
    suffix = f"_{args.spectrum_suffix}" if args.spectrum_suffix else ""
    if not spectrum_frame.empty:
        atomic_write_dataframe(spectrum_frame, args.report_dir / f"closed_loop_spectrum{suffix}.csv")
    if not mode_frame.empty:
        atomic_write_dataframe(mode_frame, args.report_dir / f"closed_loop_modes{suffix}.csv")
    atomic_write_dataframe(validation_frame, args.report_dir / f"jacobian_validation{suffix}.csv")
    atomic_write_json(
        {
            "elapsed_s": time.perf_counter() - started,
            "snapshot_count": len(snapshots),
            "requested_spectra": len(args.spectrum_kinds.split(",")) * len(snapshots),
            "successful_spectra": len(spectra),
        },
        args.report_dir / f"CLOSED_LOOP_SPECTRUM_RUN{suffix.upper()}.json",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run decisive-snapshot Q2 JVP and Arnoldi diagnostics.")
    parser.add_argument("command", choices=("capture", "integral", "spectrum", "all"))
    parser.add_argument("--checkpoint-label", default="seed_7_group_T0_physical_steps_96000000")
    parser.add_argument("--checkpoints-csv", type=Path, default=PROJECT_ROOT / "reports/q3_root_cause_diagnostics/CHECKPOINTS.csv")
    parser.add_argument("--scenario-csv", type=Path, default=PROJECT_ROOT / "reports/q3_root_cause_diagnostics/SCENARIO_MANIFEST.csv")
    parser.add_argument("--plan", type=Path, default=PROJECT_ROOT / "reports/q3_root_cause_diagnostics/decisive_snapshot_plan.csv")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports/q3_root_cause_diagnostics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--krylov-dim", type=int, default=32)
    parser.add_argument("--num-eigenvalues", type=int, default=8)
    parser.add_argument("--residual-tolerance", type=float, default=1e-5)
    parser.add_argument("--spectrum-groups", default="success,position-only,omega-related,dynamic-hard")
    parser.add_argument("--per-spectrum-group", type=int, default=8)
    parser.add_argument("--spectrum-kinds", default="raw,yaw_projected")
    parser.add_argument("--spectrum-suffix", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.report_dir = args.report_dir.resolve()
    checkpoint = _checkpoint(args.checkpoints_csv, args.checkpoint_label)
    plan = pd.read_csv(args.plan)
    snapshot_path = args.report_dir / "decisive_snapshots.pt"
    if args.command in {"capture", "all"}:
        snapshot_path = capture(args, checkpoint, plan)
    if args.command in {"integral", "spectrum", "all"} and not snapshot_path.exists():
        raise FileNotFoundError(snapshot_path)
    if args.command in {"integral", "all"}:
        integral(args, checkpoint, snapshot_path)
    if args.command in {"spectrum", "all"}:
        spectrum(args, checkpoint, snapshot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
