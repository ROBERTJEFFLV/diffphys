from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.phase1_fast_common import (
    append_blocker,
    artifact_fingerprint,
    atomic_write_dataframe,
    atomic_write_json,
    first_existing_column,
    read_table,
    sha256_file,
)
from diagnostics.phase_cvar_analysis import (
    PhaseAnalysisConfig,
    analyze_h250_phase,
    analyze_streaming_phase_profiles,
    compute_cvar_alignment,
    summarize_phase_group_profiles,
)


def _normalize_trace(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    normalized = pd.DataFrame()
    normalized["scenario_uid"] = frame[first_existing_column(frame, ("scenario_uid", "scenario_id", "uid"))]
    normalized["step"] = frame[first_existing_column(frame, ("step", "time_step", "t"))]
    if "phase" in frame.columns:
        normalized["phase"] = frame["phase"]
    for optional in ("checkpoint", "seed", "failure_group", "stage1a_stratum", "training_group"):
        if optional in frame.columns:
            target = "failure_group" if optional == "stage1a_stratum" else optional
            if target not in normalized:
                normalized[target] = frame[optional]
    aliases = {
        "position_norm": ("position_norm", "p_norm", "position_rms"),
        "velocity_norm": ("velocity_norm", "v_norm", "velocity_rms"),
        "omega_norm": ("omega_norm", "rate_norm", "omega_rms"),
        "action_delta_norm": ("action_delta_norm", "action_delta_rms"),
        "hidden_norm": ("hidden_norm", "gru_hidden_norm"),
        "integral_norm": ("integral_norm", "position_integral_norm"),
        "integral_clamp_fraction": ("integral_clamp_fraction", "clamp_fraction"),
        "branch_collective": ("branch_collective", "damping_collective"),
        "branch_torque_norm": ("branch_torque_norm", "damping_torque_norm"),
        "dense_potential": ("dense_potential", "tracking_potential"),
    }
    signals = []
    for target, options in aliases.items():
        source = first_existing_column(frame, options, required=False)
        if source is not None:
            normalized[target] = frame[source]
            signals.append(target)
    return normalized, signals


def _output_names(scope: str) -> dict[str, str]:
    suffix = "" if scope == "formal" else f"_{scope}"
    return {
        "scenario_profiles": f"h250_phase_scenario_profiles{suffix}.csv",
        "phase": f"h250_phase{suffix}.csv",
        "phase_group": f"h250_phase_group_summary{suffix}.csv",
        "frequency": f"h250_frequency{suffix}.csv",
        "cvar": f"cvar_alignment{suffix}.csv",
        "ranking": f"cvar_ranking{suffix}.csv",
        "manifest": f"FAST_COMPLETION_{scope.upper()}_MANIFEST.json",
    }


def _valid_resume(report_dir: Path, manifest_path: Path, fingerprint: str) -> bool:
    if not manifest_path.exists():
        return False
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("pipeline_fingerprint") != fingerprint:
        return False
    outputs = manifest.get("outputs", {})
    return bool(outputs) and all(
        (report_dir / name).is_file() and sha256_file(report_dir / name) == digest
        for name, digest in outputs.items()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable offline H250/CVaR Phase 1 analysis.")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports/q3_root_cause_diagnostics")
    parser.add_argument("--phase-trace", type=Path)
    parser.add_argument("--streaming-profiles", type=Path)
    parser.add_argument("--streaming-frequency", type=Path)
    parser.add_argument("--tail-trace", type=Path)
    parser.add_argument("--scope", choices=("formal", "stage1a_smoke"), required=True)
    parser.add_argument("--seed", type=int, default=1007)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    names = _output_names(args.scope)
    inputs = [path.resolve() for path in (
        args.phase_trace, args.streaming_profiles, args.streaming_frequency, args.tail_trace
    ) if path is not None]
    code = [
        Path(__file__),
        PROJECT_ROOT / "diagnostics/phase_cvar_analysis.py",
        PROJECT_ROOT / "diagnostics/phase1_fast_common.py",
    ]
    provenance = artifact_fingerprint(
        inputs,
        parameters={"scope": args.scope, "seed": args.seed},
        code_paths=code,
    )
    manifest_path = report_dir / names["manifest"]
    if not args.force and _valid_resume(report_dir, manifest_path, provenance["pipeline_fingerprint"]):
        return 0

    outputs: list[Path] = []
    blockers = report_dir / "BLOCKERS.md"
    phase_config = PhaseAnalysisConfig(seed=args.seed)
    if args.streaming_profiles:
        if args.streaming_frequency is None:
            raise ValueError("--streaming-frequency is required with --streaming-profiles")
        _, phase, frequency = analyze_streaming_phase_profiles(
            read_table(args.streaming_profiles), read_table(args.streaming_frequency), config=phase_config
        )
        profiles = pd.DataFrame()  # the fingerprinted wide streaming input is already the scenario artifact
    elif args.phase_trace:
        trace, signals = _normalize_trace(read_table(args.phase_trace))
        if not signals:
            raise RuntimeError("phase trace has no recognized signals")
        profiles, phase, frequency = analyze_h250_phase(trace, signal_columns=signals, config=phase_config)
    else:
        append_blocker(blockers, f"H250 phase analysis ({args.scope})", "No explicit phase input was supplied.")
        profiles = phase = frequency = pd.DataFrame()
    for frame, key in ((profiles, "scenario_profiles"), (phase, "phase"), (frequency, "frequency")):
        if not frame.empty:
            path = report_dir / names[key]
            atomic_write_dataframe(frame, path)
            outputs.append(path)
    if not phase.empty:
        phase_group = summarize_phase_group_profiles(phase)
        if not phase_group.empty:
            path = report_dir / names["phase_group"]
            atomic_write_dataframe(phase_group, path)
            outputs.append(path)

    tail_source = args.tail_trace or args.phase_trace
    if tail_source:
        tail, _ = _normalize_trace(read_table(tail_source))
        required = {"position_norm", "velocity_norm", "omega_norm"}
        if required.issubset(tail.columns):
            alignment, ranking = compute_cvar_alignment(tail)
            if not alignment.empty:
                path = report_dir / names["cvar"]
                atomic_write_dataframe(alignment, path)
                outputs.append(path)
            if not ranking.empty:
                path = report_dir / names["ranking"]
                atomic_write_dataframe(ranking, path)
                outputs.append(path)
        else:
            append_blocker(
                blockers,
                f"CVaR alignment ({args.scope})",
                f"Input `{tail_source}` lacks {sorted(required.difference(tail.columns))}.",
            )
    else:
        append_blocker(blockers, f"CVaR alignment ({args.scope})", "No explicit tail input was supplied.")

    provenance["outputs"] = {path.name: sha256_file(path) for path in outputs}
    atomic_write_json(provenance, manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
