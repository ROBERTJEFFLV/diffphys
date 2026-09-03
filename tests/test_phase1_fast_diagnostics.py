from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from diagnostics.closed_loop_spectrum import (
    ArnoldiConfig,
    arnoldi_spectrum,
    orthogonal_projector_from_basis,
)
from diagnostics.integral_sensitivity import compute_integral_sensitivity
from diagnostics.phase_cvar_analysis import (
    CvarAlignmentConfig,
    PhaseAnalysisConfig,
    analyze_h250_phase,
    compute_cvar_alignment,
    summarize_phase_group_profiles,
)
from diagnostics.streaming_phase1 import Phase1StreamingAccumulator, StreamingConfig


def test_integral_sensitivity_separates_main_explicit_and_damping_dependence() -> None:
    dtype = torch.float64
    integral = torch.tensor([[0.1, -0.2, 0.3], [0.2, 0.1, -0.1]], dtype=dtype)
    main = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.5, 0.2, 0.1]], dtype=dtype
    )
    explicit = -0.5 * main
    damping_dependence = 4.0 * main
    desired = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=dtype)
    update = torch.tensor([[1.0, 0.0, 0.0]] * 2, dtype=dtype)
    frame = compute_integral_sensitivity(
        integral=integral,
        main_wrench_fn=lambda value: value @ main.T,
        main_plus_integral_wrench_fn=lambda value: value @ (main + explicit).T,
        full_wrench_fn=lambda value: value @ (main + explicit + damping_dependence).T,
        desired_wrench_direction=desired,
        integral_update_direction=update,
        scenario_uid=["a", "b"],
    )
    assert np.allclose(frame["d_collective_d_ix_main"], 1.0)
    assert np.allclose(frame["d_collective_d_ix_explicit"], -0.5)
    assert np.allclose(frame["d_collective_d_ix_total"], 4.5)
    expected_cancellation = 1.0 - np.linalg.norm(0.5 * main.numpy()) / (
        np.linalg.norm(main.numpy()) + np.linalg.norm(explicit.numpy())
    )
    assert np.allclose(frame["cancellation_fraction"], expected_cancellation)
    assert np.all(frame["actual_update_wrench_cosine"] > 0)
    assert np.all(frame["finite_difference_central_relative_error"] < 1e-8)


def test_matrix_free_arnoldi_finds_projected_oscillatory_eigenvalues() -> None:
    dtype = torch.float64
    radius, angle = 0.98, 0.2
    matrix = torch.tensor(
        [
            [radius * np.cos(angle), -radius * np.sin(angle), 0.0],
            [radius * np.sin(angle), radius * np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )
    state = torch.tensor([0.3, -0.2, 0.1], dtype=dtype)
    projector = orthogonal_projector_from_basis(torch.tensor([0.0, 0.0, 1.0], dtype=dtype))
    result = arnoldi_spectrum(
        step_map=lambda value: matrix @ value,
        state=state,
        projector=projector,
        config=ArnoldiConfig(krylov_dim=3, num_eigenvalues=2, tolerance=1e-12),
    )
    assert np.isclose(result.spectral_radius, radius, atol=1e-8)
    assert result.jvp_validation_central_error < 1e-8


def test_phase_analysis_keeps_same_uid_from_two_checkpoints_separate() -> None:
    records = []
    for checkpoint, amplitude in (("a", 1.0), ("b", 0.1)):
        for step in range(1, 2001):
            records.append({
                "scenario_uid": "same-uid",
                "checkpoint": checkpoint,
                "seed": 7,
                "failure_group": "success",
                "step": step,
                "omega_norm": amplitude * np.sin(2 * np.pi * 0.4 * (step - 1) * 0.01),
            })
    profiles, _, frequency = analyze_h250_phase(
        pd.DataFrame(records),
        signal_columns=["omega_norm"],
        config=PhaseAnalysisConfig(null_permutations=40, seed=7),
    )
    assert profiles.groupby("checkpoint").size().to_dict() == {"a": 250, "b": 250}
    all_window = frequency[frequency["window"] == "all"]
    assert set(all_window["checkpoint"]) == {"a", "b"}


def test_cvar_alignment_keeps_checkpoints_separate_and_distinguishes_five_six() -> None:
    records = []
    for checkpoint, failures in (("five", 5), ("six", 6)):
        for step in range(100):
            records.append({
                "scenario_uid": "same-uid",
                "checkpoint": checkpoint,
                "seed": 7,
                "step": step,
                "position_norm": 0.06 if step < failures else 0.01,
                "velocity_norm": 0.01,
                "omega_norm": 0.01,
            })
    metrics, ranking = compute_cvar_alignment(
        pd.DataFrame(records), config=CvarAlignmentConfig(required_passes=95)
    )
    by_checkpoint = metrics.set_index("checkpoint")
    assert bool(by_checkpoint.loc["five", "steady_success"])
    assert not bool(by_checkpoint.loc["six", "steady_success"])
    assert by_checkpoint.loc["five", "position_sixth_largest_violation"] == 0
    assert by_checkpoint.loc["six", "position_sixth_largest_violation"] > 0
    assert not ranking.empty


def test_streaming_accumulator_preserves_raw_and_detrended_profiles(tmp_path) -> None:
    config = StreamingConfig(
        batch_size=2,
        signal_names=("position_norm", "velocity_norm", "omega_norm"),
        selection_tail_steps=500,
    )
    accumulator = Phase1StreamingAccumulator(config)
    for step in range(1000):
        accumulator.update(np.array(
            [[step / 1000.0, 0.1, 0.2], [0.2, step / 2000.0, 0.1]], dtype=np.float32
        ))
    phase, frequency, tail, selection = accumulator.finalize(scenario_uid=["a", "b"])
    assert phase.shape[0] == 2 * 250 * 3
    assert frequency.shape[0] == 2 * 3
    assert tail.shape[0] == 2 * 100
    assert selection.shape[0] == 2 * 500
    position = phase[(phase["scenario_uid"] == "a") & (phase["signal"] == "position_norm")]
    assert position["raw_phase_mean"].abs().mean() > 0.1
    assert position["detrended_phase_mean"].abs().max() < 1e-6
    accumulator.write(tmp_path, scenario_uid=["a", "b"])
    early = pd.read_csv(tmp_path / "early_window.csv")
    assert early.shape[0] == 2 * 500
    assert early.groupby("scenario_uid")["step"].agg(["min", "max"]).to_dict("index") == {
        "a": {"min": 1, "max": 500},
        "b": {"min": 1, "max": 500},
    }


def test_phase_group_profile_summary_reports_boundary_and_peak_to_peak() -> None:
    frame = pd.DataFrame({
        "checkpoint": ["q2"] * 4,
        "seed": [7] * 4,
        "failure_group": ["success"] * 4,
        "signal": ["omega_norm"] * 4,
        "profile": ["detrended"] * 4,
        "phase": [0, 1, 2, 3],
        "scenario_count": [8] * 4,
        "mean": [2.0, 1.0, -1.0, 0.5],
        "standard_error": [0.1] * 4,
    })
    summary = summarize_phase_group_profiles(frame).iloc[0]
    assert summary["scenario_count"] == 8
    assert summary["phase_peak_to_peak"] == 3.0
    assert summary["phase_of_max"] == 0
    assert summary["phase_of_min"] == 2
    assert summary["boundary_jump_phase0_minus_last"] == 1.5
