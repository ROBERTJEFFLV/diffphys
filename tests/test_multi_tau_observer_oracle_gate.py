from __future__ import annotations

import pytest

from tools.diagnose_multi_tau_observer_oracle import (
    PUBLICATION_STEPS,
    _evaluate_pre_registered_gate,
)


def _contract() -> tuple[dict, dict]:
    coverage = {
        "best_candidate": {
            str(step): {
                "ratio_median": 0.50,
                "ratio_p95": 0.80,
                "all_authority_cells_finite": True,
                "per_scenario_best_to_legacy_ratio": [[0.4, 0.5]],
            }
            for step in PUBLICATION_STEPS
        }
    }
    # Two final seeds x two scenarios x three publications.  Values are MSE,
    # not RMS; the gate must bootstrap bank - legacy from these paired cells.
    legacy_mse = [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]] for _ in range(2)]
    bank_mse = [[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]] for _ in range(2)]
    evaluation = {
        "legacy24": {
            "per_checkpoint_rms": {str(step): 1.0 for step in PUBLICATION_STEPS},
            "per_publication_dimension_rms": [[1.0] * 6 for _ in PUBLICATION_STEPS],
            "authority_cell_rms": {"tw0_logalpha0": 1.0},
            "scenario_mse_matrix": legacy_mse,
        },
        "bank120": {
            "per_checkpoint_rms": {str(step): 0.70 for step in PUBLICATION_STEPS},
            "per_publication_dimension_rms": [
                [0.70] * 6 for _ in PUBLICATION_STEPS
            ],
            "authority_cell_rms": {"tw0_logalpha0": 0.80},
            "scenario_mse_matrix": bank_mse,
        },
        "privileged_true_motor": {
            "per_checkpoint_rms": {str(step): 0.50 for step in PUBLICATION_STEPS},
            "per_publication_dimension_rms": [[0.5] * 6 for _ in PUBLICATION_STEPS],
            "authority_cell_rms": {"tw0_logalpha0": 0.5},
            "scenario_mse_matrix": legacy_mse,
        },
    }
    return evaluation, coverage


def test_preregistered_gate_accepts_inclusive_thresholds() -> None:
    evaluation, coverage = _contract()
    gate = _evaluate_pre_registered_gate(evaluation, coverage)
    # Numerical sub-gates pass at inclusive boundaries, and the oracle now
    # shares the exact production legacy feature contract.
    assert gate["passed"] is True
    assert gate["baseline_contract_match"] is True
    assert gate["baseline_contract_match_details"]["passed"] is True
    assert gate["motor_coverage"]["passed"] is True
    assert gate["ridge_effectiveness_t50"]["capability_dimensions_passed"] is True
    assert gate["ridge_effectiveness_t50"]["tau_dimensions_passed"] is True
    assert gate["dimension_degradation_t50"]["passed"] is True
    assert gate["paired_bootstrap_capability_mse"]["by_publication"]["50"]["passed"] is True


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("coverage", "best_candidate", "50", "ratio_p95"), 0.8001),
        (("bank120", "per_publication_dimension_rms", 1, 0), 0.751),
        (("bank120", "authority_cell_rms", "tw0_logalpha0"), 1.1001),
    ],
)
def test_preregistered_gate_rejects_each_strict_boundary(path: tuple, value: float) -> None:
    evaluation, coverage = _contract()
    target = coverage if path[0] == "coverage" else evaluation[path[0]]
    for key in path[1:-1]:
        target = target[key]
    target[path[-1]] = value
    gate = _evaluate_pre_registered_gate(evaluation, coverage)
    assert gate["passed"] is False


def test_preregistered_gate_reports_undefined_privileged_gap_as_failure() -> None:
    evaluation, coverage = _contract()
    evaluation["privileged_true_motor"]["per_checkpoint_rms"]["50"] = 1.0
    gate = _evaluate_pre_registered_gate(evaluation, coverage)
    gap = gate["gap_closure_relative_privileged"]
    assert gate["passed"] is False
    assert gap["defined"] is False
    assert gap["gap_closure_relative_privileged"] is None
