from __future__ import annotations

from tools.evaluate_structured_fixed_policy_risk import build_scenario_bank, paired_metrics


def _rows(bank, offset: float):
    rows = []
    for scenario in bank:
        value = float(scenario["scenario_id"]) + offset
        rows.append(
            {
                "scenario_id": scenario["scenario_id"],
                "authority": scenario["authority"],
                "authority_stratum": scenario["authority_stratum"],
                "terminal_energy": value,
                "terminal_position": value,
                "terminal_velocity": value,
                "terminal_omega": value,
                "max_position": value,
                "max_velocity": value,
                "max_omega": value,
                "action_rms": value,
                "wrench_residual_rms": value,
            }
        )
    return rows


def test_bank_has_at_least_64_and_balanced_authority_strata() -> None:
    bank = build_scenario_bank(64, seed=19)
    counts = {name: sum(item["authority_stratum"] == name for item in bank) for name in ("low", "mid", "high")}
    assert len(bank) == 64
    assert min(counts.values()) >= 20


def test_paired_metrics_use_eight_tail_samples_globally_and_per_group() -> None:
    bank = build_scenario_bank(64, seed=23)
    paired, groups, summary = paired_metrics(_rows(bank, 0.0), _rows(bank, 1.0))
    assert len(paired) == 64
    assert summary["effective_tail_count"] >= 8
    assert all(int(row["effective_tail_count"]) >= 8 for row in groups)
    assert all(abs(float(row["delta_mean_terminal_energy"]) - 1.0) < 1.0e-6 for row in groups)
