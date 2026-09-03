from __future__ import annotations

from pathlib import Path

import torch

from env_l2f import L2FParams, L2FSimulator
from structured_distillation import build_dagger_scenario_bank
from tools.diagnose_q2_equilibrium_bias import (
    build_report,
    measure_q2_equilibrium_bias,
)


class _BiasedTeacher:
    def initial_hidden(self, batch, *, device, dtype):
        return torch.zeros(batch, 4, device=device, dtype=dtype)

    def forward_with_aux(self, observation, hidden):
        trim = observation[:, 21:25]
        action = trim + 0.10 + hidden
        return action, hidden + 0.01, {}


def test_equilibrium_bias_advances_hidden_without_executing_teacher_action() -> None:
    simulator = L2FSimulator(L2FParams(dt=0.01))
    bank = build_dagger_scenario_bank(16, seed=5, dt=0.01, per_cell=1)
    rows = measure_q2_equilibrium_bias(
        _BiasedTeacher(), simulator, bank, steps=(1, 3)
    )
    assert [row["step"] for row in rows] == [1, 3]
    assert abs(rows[0]["action_minus_trim_rms"] - 0.10) < 1.0e-6
    assert abs(rows[1]["action_minus_trim_rms"] - 0.12) < 1.0e-6


def test_equilibrium_bias_report_records_registered_protocol() -> None:
    simulator = L2FSimulator(L2FParams(dt=0.01))
    bank = build_dagger_scenario_bank(16, seed=6, dt=0.01, per_cell=1)
    report = build_report(
        _BiasedTeacher(), simulator, bank,
        checkpoint=Path("teacher.pt"), seed=6, steps=(1,),
    )
    assert report["scenario_count"] == 16
    assert report["seed"] == 6
    assert report["rows"][0]["step"] == 1
    assert len(report["bank_sha256"]) == 64
