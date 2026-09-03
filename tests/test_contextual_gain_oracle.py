from __future__ import annotations

import torch

from structured_policy import effective_wrench_mixer
from tools.diagnose_contextual_gain_oracle import OracleData, compare_oracles


def test_contextual_oracle_uses_scenario_heldout_cells() -> None:
    torch.manual_seed(3)
    horizon, scenarios = 20, 64
    capability = torch.zeros(scenarios, 6)
    capability[:, 0] = torch.linspace(1.5, 5.4, scenarios)
    capability[:, 1] = torch.exp(torch.linspace(torch.log(torch.tensor(35.0)),
                                                torch.log(torch.tensor(2000.0)), scenarios))
    capability[:, 2] = 0.2
    capability[:, 3] = 1.7
    capability[:, 4] = 0.08
    capability[:, 5] = 0.12
    features = torch.randn(horizon, scenarios, 15)
    # A context-varying law makes the held-out bin oracle useful while every
    # bin still contains multiple train and test scenarios.
    multiplier = (1.0 + 0.5 * (capability[:, 0] > 3.4).float()).reshape(1, scenarios, 1)
    base = torch.randn(4, 15)
    wrench = multiplier * torch.einsum("oe,tbe->tbo", base, features)
    mixer = effective_wrench_mixer(capability)
    action_delta = torch.einsum("bij,tbj->tbi", torch.linalg.pinv(mixer), wrench)
    intercept = torch.zeros_like(action_delta)
    data = OracleData(
        features=features,
        wrench=wrench,
        teacher_action=action_delta.clamp(-1.0, 1.0),
        intercept_action=intercept,
        capability=capability,
        scenario_ids=torch.arange(scenarios),
    )
    report = compare_oracles(data, bins=4)
    assert int(report["contextual_covered_scenarios"]) >= 16
    assert torch.isfinite(torch.tensor(float(report["contextual_action_rms"])))
    assert float(report["contextual_over_global_action"]) < 1.0

