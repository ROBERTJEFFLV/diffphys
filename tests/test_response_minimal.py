from __future__ import annotations

import copy
import hashlib
import json

import pytest

from response_policy import ResponsePolicyConfig
from response_task import TaskLossConfig
from response_training import source_hash, validate_training_contract
from tools.check_response_preflight import assess


def progress_fixture():
    baseline = {"score": 10., "finite": True, "position_rms": 2., "velocity_rms": 2.,
                "omega_rms": 2., "steady_success_rate": 0., "motor_saturation_fraction": .1}
    improved = dict(baseline, score=8., position_rms=1., omega_rms=1.)
    row = {"task_loss": 8., "gradient_norm": 3., "numerics_finite": True,
           "position_rms": 1., "velocity_rms": 1., "omega_rms": 1.,
           "steady_success_rate": 0., "motor_saturation_fraction": .1}
    return {"updates": 150, "status": "update_budget", "history": [row.copy() for _ in range(150)],
            "development": [baseline, improved.copy(), improved.copy(), improved.copy()]}


def test_preflight_allows_initial_learning_without_deployment_success():
    report = assess(progress_fixture())
    assert report["passed"]
    assert report["recent_development"]["steady_success_rate"] == 0


def test_preflight_rejects_bad_numerics_no_improvement_and_pegged_motors():
    base = progress_fixture()
    for kind in ("nonfinite", "no_position_progress", "saturated"):
        value = copy.deepcopy(base)
        if kind == "nonfinite":
            value["history"][-1]["numerics_finite"] = False
        elif kind == "no_position_progress":
            for row in value["development"][1:]:
                row["position_rms"] = 3.
        else:
            for row in value["history"][-25:]:
                row["motor_saturation_fraction"] = 1.
        assert not assess(value)["passed"]


def test_contract_cannot_pass_with_empty_checks(tmp_path):
    from dataclasses import asdict
    record = {"source_sha256": source_hash(), "policy_config": asdict(ResponsePolicyConfig()),
              "loss_config": asdict(TaskLossConfig()), "checks": {}, "passed": True,
              "gradient_groups": {"response_recurrent": {"finite": True, "norm": 1.},
                                  "control_head": {"finite": True, "norm": 1.}}}
    record["evidence_sha256"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="failed"):
        validate_training_contract(path, ResponsePolicyConfig(), TaskLossConfig())


def test_insufficient_preflight_is_not_promoted():
    value = progress_fixture()
    value["updates"] = 99
    assert not assess(value)["passed"]
