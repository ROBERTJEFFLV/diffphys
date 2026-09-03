from __future__ import annotations

import torch

from structured_policy import StructuredPolicyConfig, StructuredRecurrentPolicy
from tools.calibrate_structured_capability import _policy_config_hash


def test_deployment_hash_includes_q_validity_and_sample_count() -> None:
    policy = StructuredRecurrentPolicy(
        StructuredPolicyConfig(hidden_dim=4, identifier_dim=4)
    )
    initial = _policy_config_hash(policy)
    policy.capability_calibration_n.fill_(512)
    assert _policy_config_hash(policy) != initial
    with_count = _policy_config_hash(policy)
    policy.capability_conformal_q[0] = 2.0
    q_changed = _policy_config_hash(policy)
    assert q_changed != with_count
    policy.capability_calibration_valid.fill_(True)
    assert _policy_config_hash(policy) != q_changed


def test_base_hash_excludes_all_mutable_calibration_buffers() -> None:
    policy = StructuredRecurrentPolicy(
        StructuredPolicyConfig(hidden_dim=4, identifier_dim=4)
    )
    initial = _policy_config_hash(policy, include_q=False)
    policy.capability_conformal_q.fill_(3.0)
    assert _policy_config_hash(policy, include_q=False) == initial
    policy.capability_calibration_valid.fill_(True)
    policy.capability_calibration_n.fill_(512)
    assert _policy_config_hash(policy, include_q=False) == initial
