"""Goal conditioning preserves hover weights and admits learnable mission inputs."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib

import pytest
import torch

from diffphys.agile_checkpoint import import_hover_checkpoint
from diffphys.env_raptor import RaptorParams, environment_contract
from diffphys.response_policy import (
    AGILE_ARCHITECTURE,
    ARCHITECTURE,
    ResponseMotorPolicy,
    ResponsePolicyConfig,
)


def _digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        for metadata in (name, str(value.dtype), str(tuple(value.shape))):
            digest.update(metadata.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _checkpoint(policy):
    config = asdict(policy.config)
    if not config["goal_conditioning"]:
        # Original hover checkpoints predate this configuration field.
        config.pop("goal_conditioning")
    model = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    schema = "raptor-multi-airframe-gaussian-v6"
    return {
        "schema": schema,
        "architecture": policy.architecture,
        "policy_config": config,
        "model": model,
        "model_sha256": _digest(model),
        "binding": {
            "dtype": str(next(policy.parameters()).dtype).removeprefix("torch."),
            "protocol": {
                "version": schema,
                "architecture": policy.architecture,
                "policy": config.copy(),
                "environment_params": {"dt": policy.config.dt},
                "environment": environment_contract(RaptorParams(policy.config.dt)),
            },
        },
    }


def _observations(count, dtype):
    values = torch.randn(count, 22, dtype=dtype) * 0.2
    # A nontrivial valid rotation makes the world/body transformation observable.
    rotation = torch.tensor(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)), dtype=dtype)
    values[:, 6:15] = rotation.reshape(1, 9)
    return values


def _goals(count, dtype):
    values = torch.randn(count, 23, dtype=dtype) * 0.4
    values[:, 9:18] = torch.eye(3, dtype=dtype).reshape(1, 9)
    values[:, 21] = 0
    values[:, 22] = 3.0
    return values


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_import_preserves_translated_hover_response_and_hidden_state(tmp_path, dtype):
    torch.manual_seed(17)
    hover = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).to(dtype=dtype)
    source = tmp_path / "hover.pt"
    torch.save(_checkpoint(hover), source)
    agile = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8, goal_conditioning=True)).to(dtype=dtype)
    provenance = import_hover_checkpoint(agile, source)
    assert agile.architecture == AGILE_ARCHITECTURE
    assert hover.architecture == ARCHITECTURE
    assert len(list(agile.parameters())) == 6
    assert agile.response_memory.input_size == 36
    assert torch.count_nonzero(agile.response_memory.weight_ih[:, 16:]) == 0
    assert provenance["weights_only"] is True
    assert provenance["input_columns_added"] == 20
    old_state = new_state = None
    for _ in range(6):
        observation = _observations(3, dtype)
        goals = _goals(3, dtype)
        translated = observation.clone()
        translated[:, :3] -= goals[:, :3]
        old = hover(translated, old_state)
        new = agile(observation, new_state, goals=goals)
        tolerance = 2e-6 if dtype == torch.float32 else 1e-12
        torch.testing.assert_close(new.action, old.action, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(new.next_state.memory, old.next_state.memory,
                                   atol=tolerance, rtol=tolerance)
        old_state, new_state = old.next_state, new.next_state


def test_goal_features_keep_the_declared_world_body_packet_order():
    policy = ResponseMotorPolicy(ResponsePolicyConfig(goal_conditioning=True)).double()
    observation = torch.zeros(1, 22, dtype=torch.float64)
    observation[0, :3] = torch.tensor((1., 2., 3.))
    observation[0, 6:15] = torch.tensor((0., -1., 0., 1., 0., 0., 0., 0., 1.))
    goals = torch.zeros(1, 23, dtype=torch.float64)
    goals[0, :3] = torch.tensor((1., 1., 1.))
    goals[0, 3:6] = torch.tensor((4., 6., 8.))
    goals[0, 6:9] = torch.tensor((7., 10., 12.))
    goals[0, 9:18] = torch.eye(3).flatten()
    goals[0, 18:21] = torch.tensor((3., 6., 9.))
    goals[0, 21:] = torch.tensor((1., 2.5))
    features = policy.control_features(observation, goals=goals)[0]
    torch.testing.assert_close(features[:3], torch.tensor((1., 0., 2.), dtype=torch.float64))
    expected_extra = torch.tensor((4., -3., 5., 8., -6., 9.,
                                   -1., 1., 0., -1., -1., 0., 0., 0., 0.,
                                   2., -1., 3., 1., 2.5), dtype=torch.float64)
    torch.testing.assert_close(features[16:], expected_extra)


def test_fresh_zero_extra_columns_receive_nonzero_training_gradient():
    torch.manual_seed(3)
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8, goal_conditioning=True)).double()
    assert torch.count_nonzero(policy.response_memory.weight_ih[:, 16:]) == 0
    observation = _observations(4, torch.float64).requires_grad_()
    goals = _goals(4, torch.float64)
    output = policy(observation, goals=goals)
    (output.action - 0.2).square().sum().backward()
    assert policy.response_memory.weight_ih.grad[:, 16:].norm() > 0
    assert observation.grad[:, :3].norm() > 0
    assert torch.isfinite(observation.grad).all()


def test_goal_packet_is_required_only_for_conditioned_policy():
    observation = _observations(2, torch.float32)
    goals = _goals(2, torch.float32)
    hover = ResponseMotorPolicy()
    agile = ResponseMotorPolicy(ResponsePolicyConfig(goal_conditioning=True))
    # Memory initialization still takes only the measured 22-D observation.
    assert agile.initial_state(observation).memory.shape == (2, 64)
    with pytest.raises(ValueError, match="goals"):
        agile(observation)
    with pytest.raises(ValueError, match="goals"):
        hover(observation, goals=goals)
    with pytest.raises(ValueError, match="goals"):
        agile(observation, goals=goals[:, :22])
    with pytest.raises(ValueError, match="goals"):
        agile(observation, goals=goals.double())
    with pytest.raises(ValueError, match="goal_conditioning"):
        ResponsePolicyConfig(goal_conditioning=1)


@pytest.mark.parametrize("problem", [
    "raw_state", "schema", "architecture", "action", "dt", "hidden", "digest",
    "missing_key", "extra_key", "shape", "nonfinite",
])
def test_bad_import_is_rejected_before_destination_mutation(tmp_path, problem):
    torch.manual_seed(9)
    hover = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8))
    value = _checkpoint(hover)
    if problem == "raw_state":
        value = value["model"]
    elif problem == "schema":
        value["schema"] = "unknown-checkpoint"
    elif problem == "architecture":
        value["architecture"] = "old-feature-shortcut-policy"
    elif problem == "action":
        value["binding"]["protocol"]["environment"]["action_convention"] = "hover-residual"
    elif problem == "dt":
        value["binding"]["protocol"]["environment_params"]["dt"] = 0.02
    elif problem == "hidden":
        value["policy_config"]["memory_dim"] = 16
    elif problem == "digest":
        value["model"]["readout.bias"][0] += 0.1
    elif problem == "missing_key":
        del value["model"]["readout.bias"]
    elif problem == "extra_key":
        value["model"]["readout.weight_control"] = torch.zeros(4, 16)
    elif problem == "shape":
        value["model"]["response_memory.weight_ih"] = torch.zeros(24, 15)
    elif problem == "nonfinite":
        value["model"]["response_memory.weight_ih"][0, 0] = float("nan")
    source = tmp_path / "bad.pt"
    torch.save(value, source)
    agile = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8, goal_conditioning=True))
    before = copy.deepcopy(agile.state_dict())
    with pytest.raises(ValueError):
        import_hover_checkpoint(agile, source)
    for name, tensor in agile.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_compatible_agile_checkpoint_weights_only_import_keeps_learned_columns(tmp_path):
    source_policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8, goal_conditioning=True))
    with torch.no_grad():
        source_policy.response_memory.weight_ih[:, 16:].fill_(0.125)
    source = tmp_path / "agile.pt"
    torch.save(_checkpoint(source_policy), source)
    destination = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8, goal_conditioning=True))
    result = import_hover_checkpoint(destination, source)
    assert result["input_columns_added"] == 0
    for name, value in source_policy.state_dict().items():
        torch.testing.assert_close(destination.state_dict()[name], value, rtol=0, atol=0)


def test_finite_saved_weights_that_overflow_destination_dtype_are_atomic(tmp_path):
    source_policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).double()
    with torch.no_grad():
        source_policy.readout.bias[0] = 1e100
    source = tmp_path / "overflow.pt"
    torch.save(_checkpoint(source_policy), source)
    destination = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8, goal_conditioning=True))
    before = copy.deepcopy(destination.state_dict())
    with pytest.raises(ValueError, match="overflow"):
        import_hover_checkpoint(destination, source)
    for name, value in destination.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_goal_conditioned_recurrence_probe_matches_independent_group_vjps():
    from diffphys.response_grad_probe import GroupGradientProbe
    from diffphys.response_groups import GroupBalanceConfig, normalize_group_rows

    torch.manual_seed(47)
    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4, goal_conditioning=True)).double()
    reference = copy.deepcopy(policy)
    observations = [_observations(5, torch.float64) for _ in range(4)]
    goals = [_goals(5, torch.float64) for _ in range(4)]
    groups = torch.tensor(((0, 1, 5), (2, 3, 4)))
    counts = torch.tensor((2, 3))
    coefficients = torch.tensor(((0.5, 0.5, 0., 0., 0.),
                                 (0., 0., 1/3, 1/3, 1/3)), dtype=torch.float64)

    def costs(actor, probe=None):
        memory = actor.initial_state(observations[0])
        values = []
        for observation, packet in zip(observations, goals):
            if probe is not None:
                probe.begin_step(torch.arange(5))
            output = actor(observation, memory, goals=packet)
            memory = output.next_state
            values.append((output.action - 0.2).square().sum(-1))
        return torch.stack(values).sum(0)

    reference_costs = costs(reference)
    reference_rows = []
    for seed in coefficients:
        gradients = torch.autograd.grad(reference_costs, tuple(reference.parameters()),
                                        0.17 * seed, retain_graph=True)
        reference_rows.append(torch.cat([value.flatten() for value in gradients]))
    reference_rows = torch.stack(reference_rows)
    probe = GroupGradientProbe(policy, groups, counts)
    with probe.capture():
        actual_costs = costs(policy, probe)
    torch.testing.assert_close(actual_costs, reference_costs, rtol=0, atol=0)
    config = GroupBalanceConfig(clip_norm=0.05)
    actual, _ = probe.backward(actual_costs, coefficients, tuple(policy.parameters()),
                               config, gradient_scale=0.17)
    torch.testing.assert_close(probe.rows, reference_rows, rtol=2e-11, atol=2e-11)
    expected, _ = normalize_group_rows(reference_rows, config.gradient_epsilon,
                                       max_norm=config.clip_norm)
    torch.testing.assert_close(torch.cat([value.flatten() for value in actual]),
                               expected, rtol=2e-11, atol=2e-11)
    assert probe.views["response_memory.weight_ih"][:, :, 16:].norm() > 0
    assert not policy.response_memory._forward_hooks
    assert not policy.readout._forward_hooks
