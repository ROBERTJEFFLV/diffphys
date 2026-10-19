"""A transferred Actor is fine-tuned and serialized as one independent model.

These small deterministic checks establish checkpoint and inference behavior;
the synthetic source weights do not establish learned hover or flight ability.
"""
from __future__ import annotations

import io
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from diffphys.agile_checkpoint import import_hover_checkpoint
from diffphys.response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from diffphys.response_task import TaskLossConfig
from diffphys.response_training import _checkpoint, binding
from tools.train_response_control import parse_args


def _run_binding(policy):
    """Use the real command/config binding required by the inference loader."""
    agile = policy.config.goal_conditioning
    args = parse_args([
        "--task", "agile" if agile else "hover", "--device", "cpu",
        "--dtype", str(next(policy.parameters()).dtype).removeprefix("torch."),
        "--memory-dim", str(policy.config.memory_dim),
        *([] if agile else ["--epsilon-p", ".01", "--epsilon-a", ".01",
                           "--lambda-R", "25"]),
    ])
    config = getattr(args, "agile_training_config", None)
    loss = config.loss if agile else TaskLossConfig(epsilon_p=.01, epsilon_a=.01,
                                                   lambda_R=25.)
    return binding(args, policy.config, loss, agile_config=config)


def _commands(dtype):
    generator = torch.Generator().manual_seed(901)
    observations = torch.randn(5, 3, 22, generator=generator, dtype=dtype) * .2
    observations[:, :, 6:15] = torch.eye(3, dtype=dtype).flatten()
    goals = torch.randn(5, 3, 23, generator=generator, dtype=dtype) * .3
    goals[:, :, 9:18] = torch.eye(3, dtype=dtype).flatten()
    goals[:, :, 21] = torch.arange(5, dtype=dtype)[:, None] / 4
    goals[:, :, 22] = 1. - torch.arange(5, dtype=dtype)[:, None] * .01
    return observations, goals


def _sequence(policy, observations, goals):
    state = None
    actions = []
    for observation, goal in zip(observations, goals):
        output = policy(observation, state, goals=goal)
        actions.append(output.action)
        state = output.next_state
    return torch.stack(actions), state.memory


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_finetuned_checkpoint_loads_without_source_in_a_fresh_process(tmp_path, dtype):
    torch.manual_seed(812)
    hover = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).to(dtype=dtype)
    source_path = tmp_path / "original_hover.pt"
    torch.save(_checkpoint(hover, torch.optim.Adam(hover.parameters()),
                           {"updates": 0}, _run_binding(hover)), source_path)

    policy = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8,
                                                      goal_conditioning=True)).to(dtype=dtype)
    provenance = import_hover_checkpoint(policy, source_path)
    original = {name: value.detach().clone() for name, value in hover.state_dict().items()}
    for name, value in policy.state_dict().items():
        if name == "response_memory.weight_ih":
            torch.testing.assert_close(value[:, :16], original[name], rtol=0, atol=0)
            assert torch.count_nonzero(value[:, 16:]) == 0
        else:
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    # A multi-step differentiable objective exercises both the imported GRU
    # recurrence and the new goal columns. It tests trainability, not flight.
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    observations, goals = _commands(dtype)
    actions, _ = _sequence(policy, observations, goals)
    (actions - .15).square().mean().backward()
    for name, parameter in policy.named_parameters():
        assert parameter.requires_grad, name
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name
    assert torch.count_nonzero(policy.response_memory.weight_ih.grad[:, :16]) > 0
    assert torch.count_nonzero(policy.response_memory.weight_ih.grad[:, 16:]) > 0
    optimizer.step()
    for name, value in policy.state_dict().items():
        copied = value[:, :16] if name == "response_memory.weight_ih" else value
        assert not torch.equal(copied, original[name]), name
    assert torch.count_nonzero(policy.response_memory.weight_ih[:, 16:]) > 0

    output_path = tmp_path / "agile_finetuned.pt"
    torch.save(_checkpoint(policy, optimizer,
                           {"updates": 1, "initialization": provenance},
                           _run_binding(policy)), output_path)
    source_path.unlink()
    assert not source_path.exists()

    # Only the output checkpoint path is passed to a new interpreter. Measured
    # state and task commands enter through stdin; no original Actor survives
    # in that process and no checkpoint search or base filename is available.
    payload = io.BytesIO()
    torch.save({"observations": observations, "goals": goals}, payload)
    program = """
import io
import sys
import torch
from diffphys.response_training import load_policy_checkpoint

torch.set_num_threads(1)
policy, saved = load_policy_checkpoint(sys.argv[1], "cpu", getattr(torch, sys.argv[2]))
policy.eval()
inputs = torch.load(io.BytesIO(sys.stdin.buffer.read()), weights_only=True)
state = None
actions = []
with torch.no_grad():
    for observation, goals in zip(inputs["observations"], inputs["goals"]):
        output = policy(observation, state, goals=goals)
        actions.append(output.action)
        state = output.next_state
torch.save({"model": policy.state_dict(), "actions": torch.stack(actions),
            "memory": state.memory, "model_sha256": saved["model_sha256"]},
           sys.stdout.buffer)
"""
    completed = subprocess.run([
        sys.executable, "-c", program, str(output_path),
        str(dtype).removeprefix("torch."),
    ], cwd=Path(__file__).resolve().parents[1], input=payload.getvalue(),
       capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    loaded = torch.load(io.BytesIO(completed.stdout), weights_only=True)
    saved = torch.load(output_path, weights_only=True)
    assert loaded["model_sha256"] == saved["model_sha256"]
    assert set(loaded["model"]) == set(original)
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(loaded["model"][name], value, rtol=0, atol=0)
    with torch.no_grad():
        expected_actions, expected_memory = _sequence(policy, observations, goals)
    torch.testing.assert_close(loaded["actions"], expected_actions, rtol=0, atol=0)
    torch.testing.assert_close(loaded["memory"], expected_memory, rtol=0, atol=0)
