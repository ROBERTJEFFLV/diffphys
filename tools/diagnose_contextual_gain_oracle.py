"""Test whether slow context can explain Q2's fast feedback law.

This is a falsification diagnostic, not a trainer.  It removes either the
analytic physical trim (the formal architecture) or a same-latent behavioural
intercept (a diagnostic only) from frozen Q2, maps the remaining action to the
physical-fit specific-wrench coordinates, and compares three regressors:

* one global gain;
* one gain per (thrust-to-weight, log roll-authority) stratum;
* one gain per scenario (an optimistic lower bound).

The stratum gain is always evaluated on scenarios that were not used to fit
that stratum.  A contextual controller is worth implementing only if this
held-out oracle materially improves on the global gain.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.formal_rollout import load_q2_policy  # noqa: E402
from env_l2f import L2FParams, L2FSimulator  # noqa: E402
from equilibrium_control import (  # noqa: E402
    analytic_equilibrium_target,
    body_z_error_body_frame,
)
from policy_observation import (  # noqa: E402
    build_policy_observation,
    initial_observation_state,
    update_position_integral,
)
from structured_policy import effective_wrench_mixer  # noqa: E402


DEFAULT_Q2 = (
    ROOT
    / "reports/q_residual_h500_u2000_gpu2/seed_7/group_Q2/checkpoints/model_update_2000.pt"
)


@dataclass(frozen=True)
class OracleData:
    features: torch.Tensor
    wrench: torch.Tensor
    teacher_action: torch.Tensor
    intercept_action: torch.Tensor
    capability: torch.Tensor
    scenario_ids: torch.Tensor


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _raw_capability(state) -> torch.Tensor:
    return torch.stack(
        (
            state.thrust_to_weight,
            state.alpha_roll_max,
            state.eta_yaw,
            state.jz_over_jxy,
            state.motor_time_rising,
            state.motor_time_falling,
        ),
        dim=-1,
    )


@torch.no_grad()
def collect_oracle_data(
    teacher,
    simulator: L2FSimulator,
    state,
    *,
    horizon: int,
    intercept_kind: str = "analytic-trim",
) -> OracleData:
    if intercept_kind not in ("analytic-trim", "same-latent"):
        raise ValueError("intercept_kind must be analytic-trim or same-latent")
    batch = state.position.shape[0]
    hidden = teacher.initial_hidden(
        batch, device=state.position.device, dtype=state.position.dtype
    )
    observation_state = initial_observation_state(
        batch, device=state.position.device, dtype=state.position.dtype
    )
    target = analytic_equilibrium_target(state, gravity=simulator.params.gravity)
    capability = _raw_capability(state)
    mixer = effective_wrench_mixer(capability)
    scales = state.position.new_tensor(
        (.1, .1, .1, .1, .1, .1, .1, .1, .5, .5, .5, .1, .1, .1, .1)
    )
    features: List[torch.Tensor] = []
    wrenches: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    intercepts: List[torch.Tensor] = []
    for _ in range(horizon):
        observation, observed_position = build_policy_observation(
            state,
            observation_state,
            mode="integral25",
            noise_max=0.0,
            integral_input_frame="body",
            integral_input_multiplier=1.0,
        )
        action, hidden, _ = teacher.forward_with_aux(observation, hidden)
        # Keep exactly the hidden state used by the deployed action head, while
        # zeroing the direct fast-error inputs of the integral/damping heads.
        # The teacher has no explicit equilibrium variable; analytic trim is
        # therefore the least ambiguous previous-action counterfactual.
        equilibrium_observation = observation.clone()
        equilibrium_observation[:, 15:18] = 0.0
        equilibrium_observation[:, 21:25] = target.motor_trim
        latent = F.leaky_relu(hidden, negative_slope=teacher.negative_slope)
        same_latent_intercept, _ = teacher._action_from_latent(
            equilibrium_observation, latent
        )
        # Only the simulator-derived trim has equilibrium semantics.  The
        # same-latent value still contains p/v/R information already encoded
        # by Q2's encoder/GRU and is retained solely to reproduce the earlier
        # behavioural diagnostic.
        intercept = (
            target.motor_trim
            if intercept_kind == "analytic-trim"
            else same_latent_intercept
        )
        tilt = body_z_error_body_frame(state.rotation, target.body_z)[:, :2]
        raw_error = torch.cat(
            (
                state.position,
                state.velocity,
                tilt,
                state.omega,
                state.motor - target.motor_trim,
            ),
            dim=-1,
        )
        feature = torch.tanh(raw_error / scales)
        wrench = torch.bmm(
            mixer, (action - intercept).unsqueeze(-1)
        ).squeeze(-1)
        features.append(feature)
        wrenches.append(wrench)
        actions.append(action)
        intercepts.append(intercept)
        observation_state = update_position_integral(
            observation_state,
            observed_position,
            dt=simulator.params.dt,
            integral_limit=0.5,
            integral_leak=0.0,
        )
        state = simulator.step(state, action, grad_decay=1.0)
    scenario_ids = torch.arange(batch, device=state.position.device)
    return OracleData(
        features=torch.stack(features),
        wrench=torch.stack(wrenches),
        teacher_action=torch.stack(actions),
        intercept_action=torch.stack(intercepts),
        capability=capability,
        scenario_ids=scenario_ids,
    )


def _ridge(features: torch.Tensor, targets: torch.Tensor, ridge: float) -> torch.Tensor:
    x = features.reshape(-1, features.shape[-1]).double()
    y = targets.reshape(-1, targets.shape[-1]).double()
    identity = torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
    return torch.linalg.solve(x.T @ x + float(ridge) * identity, x.T @ y).T.to(features)


def _quantile_bins(value: torch.Tensor, count: int) -> torch.Tensor:
    value = value.contiguous()
    edges = torch.quantile(
        value,
        torch.linspace(0.0, 1.0, count + 1, device=value.device, dtype=value.dtype),
    )
    # bucketize uses the inner edges; clamp protects repeated quantiles.
    return torch.bucketize(value, edges[1:-1].contiguous()).clamp(0, count - 1)


def _normalized_rms(prediction: torch.Tensor, target: torch.Tensor,
                    scale: torch.Tensor) -> float:
    return float(torch.sqrt(((prediction - target) / scale).square().mean()))


def _action_rms(
    predicted_wrench: torch.Tensor,
    data: OracleData,
    time_slice: slice,
    scenario_mask: torch.Tensor,
) -> float:
    mixer = effective_wrench_mixer(data.capability[scenario_mask])
    inverse = torch.linalg.pinv(mixer)
    delta = torch.einsum(
        "bij,tbj->tbi", inverse, predicted_wrench
    )
    predicted = (data.intercept_action[time_slice, scenario_mask] + delta).clamp(-1.0, 1.0)
    target = data.teacher_action[time_slice, scenario_mask]
    return float(torch.sqrt((predicted - target).square().mean()))


def compare_oracles(data: OracleData, *, bins: int = 4, cadence: int = 25,
                    ridge: float = 1.0e-3) -> Dict[str, object]:
    if bins < 2:
        raise ValueError("bins must be at least two")
    horizon, scenarios = data.features.shape[:2]
    if scenarios < 2 * bins * bins:
        raise ValueError("need at least two scenarios per contextual stratum on average")
    if cadence < 1:
        raise ValueError("cadence must be positive")
    all_time = slice(0, horizon)
    time_split = max(1, int(0.70 * horizon))
    train_time = slice(0, time_split)
    test_time = slice(time_split, horizon)
    # Alternating ranks inside each context cell give every nonempty cell a
    # deterministic scenario-held-out split whenever it contains >=2 members.
    tw_bin = _quantile_bins(data.capability[:, 0], bins)
    alpha_bin = _quantile_bins(data.capability[:, 1].log(), bins)
    cell = tw_bin * bins + alpha_bin
    train_scenario = torch.zeros(scenarios, dtype=torch.bool, device=cell.device)
    for value in range(bins * bins):
        members = torch.nonzero(cell == value).flatten()
        train_scenario[members[::2]] = True
    test_scenario = ~train_scenario
    if not bool(test_scenario.any()):
        raise RuntimeError("context split produced no held-out scenarios")

    training_wrench = data.wrench[all_time, train_scenario]
    scale = torch.quantile(training_wrench.abs().reshape(-1, 4), 0.90, dim=0).clamp_min(1.0e-3)
    global_gain = _ridge(
        data.features[all_time, train_scenario], training_wrench / scale, ridge
    ) * scale[:, None]
    global_prediction = torch.einsum(
        "oe,tbe->tbo", global_gain, data.features[all_time, test_scenario]
    )
    heldout_target = data.wrench[all_time, test_scenario]

    contextual_prediction = torch.zeros_like(heldout_target)
    contextual_covered = torch.zeros(
        test_scenario.sum(), dtype=torch.bool, device=cell.device
    )
    heldout_ids = torch.nonzero(test_scenario).flatten()
    for value in range(bins * bins):
        train_members = train_scenario & (cell == value)
        test_members = heldout_ids[cell[heldout_ids] == value]
        if not bool(train_members.any()) or test_members.numel() == 0:
            continue
        gain = _ridge(
            data.features[all_time, train_members],
            data.wrench[all_time, train_members] / scale,
            ridge,
        ) * scale[:, None]
        locations = torch.nonzero(
            (heldout_ids[:, None] == test_members[None, :]).any(dim=1)
        ).flatten()
        contextual_prediction[:, locations] = torch.einsum(
            "oe,tbe->tbo", gain, data.features[all_time, test_members]
        )
        contextual_covered[locations] = True
    if not bool(contextual_covered.any()):
        raise RuntimeError("no contextual stratum had both train and held-out scenarios")

    # A cadence-conditioned oracle tests whether one held physical gain per
    # 25-step slow-context interval explains the teacher better than a single
    # episode-wide gain.  It uses the same scenario-held-out split.
    piecewise_prediction = torch.zeros_like(heldout_target)
    for start in range(0, horizon, cadence):
        interval = slice(start, min(horizon, start + cadence))
        for value in range(bins * bins):
            train_members = train_scenario & (cell == value)
            test_members = heldout_ids[cell[heldout_ids] == value]
            if not bool(train_members.any()) or test_members.numel() == 0:
                continue
            gain = _ridge(
                data.features[interval, train_members],
                data.wrench[interval, train_members] / scale,
                ridge,
            ) * scale[:, None]
            locations = torch.nonzero(
                (heldout_ids[:, None] == test_members[None, :]).any(dim=1)
            ).flatten()
            piecewise_prediction[interval, locations] = torch.einsum(
                "oe,tbe->tbo", gain, data.features[interval, test_members]
            )

    # Per-scenario temporal holdout is intentionally optimistic.  It answers
    # whether an affine fast law can represent each scenario at all.
    per_scenario_prediction = torch.zeros_like(data.wrench[test_time])
    for scenario in range(scenarios):
        gain = _ridge(
            data.features[train_time, scenario:scenario + 1],
            data.wrench[train_time, scenario:scenario + 1] / scale,
            ridge,
        ) * scale[:, None]
        per_scenario_prediction[:, scenario] = torch.einsum(
            "oe,te->to", gain, data.features[test_time, scenario]
        )

    contextual_target = heldout_target[:, contextual_covered]
    contextual_value = contextual_prediction[:, contextual_covered]
    piecewise_value = piecewise_prediction[:, contextual_covered]
    global_for_context = global_prediction[:, contextual_covered]
    global_error = _normalized_rms(global_for_context, contextual_target, scale)
    contextual_error = _normalized_rms(contextual_value, contextual_target, scale)
    piecewise_error = _normalized_rms(piecewise_value, contextual_target, scale)
    per_scenario_error = _normalized_rms(
        per_scenario_prediction, data.wrench[test_time], scale
    )
    contextual_mask = torch.zeros_like(test_scenario)
    contextual_mask[heldout_ids[contextual_covered]] = True
    contextual_action_error = _action_rms(
        contextual_value, data, all_time, contextual_mask
    )
    global_action_error = _action_rms(
        global_for_context, data, all_time, contextual_mask
    )
    piecewise_action_error = _action_rms(
        piecewise_value, data, all_time, contextual_mask
    )
    return {
        "scenario_count": scenarios,
        "horizon": horizon,
        "train_scenarios": int(train_scenario.sum()),
        "heldout_scenarios": int(test_scenario.sum()),
        "contextual_covered_scenarios": int(contextual_covered.sum()),
        "wrench_p90_scale": [float(value) for value in scale],
        "global_normalized_wrench_rms": global_error,
        "contextual_normalized_wrench_rms": contextual_error,
        "piecewise_contextual_normalized_wrench_rms": piecewise_error,
        "per_scenario_temporal_normalized_wrench_rms": per_scenario_error,
        "contextual_over_global": contextual_error / max(global_error, 1.0e-12),
        "contextual_action_rms": contextual_action_error,
        "piecewise_contextual_action_rms": piecewise_action_error,
        "global_action_rms_on_contextual_subset": global_action_error,
        "contextual_over_global_action": (
            contextual_action_error / max(global_action_error, 1.0e-12)
        ),
        "contextual_halves_global_error": contextual_error <= 0.5 * global_error,
        "contextual_halves_global_action_error": (
            contextual_action_error <= 0.5 * global_action_error
        ),
        "piecewise_halves_contextual_wrench_error": (
            piecewise_error <= 0.5 * contextual_error
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Falsify or support contextual fast-gain scheduling")
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_Q2)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/contextual_gain_oracle.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenarios", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--cadence", type=int, default=25)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument(
        "--intercept-kind", choices=("analytic-trim", "same-latent"),
        default="analytic-trim",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scenarios < 2 * args.bins * args.bins or args.horizon < 4:
        raise ValueError("scenario/horizon budget is too small for held-out oracle comparison")
    device = _device(args.device)
    torch.manual_seed(args.seed)
    teacher, checkpoint_args = load_q2_policy(
        args.source_checkpoint, device=device, dtype=torch.float32
    )
    simulator = L2FSimulator(L2FParams(dt=float(checkpoint_args.get("dt", 0.01))))
    state = simulator.reset(
        args.scenarios,
        device=device,
        dtype=torch.float32,
        sample_dynamics=True,
        sampled_dynamics_level="broad",
        broad_sampler="physical-fit",
        balanced_dynamics_sampling=args.scenarios % 256 == 0,
        sample_external_force=True,
    )
    data = collect_oracle_data(
        teacher, simulator, state, horizon=args.horizon,
        intercept_kind=args.intercept_kind,
    )
    report = compare_oracles(
        data, bins=args.bins, cadence=args.cadence, ridge=args.ridge
    )
    report.update(
        {
            "source_checkpoint": str(args.source_checkpoint.resolve()),
            "device": str(device),
            "seed": args.seed,
            "bins": args.bins,
            "cadence": args.cadence,
            "intercept_kind": args.intercept_kind,
            "intercept": (
                "simulator analytic physical equilibrium motor trim"
                if args.intercept_kind == "analytic-trim"
                else (
                    "same-latent Q2 head intercept; diagnostic only, still "
                    "contains fast-state information and is not equilibrium trim"
                )
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
