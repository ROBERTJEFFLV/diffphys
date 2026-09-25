"""Training-only physical-group score normalization, with no extra backward."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from env_l2f import L2FState


GROUP_BALANCE_VERSION = "physical-group-score-rms-v1"
GROUP_FEATURE_NAMES = ("thrust_to_weight", "torque_to_inertia",
                       "motor_time_rising", "motor_time_falling")


@dataclass(frozen=True)
class GroupBalanceConfig:
    enabled: bool = False
    max_groups: int = 16
    min_scenarios: int = 32
    scale_mode: str = "rms"
    scale_floor: float = 1.0

    def __post_init__(self):
        if (not isinstance(self.max_groups, int) or isinstance(self.max_groups, bool)
                or not 1 <= self.max_groups <= 64
                or self.max_groups & (self.max_groups - 1)):
            raise ValueError("group-max-groups must be a power of two in [1,64]")
        if (not isinstance(self.min_scenarios, int) or isinstance(self.min_scenarios, bool)
                or self.min_scenarios < 32):
            raise ValueError("group-min-scenarios must be at least 32")
        if self.scale_mode not in ("none", "rms"):
            raise ValueError("group-scale-mode must be none or rms")
        if not math.isfinite(self.scale_floor) or self.scale_floor <= 0:
            raise ValueError("group-scale-floor must be finite and positive")

    @classmethod
    def from_args(cls, args):
        return cls(args.group_balance, args.group_max_groups, args.group_min_scenarios,
                   args.group_scale_mode, args.group_scale_floor)


def _group_physics(initial: L2FState) -> torch.Tensor:
    # These are initial, fixed physical parameters, never Actor observations.
    # Mean rotor times equal each rotor time in the current reference sampler.
    return torch.stack((initial.thrust_to_weight, initial.torque_to_inertia,
                        initial.motor_time_rising.mean(-1),
                        initial.motor_time_falling.mean(-1)), -1).detach()


@torch.no_grad()
def physics_group_layout(initial: L2FState, config: GroupBalanceConfig):
    """Batched balanced k-d splits in physical space, not trajectory difficulty.

    Columns: log(TWR), log(TTI), log(rising time), log(falling time), normalized
    by the log spans of the reference sampler. Split all current nodes in
    parallel on each node's widest feature, at its median rank. All groups have
    >= min_scenarios UNIQUE initial scenes. A constant node stops further
    splitting at this depth; identical aircraft are never arbitrarily split.
    Returned row table pads with index N; the actual flight order is unchanged.
    IDs are batch-relative, so no EMA statistics are shared across changing IDs.
    """
    raw = _group_physics(initial)
    n = initial.position.shape[0]
    if n < config.min_scenarios:
        raise ValueError(f"group balancing needs at least {config.min_scenarios} unique scenes")
    if raw.shape != (n, 4) or not bool((torch.isfinite(raw) & (raw > 0)).all()):
        raise ValueError("physical group features must be finite and positive")
    spans = raw.new_tensor((math.log(5.0/1.5), math.log(1200.0/40.0),
                            math.log(.10/.03), math.log(.30/.03)))
    features = torch.cat((raw.log()/spans, raw.new_zeros(1, 4)), 0)
    indices = torch.arange(n, device=raw.device).unsqueeze(0)
    counts = torch.full((1,), n, device=raw.device, dtype=torch.long)
    limit = min(config.max_groups, n // config.min_scenarios)
    while 2 * indices.shape[0] <= limit:
        width = indices.shape[1]
        valid = indices < n
        values = features[indices]
        spread = (values.masked_fill(~valid[..., None], -torch.inf).amax(1)
                  - values.masked_fill(~valid[..., None], torch.inf).amin(1))
        # At most log2(max_groups) small scalar checks, never one per scene.
        if not bool((spread.amax(-1) > 0).all()):
            break
        axis = spread.argmax(-1)
        key = values.gather(2, axis[:, None, None].expand(-1, width, 1)).squeeze(-1)
        order = key.masked_fill(~valid, torch.inf).argsort(dim=1, stable=True)
        ordered = indices.gather(1, order)
        left_count = counts // 2
        right_count = counts - left_count
        child_width = (width + 1) // 2
        columns = torch.arange(child_width, device=raw.device)[None, :]
        left = ordered[:, :child_width].masked_fill(columns >= left_count[:, None], n)
        right = ordered.gather(1, (columns + left_count[:, None]).clamp_max(width - 1))
        right = right.masked_fill(columns >= right_count[:, None], n)
        indices = torch.stack((left, right), 1).reshape(-1, child_width)
        counts = torch.stack((left_count, right_count), 1).reshape(-1)
    return indices, counts


@torch.no_grad()
def group_balanced_weights(costs, base_weights, initial, config: GroupBalanceConfig):
    """Detached group scores -> one vector of coefficients -> ONE task VJP.

    Original weights include 1/N and pooled CVaR. With a_i=N*w_i, use
        J_opt = (1/G) sum_g (1/n_g) sum_{i in g} a_i*C_i / stopgrad(s_g),
        s_g = max(RMS_{i in g}(C_i), scale_floor), or 1 for scale_mode=none.
    No group mean subtraction (which would cancel the objective), no gradient
    norm computation, no per-scene clipping, and no new autograd graph here.
    Equal counts and s_g=1 recover the original weights, including CVaR.
    Group statistics include early failures, not only survivors.
    """
    if not config.enabled:
        return base_weights, None
    if (costs.ndim != 1 or costs.shape != base_weights.shape or base_weights.requires_grad
            or costs.numel() != initial.position.shape[0]):
        raise ValueError("group weights require scalar scene costs and detached base weights")
    if not bool((torch.isfinite(costs) & torch.isfinite(base_weights)
                 & (costs >= 0) & (base_weights >= 0)).all()):
        raise FloatingPointError("nonfinite or negative group costs/weights")
    if not bool((base_weights > 0).any()):
        raise ValueError("group weights require some positive base weight")
    indices, counts = physics_group_layout(initial, config)
    n, groups = costs.numel(), counts.numel()
    valid = indices < n
    table = torch.cat((costs.detach(), costs.new_zeros(1)))[indices]
    count = counts.to(costs.dtype)
    # Stable FP32/FP64 RMS: do not square large unscaled costs.
    peak = table.amax(1)
    scaled = table / peak.clamp_min(torch.finfo(table.dtype).tiny)[:, None]
    rms = peak * (scaled.square().sum(1) / count).sqrt()
    scales = rms.clamp_min(config.scale_floor) if config.scale_mode == "rms" else torch.ones_like(rms)
    factors = ((n / (groups * count)) / scales)[:, None].expand_as(table)
    # Sorting the small integer membership table inverts it without atomic sums
    # or data-dependent GPU nonzero; padded N indices are last and ignored.
    inverse = indices.reshape(-1).argsort(stable=True)[:n]
    weights = base_weights.detach() * factors.reshape(-1)[inverse]
    if not bool(torch.isfinite(weights).all()):
        raise FloatingPointError("nonfinite normalized group weights")

    raw = _group_physics(initial)
    physics = torch.cat((raw, raw.new_zeros(1, 4)))[indices]
    low = physics.masked_fill(~valid[..., None], torch.inf).amin(1)
    high = physics.masked_fill(~valid[..., None], -torch.inf).amax(1)
    risk = torch.cat((n * base_weights.detach(), costs.new_zeros(1)))[indices]
    # Diagnostic values stay on-device; the trainer transfers this small table
    # once when logging after the update. No loop over scenes or group VJPs.
    columns = ["count", "cost_rms", "scale", "risk_multiplier_mean"]
    columns += [key + suffix for key in GROUP_FEATURE_NAMES for suffix in ("_min", "_max")]
    values = torch.cat((torch.stack((count, rms, scales, risk.sum(1)/count), 1),
                        torch.stack((low, high), -1).flatten(1)), 1)
    report = {"version": GROUP_BALANCE_VERSION, "group_count": groups,
              "minimum_scenarios": config.min_scenarios, "scale_mode": config.scale_mode,
              "columns": columns, "values": values}
    return weights.detach(), report

