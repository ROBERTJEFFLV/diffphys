"""Training-only physical-cell gradients, capped (never amplified) before averaging."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from env_raptor import RaptorState
from response_sampling import CELL_COUNT, physics_cell_ids, yaw_authority


GROUP_BALANCE_VERSION = "physical-group-fixed-cap-v2"
GROUP_FEATURE_NAMES = ("thrust_to_weight", "torque_to_inertia",
                       "motor_time_rising", "motor_time_falling")


@dataclass(frozen=True)
class GroupBalanceConfig:
    max_groups: int = 16
    min_scenarios: int = 32
    gradient_epsilon: float = 1e-12
    vjp_chunk_size: int = 16
    layout: str = "adaptive"  # Legacy random-pool partition; production uses coverage128.
    clip_norm: float = 1.0  # Provisional fixed cap AFTER gradient_scale/CVaR, BEFORE averaging.

    def __post_init__(self):
        if self.layout not in ("adaptive", "coverage128"):
            raise ValueError("group layout must be adaptive or coverage128")
        if self.layout == "coverage128" and self.max_groups != CELL_COUNT:
            raise ValueError("coverage128 requires --group-max-groups 128, not adaptive regrouping")
        if not math.isfinite(self.clip_norm) or self.clip_norm <= 0:
            raise ValueError("group-clip-norm must be finite and positive")
        if (not isinstance(self.max_groups, int) or isinstance(self.max_groups, bool)
                or not 1 <= self.max_groups <= 128
                or self.max_groups & (self.max_groups - 1)):
            raise ValueError("group-max-groups must be a power of two in [1,128]")
        if (not isinstance(self.min_scenarios, int) or isinstance(self.min_scenarios, bool)
                or self.min_scenarios < (32 if self.layout == "adaptive" else 1)):
            raise ValueError("group-min-scenarios must be at least 32 for adaptive, 1 for coverage128")
        if not math.isfinite(self.gradient_epsilon) or self.gradient_epsilon <= 0:
            raise ValueError("group-gradient-epsilon must be finite and positive")
        if (not isinstance(self.vjp_chunk_size, int) or isinstance(self.vjp_chunk_size, bool)
                or not 1 <= self.vjp_chunk_size <= 64):
            raise ValueError("group-vjp-chunk-size must be an integer in [1,64]")


    @classmethod
    def from_args(cls, args):
        return cls(args.group_max_groups, args.group_min_scenarios,
                   args.group_gradient_epsilon, args.group_vjp_chunk_size,
                   "coverage128" if getattr(args, "train_sampling", "random") == "coverage128" else "adaptive",
                   getattr(args, "group_clip_norm", 1.0))


def _group_physics(initial: RaptorState) -> torch.Tensor:
    # These are initial, fixed physical parameters, never Actor observations.
    # Mean rotor times equal each rotor time in the current reference sampler.
    return torch.stack((initial.thrust_to_weight, initial.torque_to_inertia,
                        initial.motor_time_rising.mean(-1),
                        initial.motor_time_falling.mean(-1)), -1).detach()


@torch.no_grad()
def physics_group_layout(initial: RaptorState, config: GroupBalanceConfig):
    """Fixed coverage128 IDs, or legacy adaptive k-d splits for random pools.

    Columns: log(TWR), log(TTI), log(rising time), log(falling time), normalized
    by the log spans of the reference sampler. Split all current nodes in
    parallel on each node's widest feature, at its median rank. All groups have
    >= min_scenarios UNIQUE initial scenes. A constant node stops further
    splitting at this depth; identical aircraft are never arbitrarily split.
    Returned row table pads with index N; the actual flight order is unchanged.
    IDs are batch-relative, so no EMA statistics are shared across changing IDs.
    """
    n = initial.position.shape[0]
    if config.layout == "coverage128":
        # Reuse the SAME frozen TTI/rise/fall/yaw IDs as the sampler. Initial
        # membership survives termination; never regroup by live survivors.
        ids = physics_cell_ids(initial)
        counts = torch.bincount(ids, minlength=CELL_COUNT)
        if (n % CELL_COUNT or n // CELL_COUNT < config.min_scenarios
                or not bool((counts == n // CELL_COUNT).all())):
            raise ValueError("coverage128 gradient groups require equal nonempty quotas in all 128 cells")
        return ids.argsort(stable=True).reshape(CELL_COUNT, n // CELL_COUNT), counts
    raw = _group_physics(initial)
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
def group_gradient_coefficients(base_weights, initial, config: GroupBalanceConfig):
    """Build G vector-Jacobian seeds, never N per-scene seeds or cost scales.

    Row g defines L_g = (N/n_g) sum_{i in g} w_i C_i, with the original
    detached pooled CVaR weights w_i. Equal group sizes recover the original
    objective when group gradients are averaged without normalization.
    All initial scenes, including early failures, belong to exactly one row.
    """
    n = initial.position.shape[0]
    if (base_weights.shape != (n,) or base_weights.requires_grad
            or base_weights.device != initial.position.device):
        raise ValueError("group gradients need detached per-scene weights on the state device")
    if not bool((torch.isfinite(base_weights) & (base_weights >= 0)).all()):
        raise FloatingPointError("nonfinite or negative group risk weights")
    if not bool((base_weights > 0).any()):
        raise ValueError("group risk weights must include a positive value")
    indices, counts = physics_group_layout(initial, config)
    groups, width = indices.shape
    valid = indices < n
    inverse = indices.reshape(-1).argsort(stable=True)[:n]
    group_ids = torch.arange(groups, device=indices.device).repeat_interleave(width)[inverse]
    membership = group_ids[None, :] == torch.arange(groups, device=indices.device)[:, None]
    coefficients = membership.to(base_weights.dtype) * base_weights[None, :]
    coefficients = coefficients * (n / counts.to(base_weights.dtype))[:, None]
    if not bool(torch.isfinite(coefficients).all()):
        raise FloatingPointError("nonfinite group gradient coefficients")
    raw = _group_physics(initial)
    physics = torch.cat((raw, raw.new_zeros(1, 4)))[indices]
    low = physics.masked_fill(~valid[..., None], torch.inf).amin(1)
    high = physics.masked_fill(~valid[..., None], -torch.inf).amax(1)
    columns = ["count"]
    columns += [key + suffix for key in GROUP_FEATURE_NAMES for suffix in ("_min", "_max")]
    values = torch.cat((counts.to(raw.dtype)[:, None],
                        torch.stack((low, high), -1).flatten(1)), 1)
    report = {"version": GROUP_BALANCE_VERSION, "group_count": groups,
              "minimum_scenarios": config.min_scenarios, "columns": columns, "values": values,
              "layout": config.layout, "group_ids": list(range(groups)),
              "scene_group_ids": group_ids.detach()}
    if config.layout == "coverage128":
        authority = yaw_authority(initial)[indices]
        report["yaw_authority_min_max"] = torch.stack((authority.amin(1), authority.amax(1)), 1)
    return coefficients.detach(), report


@torch.no_grad()
def normalize_group_rows(rows: torch.Tensor, epsilon: float, *, max_norm: float = 1.0):
    """Compatibility name for SHRINK-ONLY clipping followed by the group mean.

    Rows already include pooled CVaR and gradient_scale. The cap is fixed in the
    run binding, not a batch statistic. Zero/small rows keep scale 1, zero rows
    retain their 1/G share. `normalized_gradient_norm`/`target_norm` remain log
    aliases for the clipped norm/fixed cap, not a median-normalization operation.
    """
    if rows.ndim != 2 or min(rows.shape) < 1 or rows.dtype not in (torch.float32, torch.float64):
        raise ValueError("group rows must be a nonempty float32/float64 [G,P] tensor")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("group gradient epsilon must be finite and positive")
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("group-clip-norm must be finite and positive")
    if not bool(torch.isfinite(rows).all()):
        raise FloatingPointError("nonfinite group gradient before clipping")
    # Only G*P parameter data is promoted, never the H500 graph/adjoints.
    work = rows.double()
    def stable_norm(x):
        peak = x.abs().amax(-1)
        scaled = x / peak.clamp_min(torch.finfo(x.dtype).tiny)[..., None]
        return peak * scaled.square().sum(-1).sqrt()
    norms = stable_norm(work)
    if not bool(torch.isfinite(norms).all()):
        raise FloatingPointError("nonfinite group gradient norm")
    # Guard the denominator only; do not perturb any norm below the cap.
    scales = torch.where(norms > max_norm,
                         max_norm / norms.clamp_min(torch.finfo(work.dtype).tiny),
                         torch.ones_like(norms))
    clipped = work * scales[:, None]
    result = (clipped / rows.shape[0]).sum(0).to(rows.dtype)
    values = torch.stack((norms, stable_norm(clipped), scales, torch.full_like(norms, max_norm)), 1)
    if not bool(torch.isfinite(result).all() & torch.isfinite(values).all()):
        raise FloatingPointError("nonfinite clipped group gradient")
    return result, {"version": GROUP_BALANCE_VERSION, "clip_norm": max_norm,
                    "columns": ["raw_gradient_norm", "normalized_gradient_norm", "multiplier", "target_norm"],
                    "values": values}


def _batched_group_vjp(costs, parameters, seeds, *, retain_graph):
    """Vectorize the existing graph's VJP with modern vmap, preserving None."""
    active = []

    def vjp(seed):
        gradients = torch.autograd.grad(
            costs, parameters, grad_outputs=seed, allow_unused=True,
            retain_graph=retain_graph,
        )
        # vmap runs this Python body once per chunk. Graph connectivity is
        # independent of the cotangent values; only tensor outputs are batched.
        active[:] = [gradient is not None for gradient in gradients]
        tensors = tuple(gradient for gradient in gradients if gradient is not None)
        # An all-unused graph still needs a tensor output for vmap. Discard this
        # placeholder below: unused parameters must not acquire zero .grad/Adam.
        return tensors if tensors else (seed.new_zeros(()),)

    batched = iter(torch.vmap(vjp)(seeds))
    return tuple(next(batched) if used else None for used in active)


def backward_group_gradients(costs, coefficients, parameters, config: GroupBalanceConfig,
                             *, gradient_scale: float):
    """Compute G group VJPs on the SAME forward graph, then clip and merge.

    Chunk>1 uses modern vmap over group cotangents, not individual-scene
    cotangents. The last chunk frees the graph. Chunk=1 is a serial reference,
    never a silent retry.
    No parameter .grad is written until the caller receives all finite results.
    """
    if (costs.ndim != 1 or coefficients.ndim != 2 or coefficients.shape[1] != costs.numel()
            or coefficients.shape[0] < 1 or coefficients.requires_grad or not parameters):
        raise ValueError("invalid group VJP inputs")
    if not math.isfinite(gradient_scale) or gradient_scale <= 0:
        raise ValueError("gradient_scale must be finite and positive")
    if not bool(torch.isfinite(costs).all() & torch.isfinite(coefficients).all()):
        raise FloatingPointError("nonfinite group costs or coefficients")
    groups = coefficients.shape[0]
    chunks, used, calls = [], [False] * len(parameters), 0
    for start in range(0, groups, config.vjp_chunk_size):
        end = min(start + config.vjp_chunk_size, groups)
        seeds = gradient_scale * coefficients[start:end]
        if not bool(torch.isfinite(seeds).all()):
            raise FloatingPointError("nonfinite group VJP seeds")
        if end - start > 1:
            gradients = _batched_group_vjp(
                costs, parameters, seeds, retain_graph=end < groups)
        else:
            gradients = torch.autograd.grad(
                costs, parameters, grad_outputs=seeds[0], allow_unused=True,
                retain_graph=end < groups,
            )
        parts = []
        for j, (parameter, gradient) in enumerate(zip(parameters, gradients)):
            used[j] = used[j] or gradient is not None
            if gradient is None:
                part = parameter.new_zeros(end - start, parameter.numel())
            else:
                part = gradient.detach().reshape(end - start, -1)
            parts.append(part)
        chunks.append(torch.cat(parts, 1))
        calls += 1
    rows = torch.cat(chunks, 0)
    combined, report = normalize_group_rows(rows, config.gradient_epsilon, max_norm=config.clip_norm)
    offsets, result = 0, []
    for parameter, active in zip(parameters, used):
        size = parameter.numel()
        result.append(combined[offsets:offsets + size].reshape_as(parameter) if active else None)
        offsets += size
    report.update(group_count=groups, vjp_calls=calls, vjp_chunk_size=config.vjp_chunk_size)
    return result, report
