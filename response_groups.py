"""Training-only physical-group gradients, normalized before aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import math
import threading

import torch

from env_l2f import L2FState


GROUP_BALANCE_VERSION = "physical-group-gradient-median-v1"
GROUP_FEATURE_NAMES = ("thrust_to_weight", "torque_to_inertia",
                       "motor_time_rising", "motor_time_falling")


@dataclass(frozen=True)
class GroupBalanceConfig:
    enabled: bool = False
    max_groups: int = 16
    min_scenarios: int = 32
    gradient_epsilon: float = 1e-12
    vjp_chunk_size: int = 16
    gru_vmap_mode: str = "fallback"

    def __post_init__(self):
        if (not isinstance(self.max_groups, int) or isinstance(self.max_groups, bool)
                or not 1 <= self.max_groups <= 64
                or self.max_groups & (self.max_groups - 1)):
            raise ValueError("group-max-groups must be a power of two in [1,64]")
        if (not isinstance(self.min_scenarios, int) or isinstance(self.min_scenarios, bool)
                or self.min_scenarios < 32):
            raise ValueError("group-min-scenarios must be at least 32")
        if not math.isfinite(self.gradient_epsilon) or self.gradient_epsilon <= 0:
            raise ValueError("group-gradient-epsilon must be finite and positive")
        if (not isinstance(self.vjp_chunk_size, int) or isinstance(self.vjp_chunk_size, bool)
                or not 1 <= self.vjp_chunk_size <= 64):
            raise ValueError("group-vjp-chunk-size must be an integer in [1,64]")

        if self.gru_vmap_mode not in ("fallback", "native", "verify", "sparse", "sparse-verify"):
            raise ValueError("invalid group-gru-vmap-mode")

    @classmethod
    def from_args(cls, args):
        return cls(args.group_balance, args.group_max_groups, args.group_min_scenarios,
                   args.group_gradient_epsilon, args.group_vjp_chunk_size,
                   getattr(args, "group_gru_vmap_mode", "fallback"))


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
def group_gradient_coefficients(base_weights, initial, config: GroupBalanceConfig):
    """Build G vector-Jacobian seeds, never N per-scene seeds or cost scales.

    Row g defines L_g = (N/n_g) sum_{i in g} w_i C_i, with the original
    detached pooled CVaR weights w_i. Equal group sizes recover the original
    objective when group gradients are averaged without normalization.
    All initial scenes, including early failures, belong to exactly one row.
    """
    if not config.enabled:
        return None, None
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
              "minimum_scenarios": config.min_scenarios, "columns": columns, "values": values}
    return coefficients.detach(), report


@torch.no_grad()
def normalize_group_rows(rows: torch.Tensor, epsilon: float):
    """Equalize whole-Actor group norms, then average; NOT cost normalization.

    h_g already includes gradient_scale and CVaR. m is the lower median of
    nonzero ||h_g|| (zero if all zero); q_g=m/max(||h_g||,epsilon).
    Return mean_g(q_g*h_g). Zero groups stay zero and keep their 1/G share;
    below-epsilon groups are not amplified all the way to m. No clipping,
    layerwise normalization, reweighting by cost, or differentiation of q_g.
    """
    if rows.ndim != 2 or min(rows.shape) < 1 or rows.dtype not in (torch.float32, torch.float64):
        raise ValueError("group rows must be a nonempty float32/float64 [G,P] tensor")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("group gradient epsilon must be finite and positive")
    if not bool(torch.isfinite(rows).all()):
        raise FloatingPointError("nonfinite group gradient before normalization")
    # Only G*P parameter data is promoted, not the H500 graph or batched adjoints.
    work = rows.double()
    def stable_norm(x):
        peak = x.abs().amax(-1)
        scaled = x / peak.clamp_min(torch.finfo(x.dtype).tiny)[..., None]
        return peak * scaled.square().sum(-1).sqrt()
    norms = stable_norm(work)
    if not bool(torch.isfinite(norms).all()):
        raise FloatingPointError("nonfinite group gradient norm")
    positive = norms > 0
    ordered = norms.masked_fill(~positive, torch.inf).sort().values
    rank = ((positive.sum() - 1).clamp_min(0) // 2).reshape(1)
    target = torch.where(positive.any(), ordered.gather(0, rank)[0], norms.new_zeros(()))
    scales = torch.where(positive, target / norms.clamp_min(epsilon), torch.zeros_like(norms))
    normalized = work * scales[:, None]
    # Divide before summing to avoid an unnecessarily large intermediate sum.
    result = (normalized / rows.shape[0]).sum(0).to(rows.dtype)
    values = torch.stack((norms, stable_norm(normalized), scales, target.expand_as(norms)), 1)
    if not bool(torch.isfinite(result).all() & torch.isfinite(values).all()):
        raise FloatingPointError("nonfinite normalized group gradient")
    return result, {"version": GROUP_BALANCE_VERSION,
                    "columns": ["raw_gradient_norm", "normalized_gradient_norm", "multiplier", "target_norm"],
                    "values": values}



# Opt-in CUDA batching adapters; never replace the deployed GRU or forward graph.
_RULE_LOCK = threading.RLock()
_OP_NAME = "aten::_thnn_fused_gru_cell_backward"


def _stack_native_backward(native, grad, workspace, has_bias, *, verify=False):
    """[groups, scenes, hidden] -> one original elementwise CUDA invocation.

    The workspace is the saved native [scenes, 5*hidden] gate state, not a
    recomputation. Bias reductions retain the original [scenes, 3*hidden]
    shape and order per group; NEVER reduce across the group dimension.
    This helper also permits CPU stand-ins for shape/dispatch tests. Those
    tests do not establish CUDA arithmetic equivalence or CUDA speed.
    """
    if (grad.ndim != 3 or workspace.ndim != 3 or grad.shape[:2] != workspace.shape[:2]
            or workspace.shape[2] != 5 * grad.shape[2]
            or grad.dtype != workspace.dtype or grad.device != workspace.device
            or grad.dtype not in (torch.float32, torch.float64)):
        raise ValueError("native GRU batching needs matching float32/64 [G,B,H], [G,B,5H]")
    groups, scenes, hidden = grad.shape
    if groups < 1 or scenes < 1 or hidden < 1:
        raise ValueError("native GRU batching needs nonempty dimensions")
    gi, gh, hx, _, _ = native(
        grad.reshape(groups * scenes, hidden),
        workspace.reshape(groups * scenes, 5 * hidden), False,
    )
    gi = gi.reshape(groups, scenes, 3 * hidden)
    gh = gh.reshape(groups, scenes, 3 * hidden)
    hx = hx.reshape(groups, scenes, hidden)
    # Keep these small reductions identical to the legacy per-group operator.
    bi = torch.stack([gi[g].sum(0) for g in range(groups)]) if has_bias else None
    bh = torch.stack([gh[g].sum(0) for g in range(groups)]) if has_bias else None
    result = (gi, gh, hx, bi, bh)
    if verify:
        references = [native(grad[g], workspace[g], has_bias) for g in range(groups)]
        for i, actual in enumerate(result):
            if actual is None:
                equal = all(row[i] is None for row in references)
            else:
                expected = torch.stack([row[i] for row in references])
                equal = torch.equal(actual, expected)
            if not equal:
                raise RuntimeError("native GRU vmap verification failed before Actor gradient publication")
    return result


def _sparse_native_backward(native, grad, workspace, has_bias, *, flags, verify=False):
    """Exploit disjoint group support ONLY inside the row-local GRU kernel.

    Keep all later [G,B,...] tensors and native parameter/bias reductions in
    their original shapes. No scene slicing, no rerun, and no changed GEMM
    reduction order. Validate the support assumption before publishing any
    Actor gradients. Detection stays on-device until the scope ends.
    """
    if (grad.ndim != 3 or workspace.ndim != 2
            or workspace.shape != (grad.shape[1], 5 * grad.shape[2])
            or grad.device != workspace.device or grad.dtype != workspace.dtype
            or grad.dtype not in (torch.float32, torch.float64)):
        raise ValueError("sparse GRU batching needs [G,B,H] and shared [B,5H]")
    groups, scenes, hidden = grad.shape
    if groups < 1 or scenes < 1 or hidden < 1:
        raise ValueError("sparse GRU dimensions must be nonempty")
    support = (grad != 0).any(-1)
    flags.append((support.sum(0) <= 1).all() & torch.isfinite(grad).all()
                 & torch.isfinite(workspace).all())
    owner = support.to(torch.int64).argmax(0)
    selected = grad.gather(0, owner[None, :, None].expand(1, scenes, hidden))[0]
    gi, gh, hx, _, _ = native(selected, workspace, False)
    mask = (torch.arange(groups, device=grad.device)[:, None] == owner[None])[:, :, None]
    gi = torch.where(mask, gi[None], 0.0)
    gh = torch.where(mask, gh[None], 0.0)
    hx = torch.where(mask, hx[None], 0.0)
    bi = torch.stack([gi[g].sum(0) for g in range(groups)]) if has_bias else None
    bh = torch.stack([gh[g].sum(0) for g in range(groups)]) if has_bias else None
    result = (gi, gh, hx, bi, bh)
    if verify:
        refs = [native(grad[g], workspace, has_bias) for g in range(groups)]
        for i, actual in enumerate(result):
            equal = (all(row[i] is None for row in refs) if actual is None else
                     torch.equal(actual, torch.stack([row[i] for row in refs])))
            if not equal:
                raise RuntimeError("sparse GRU vmap verification failed before Actor gradient publication")
    return result



def _make_batch_rule(native, *, verify, owner_thread=None, sparse=False, flags=None, allowed_graphs=None, stats=None):
    """Narrow compatibility adapter for the current FuncTorchBatched dispatcher.

    PyTorch 2.2 lacks the newer public register_vmap helper. Isolate the small
    private-API dependency here, rather than monkeypatching autograd/GRUCell.
    Inputs are unwrapped only at the current level. Nested vmap is rejected.
    """
    from torch._C import _functorch as ft

    def rule(grad, workspace, has_bias):
        level = ft.maybe_current_level()
        if level is None:
            raise RuntimeError("GRU batching rule invoked outside vmap")
        gu, gd = ft._unwrap_batched(grad, level)
        wu, wd = ft._unwrap_batched(workspace, level)
        if ft.is_batchedtensor(gu) or ft.is_batchedtensor(wu):
            raise RuntimeError("nested GRU vmap is not supported by this opt-in backend")
        if gd is None and wd is None:
            raise RuntimeError("GRU batching rule received no batched input")
        groups = gu.shape[gd] if gd is not None else wu.shape[wd]
        gu = gu.movedim(gd, 0) if gd is not None else gu.unsqueeze(0).expand(groups, *gu.shape)
        shared_workspace = wu if wd is None else None
        wu = wu.movedim(wd, 0) if wd is not None else wu.unsqueeze(0).expand(groups, *wu.shape)
        with torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(torch._C.DispatchKey.FuncTorchBatched)
        ):
            # A temporary registration is process-wide. Other threads retain
            # ordinary per-group behavior instead of silently opting in.
            if ((allowed_graphs is not None and torch._C._current_graph_task_id() not in allowed_graphs)
                    or (allowed_graphs is None and owner_thread is not None
                        and threading.get_ident() != owner_thread)):
                columns = [native(gu[g], wu[g], has_bias) for g in range(groups)]
                result = tuple(None if columns[0][i] is None else
                               torch.stack([row[i] for row in columns]) for i in range(5))
            elif sparse:
                if stats is not None:
                    stats['calls'] = stats.get('calls', 0) + 1
                    stats['logical_rows'] = stats.get('logical_rows', 0) + gu.shape[0] * gu.shape[1]
                    stats['native_rows'] = stats.get('native_rows', 0) + gu.shape[1]
                if shared_workspace is None:
                    raise RuntimeError("sparse GRU backend requires an unbatched saved workspace")
                result = _sparse_native_backward(native, gu, shared_workspace, has_bias,
                                                 flags=flags, verify=verify)
            else:
                if stats is not None:
                    stats['calls'] = stats.get('calls', 0) + 1
                    stats['logical_rows'] = stats.get('logical_rows', 0) + gu.shape[0] * gu.shape[1]
                    stats['native_rows'] = stats.get('native_rows', 0) + gu.shape[0] * gu.shape[1]
                result = _stack_native_backward(native, gu, wu, has_bias, verify=verify)
        return tuple(None if x is None else ft._add_batch_dim(x, 0, level) for x in result)
    return rule


@contextmanager
def native_gru_vmap(mode="fallback", costs=None, stats=None):
    """Scoped operator registration, always removed on success or failure.

    No override of an upstream batching rule. CPU ordinary GRU does not call
    the CUDA fused operator; a CPU run is not a validation of this backend.
    """
    if mode not in ("fallback", "native", "verify", "sparse", "sparse-verify"):
        raise ValueError("invalid group-gru-vmap-mode")
    if mode == "fallback":
        yield
        return
    with _RULE_LOCK:
        if torch._C._dispatch_has_kernel_for_dispatch_key(_OP_NAME, "FuncTorchBatched"):
            # An upstream/native rule or outer instance already owns dispatch.
            # Fail closed rather than replacing unknown semantics.
            raise RuntimeError("GRU already has a batching rule; use the unchanged fallback backend")
        native = torch.ops.aten._thnn_fused_gru_cell_backward.default
        library = torch.library.Library("aten", "IMPL", "FuncTorchBatched")
        flags = []
        allowed_graphs = set() if costs is not None else None
        handle = None
        try:
            if costs is not None:
                # CUDA autograd may execute on a worker thread. Tag the graph
                # task via the cost hook instead of checking Python thread ID.
                def tag_graph(gradient):
                    allowed_graphs.add(torch._C._current_graph_task_id())
                handle = costs.register_hook(tag_graph)
            library.impl("_thnn_fused_gru_cell_backward", _make_batch_rule(
                native, verify=mode in ("verify", "sparse-verify"), owner_thread=threading.get_ident(),
                sparse=mode.startswith("sparse"), flags=flags, allowed_graphs=allowed_graphs, stats=stats))
            yield
            if flags and not bool(torch.stack(flags).all()):
                raise RuntimeError("sparse GRU requires finite, disjoint per-scene group cotangents")
        finally:
            if handle is not None:
                handle.remove()
            library._destroy()


def _batched_group_vjp(costs, parameters, seeds, *, retain_graph, gru_vmap_mode="fallback", gru_vmap_stats=None):
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

    if gru_vmap_mode == "fallback":
        batched = iter(torch.vmap(vjp)(seeds))
    else:
        with native_gru_vmap(gru_vmap_mode, costs, gru_vmap_stats):
            batched = iter(torch.vmap(vjp)(seeds))
    return tuple(next(batched) if used else None for used in active)


def backward_group_gradients(costs, coefficients, parameters, config: GroupBalanceConfig,
                             *, gradient_scale: float):
    """Compute G group VJPs on the SAME forward graph, then normalize and merge.

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
    gru_stats = {}
    for start in range(0, groups, config.vjp_chunk_size):
        end = min(start + config.vjp_chunk_size, groups)
        seeds = gradient_scale * coefficients[start:end]
        if not bool(torch.isfinite(seeds).all()):
            raise FloatingPointError("nonfinite group VJP seeds")
        if end - start > 1:
            gradients = _batched_group_vjp(
                costs, parameters, seeds, retain_graph=end < groups,
                gru_vmap_mode=config.gru_vmap_mode, gru_vmap_stats=gru_stats)
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
    combined, report = normalize_group_rows(rows, config.gradient_epsilon)
    offsets, result = 0, []
    for parameter, active in zip(parameters, used):
        size = parameter.numel()
        result.append(combined[offsets:offsets + size].reshape_as(parameter) if active else None)
        offsets += size
    report.update(group_count=groups, vjp_calls=calls, vjp_chunk_size=config.vjp_chunk_size,
                  gru_vmap_mode=config.gru_vmap_mode, gru_vmap_stats=gru_stats)
    return result, report
