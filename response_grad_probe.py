"""Training-only group derivatives from one native closed-loop backward traversal.

The Actor's GRUCell/Linear forward and native state adjoints are NOT replaced.
Tensor-output hooks observe each local adjoint, reconstruct only local parameter
partials, and reduce them to fixed groups before summing across time. Applicable
only to the current scene-independent GRUCell + affine readout Actor.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

from response_policy import ResponseMotorPolicy

PROBE_VERSION = "native-gru-linear-group-probe-v1"


@torch.no_grad()
def gru_gate_deltas(
    layer: nn.GRUCell, x: torch.Tensor, h: torch.Tensor, adjoint: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local derivatives of the two gate affine projections in PyTorch r,z,n order.

    The reset gate multiplies (W_hn h + b_hn), including the recurrent bias.
    These recomputed gates never enter the forward flight or state adjoints.
    Native CPU/CUDA fusion and reduction order can cause floating-point
    differences relative to independent VJPs; this is not a norm estimator.
    """
    gi = F.linear(x, layer.weight_ih, layer.bias_ih)
    gh = F.linear(h, layer.weight_hh, layer.bias_hh)
    ir, iz, inn = gi.chunk(3, -1)
    hr, hz, hn = gh.chunk(3, -1)
    r = torch.sigmoid(ir + hr)
    z = torch.sigmoid(iz + hz)
    n = torch.tanh(inn + r * hn)
    dn = adjoint * (1 - z) * (1 - n.square())
    dz = adjoint * (h - n) * z * (1 - z)
    dr = dn * hn * r * (1 - r)
    return torch.cat((dr, dz, dn), -1), torch.cat((dr, dz, dn * r), -1)


class GroupGradientProbe:
    """Observe once; directly accumulate GxP, never NxP or HxNxP derivatives.

    Every live row has a UNIQUE padded group slot derived from the original
    initial partition. Packing uses index_copy (not a floating atomic scatter);
    bmm reduces the member dimension. The only persistent derivative storage
    is GxP FP64 accumulators. Captured inputs are detached, not copied/replayed.

    One scalar VJP uses the SUM of all group seeds. Row-independent physics,
    sensing, Actor features and detached CVaR make each row's adjoint belong
    solely to that scene's group. BatchNorm, cross-scene attention, coupled
    multi-agent dynamics, parameter-dependent losses outside these two modules,
    higher-order differentiation and parameter sharing beyond this Actor are
    deliberately unsupported.
    """
    def __init__(self, policy: ResponseMotorPolicy, indices: torch.Tensor, counts: torch.Tensor):
        if type(policy) is not ResponseMotorPolicy:
            raise ValueError("group probe supports only the current ResponseMotorPolicy")
        if type(policy.response_memory) is not nn.GRUCell or type(policy.readout) is not nn.Linear:
            raise ValueError("group probe requires native GRUCell and Linear")
        expected = {
            "response_memory.weight_ih", "response_memory.weight_hh",
            "response_memory.bias_ih", "response_memory.bias_hh",
            "readout.weight", "readout.bias",
        }
        self.named = list(policy.named_parameters())
        if {name for name, _ in self.named} != expected or any(not p.requires_grad for _,p in self.named):
            raise ValueError("group probe requires the six unchanged, trainable Actor parameter tensors")
        self.policy = policy
        self.parameters = [p for _, p in self.named]
        self.versions = [(id(p), p._version) for p in self.parameters]
        self.n = int(counts.sum())
        if indices.ndim != 2 or indices.shape[0] != counts.numel() or self.n < 1:
            raise ValueError("invalid probe group layout")
        self.groups, self.width = indices.shape
        p0 = self.parameters[0]
        if (p0.dtype not in (torch.float32, torch.float64)
                or torch.is_autocast_enabled(p0.device.type)):
            raise ValueError("probe supports float32/64, not mixed precision/autocast")
        if any(p.dtype != p0.dtype or p.device != p0.device for p in self.parameters):
            raise ValueError("all probe parameters must share dtype/device")
        if indices.device != p0.device or counts.device != p0.device:
            raise ValueError("probe group layout must match Actor device")
        flat = indices.flatten()
        valid = flat < self.n
        if not torch.equal(flat[valid].sort().values, torch.arange(self.n, device=flat.device)):
            raise ValueError("probe requires a unique complete scene partition")
        self.slots = torch.empty(self.n, device=flat.device, dtype=torch.long)
        self.slots[flat[valid]] = torch.arange(flat.numel(), device=flat.device)[valid]
        self.scene_group_ids = self.slots // self.width
        self.rows = torch.zeros(self.groups, sum(p.numel() for p in self.parameters),
                                dtype=torch.float64, device=p0.device)
        self.views = {}
        offset = 0
        for name,p in self.named:
            self.views[name] = self.rows[:, offset:offset+p.numel()].view(self.groups, *p.shape)
            offset += p.numel()
        self._module_handles = []
        self._tensor_handles = []
        self._payloads = []
        self._current_slots = None
        self._capturing = False
        self._has_captured = False
        self._armed = False
        self._used = False
        self._closed = False
        self.captured = {"gru": 0, "linear": 0}
        self.observed = {"gru": 0, "linear": 0}

    @contextmanager
    def capture(self) -> Iterator["GroupGradientProbe"]:
        if self._has_captured or self._used or self._closed:
            raise RuntimeError("group probe capture is single-use")
        self._capturing = True
        self._has_captured = True
        self._module_handles = [
            self.policy.response_memory.register_forward_hook(self._gru_forward),
            self.policy.readout.register_forward_hook(self._linear_forward),
        ]
        try:
            yield self
        except BaseException:
            self.close()
            raise
        finally:
            for handle in self._module_handles:
                handle.remove()
            self._module_handles.clear()
            self._current_slots = None
            self._capturing = False

    def begin_step(self, indices: torch.Tensor) -> None:
        """Called only by rollout, BEFORE Actor execution; indices are original rows."""
        if not self._capturing:
            raise RuntimeError("probe step requires active forward capture")
        self._current_slots = self.slots.index_select(0, indices)

    def _watch(self, kind, inputs, output) -> None:
        if self._current_slots is None or len(output) != len(self._current_slots):
            raise RuntimeError("missing original scene IDs for Actor probe")
        if not torch.is_grad_enabled() or not output.requires_grad:
            raise RuntimeError("probe capture requires the full native autograd graph")
        payload = [tuple(x.detach() for x in inputs), self._current_slots]
        self._payloads.append(payload)
        self.captured[kind] += 1

        def observe(adjoint):
            if not self._armed:
                return None  # Independent diagnostics can traverse an unarmed graph.
            if not payload:
                raise RuntimeError("a group probe output was differentiated twice")
            try:
                with torch.no_grad():
                    values, slots = payload
                    if kind == "gru":
                        self._gru_partial(values[0], values[1], adjoint.detach(), slots)
                    else:
                        self._linear_partial(values[0], adjoint.detach(), slots)
                self.observed[kind] += 1
            finally:
                payload.clear()
            return None  # NEVER alter the native state/physics/GRU adjoint.

        self._tensor_handles.append(output.register_hook(observe))

    def _gru_forward(self, module, inputs, output) -> None:
        if len(inputs) != 2:
            raise RuntimeError("GRU probe requires explicit recurrent hidden state")
        self._watch("gru", inputs, output)

    def _linear_forward(self, module, inputs, output) -> None:
        self._watch("linear", inputs, output)

    def _pack(self, value, slots):
        packed = value.new_zeros(self.groups*self.width, value.shape[-1])
        packed.index_copy_(0, slots, value)
        return packed.view(self.groups, self.width, value.shape[-1])

    def _affine_partial(self, prefix, x, delta, slots):
        a, d = self._pack(x, slots), self._pack(delta, slots)
        self.views[prefix+".weight"].add_(torch.bmm(d.transpose(1,2), a).double())
        self.views[prefix+".bias"].add_(d.sum(1).double())

    def _linear_partial(self, x, delta, slots):
        self._affine_partial("readout", x, delta, slots)

    def _gru_partial(self, x, h, delta, slots):
        di, dh = gru_gate_deltas(self.policy.response_memory, x, h, delta)
        a, b = self._pack(x, slots), self._pack(h, slots)
        i, r = self._pack(di, slots), self._pack(dh, slots)
        self.views["response_memory.weight_ih"].add_(torch.bmm(i.transpose(1,2),a).double())
        self.views["response_memory.weight_hh"].add_(torch.bmm(r.transpose(1,2),b).double())
        self.views["response_memory.bias_ih"].add_(i.sum(1).double())
        self.views["response_memory.bias_hh"].add_(r.sum(1).double())

    def close(self) -> None:
        self._armed = False
        for handle in self._module_handles + self._tensor_handles:
            handle.remove()
        for payload in self._payloads:
            payload.clear()
        self._module_handles.clear()
        self._tensor_handles.clear()
        self._payloads.clear()
        self._current_slots = None
        self._closed = True

    def backward(self, costs, coefficients, parameters, config, *, gradient_scale):
        from response_groups import normalize_group_rows

        if self._used or self._closed:
            raise RuntimeError("group probe backward is single-use")
        self._used = True
        try:
            if [(id(p),p._version) for p in parameters] != self.versions:
                raise RuntimeError("Actor parameters changed between captured forward and backward")
            if costs.shape != (self.n,) or coefficients.shape != (self.groups,self.n):
                raise ValueError("group probe cost/coefficient shape mismatch")
            if coefficients.requires_grad or not math.isfinite(gradient_scale) or gradient_scale <= 0:
                raise ValueError("group probe requires detached weights and positive gradient scale")
            group_range = torch.arange(self.groups, device=costs.device)[:,None]
            off_group = group_range != self.scene_group_ids[None,:]
            if bool((coefficients.masked_select(off_group) != 0).any()):
                raise ValueError("group probe seeds mix scene partitions")
            seeds = gradient_scale * coefficients.sum(0)
            if not bool(torch.isfinite(costs).all() & torch.isfinite(seeds).all()):
                raise FloatingPointError("nonfinite probe costs or seeds")
            self._armed = True
            # One traversal. The aggregate output is not published to .grad;
            # local hook partials preserve group ownership before parameter sums.
            torch.autograd.grad(costs, parameters, grad_outputs=seeds, allow_unused=True)
            self._armed = False
            if self.observed != self.captured or not self.captured["gru"]:
                raise RuntimeError("incomplete group probe backward observation")
            combined, report = normalize_group_rows(self.rows, config.gradient_epsilon,
                                                      max_norm=config.clip_norm)
            result, offset = [], 0
            for p in parameters:
                result.append(combined[offset:offset+p.numel()].reshape_as(p).to(p))
                offset += p.numel()
            report.update(backend="probe", probe_version=PROBE_VERSION,
                          group_count=self.groups, vjp_calls=1, graph_traversals=1,
                          vjp_chunk_size=None, observed_calls=dict(self.observed),
                          accumulator_bytes=self.rows.numel()*self.rows.element_size())
            return result, report
        finally:
            self.close()
