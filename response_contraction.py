"""Training-only, sampled differential dissipativity of the true closed loop.

The metric is uniformly SPD. We require non-expansion of the FULL dynamic
state and dissipation of task-relevant perturbations, not convergence of an
arbitrary absolute heading. No gradient decay, Critic, or altered plant is used.
Finite directional checks are regularization/diagnostics, NOT a certificate.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from env_l2f import quaternion_rotation
from response_task import ResponseClosedLoopState, rollout, _select_rows

CONTRACTION_VERSION = "sampled-task-dissipativity-v2"
LEGACY_CONTRACTION_VERSION = "sampled-task-dissipativity-v1"
CONTEXT_DIM = 20
PHYSICS_CONTEXT_DIM = 67


@dataclass(frozen=True)
class ContractionConfig:
    weight: float = 0.0  # Explicit opt-in; reference launch configs enable it.
    steps: int = 10      # 100 ms, independent of the task BPTT window.
    samples: int = 8
    directions: int = 2
    rate: float = 0.1    # Task-direction dissipation per second, NOT full-state rate.
    hidden_dim: int = 64
    rank: int = 4
    metric_min: float = 0.5
    metric_max: float = 2.0
    context: str = 'legacy'  # Keep original checkpoints and ablation baseline.
    fusion: str = 'concat'
    sampling: str = 'uniform'
    prefix_weight: float = 0.0
    prefix_max_gain_squared: float = 4.0  # Norm amplification budget = 2.
    tail_fraction: float = 1.0  # Mean by default; <1 targets worst sampled pairs.
    actor_max_ratio: float = 0.0  # 0 disables auxiliary/task gradient norm cap.
    metric_gradient_scale: float = 0.0  # 0 retains the legacy shared multiplier.

    def __post_init__(self) -> None:
        for name in ('weight', 'rate', 'metric_min', 'metric_max', 'prefix_weight',
                     'prefix_max_gain_squared', 'tail_fraction', 'actor_max_ratio',
                     'metric_gradient_scale'):
            if not math.isfinite(getattr(self, name)):
                raise ValueError('contraction constants must be finite')
        if self.weight < 0 or self.rate < 0 or not 0 < self.metric_min < self.metric_max:
            raise ValueError('invalid contraction weight, rate or metric bounds')
        for name in ('steps', 'samples', 'directions', 'hidden_dim', 'rank'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError('contraction sizes must be positive integers')
        if self.context not in ('legacy', 'physics') or self.fusion not in ('concat', 'film_gated'):
            raise ValueError('unknown contraction context or fusion')
        if self.sampling not in ('uniform', 'mass_quantiles'):
            raise ValueError('unknown contraction sampling')
        if (min(self.prefix_weight, self.actor_max_ratio, self.metric_gradient_scale) < 0
                or self.prefix_max_gain_squared < 1 or not 0 < self.tail_fraction <= 1):
            raise ValueError('invalid contraction guidance constants')

    @classmethod
    def from_args(cls, args):
        return cls(**{f.name: getattr(args, 'contraction_' + f.name) for f in fields(cls)})


def _detached(closed: ResponseClosedLoopState) -> ResponseClosedLoopState:
    return ResponseClosedLoopState(*(
        replace(s, **{f.name: getattr(s, f.name).detach() for f in fields(s)})
        for s in (closed.physical, closed.policy)
    ))


def _small_quaternion(v: torch.Tensor) -> torch.Tensor:
    # Smooth right retraction. At zero, its rotation tangent is exactly v.
    half = v * 0.5
    return torch.cat((torch.ones_like(half[:, :1]), half), -1) / (1 + half.square().sum(-1, keepdim=True)).sqrt()


def _multiply_quaternion(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    w = a[:, :1]*b[:, :1] - (a[:, 1:]*b[:, 1:]).sum(-1, keepdim=True)
    xyz = a[:, :1]*b[:, 1:] + b[:, :1]*a[:, 1:] + torch.linalg.cross(a[:, 1:], b[:, 1:])
    return torch.cat((w, xyz), -1)


class StateGeometry:
    """Legal rotation perturbations and fixed, nondimensional dynamic scales.

    The tangent has memory_dim+36 entries. Its embedding has memory_dim+48
    entries because both rotations use nine sign-invariant matrix entries.
    This redundancy is not treated as nine independent rotation freedoms.
    History is a real controller state, not reinitialized from current physics.
    """
    def __init__(self, closed: ResponseClosedLoopState, integral_limit: float) -> None:
        self.base = _detached(closed)
        p = self.base.physical
        self.integral_limit = integral_limit
        self.memory_dim = closed.policy.memory.shape[-1]
        self.tangent_dim = self.memory_dim + 36
        self.embedding_dim = self.memory_dim + 48
        self.position_scale = p.position_limit[:, None]
        self.motor_scale = (p.motor_max - p.motor_min)[:, None] * .5
        if bool((self.position_scale <= 0).any() | (self.motor_scale <= 0).any()):
            raise ValueError('state scales must be positive')
        rows = torch.arange(p.position.shape[0], device=p.position.device)
        # The cached rotation is a measured matrix, R_true + fixed sensor noise.
        index = (p.step_index - 1).clamp_min(0)
        if p.noise_tape.shape[1] == 1:
            index = torch.zeros_like(index)
        self.previous_rotation_noise = p.noise_tape[rows, index, 6:15].reshape(-1, 3, 3)

    def retract(self, delta: torch.Tensor) -> ResponseClosedLoopState:
        if delta.shape != (self.base.physical.position.shape[0], self.tangent_dim):
            raise ValueError('wrong tangent shape')
        widths = (3,3,3,3,4,4,self.memory_dim,3,3,3,3,4)
        dp,dv,dr,dw,dm,da,dh,di,dpv,dpw,dpr,do = delta.split(widths, -1)
        p, s = self.base.physical, self.base.policy
        old_true = s.previous_rotation - self.previous_rotation_noise
        rotated_delta = old_true @ quaternion_rotation(_small_quaternion(dpr)) - old_true
        return ResponseClosedLoopState(
            replace(p, position=p.position+self.position_scale*dp,
                    velocity=p.velocity+2*dv,
                    orientation=_multiply_quaternion(p.orientation, _small_quaternion(dr)),
                    omega=p.omega+10*dw, motor=p.motor+self.motor_scale*dm,
                    previous_action=p.previous_action+da),
            replace(s, memory=s.memory+dh, integral=s.integral+self.integral_limit*di,
                    previous_velocity=s.previous_velocity+2*dpv,
                    previous_omega=s.previous_omega+10*dpw,
                    previous_rotation=s.previous_rotation+rotated_delta,
                    older_action=s.older_action+do),
        )

    def pack(self, closed: ResponseClosedLoopState) -> torch.Tensor:
        p, s = closed.physical, closed.policy
        return torch.cat((
            p.position/self.position_scale, p.velocity/2,
            p.rotation.flatten(1)/math.sqrt(2), p.omega/10,
            (p.motor-p.motor_min[:,None])/self.motor_scale-1, p.previous_action,
            s.memory, s.integral/self.integral_limit, s.previous_velocity/2,
            s.previous_omega/10, s.previous_rotation.flatten(1)/math.sqrt(2), s.older_action,
        ), -1)

    @staticmethod
    def context(closed: ResponseClosedLoopState, mode: str = 'legacy') -> torch.Tensor:
        p = closed.physical
        # Privileged constants condition ONLY the training metric.
        legacy = torch.cat((
            p.mass[:,None].log()/5, p.inertia.log()/15,
            p.motor_time_rising.log()/5, p.motor_time_falling.log()/5,
            p.thrust_to_weight[:,None]/5, p.torque_to_inertia[:,None]/1200,
            p.external_force/(p.mass[:,None]*9.81),
            p.external_torque/(p.inertia*1000),
        ), -1)
        if mode == 'legacy':
            return legacy
        if mode != 'physics':
            raise ValueError('unknown metric context')
        # Express thrust in the normalized rotor coordinate r in [0,1], so
        # RPM and normalized-motor protocols have comparable physical units.
        c0, c1, c2 = p.thrust_coefficients.unbind(-1)
        low = p.motor_min[:, None]
        span = (p.motor_max - p.motor_min)[:, None]
        polynomial = torch.stack((c0 + c1*low + c2*low.square(),
                                  (c1 + 2*c2*low)*span, c2*span.square()), -1)
        polynomial = polynomial / (p.mass[:, None, None] * 9.81)
        sensor_scale = torch.cat((p.position_limit[:, None].expand(-1, 3),
                                  torch.full_like(p.velocity, 2.),
                                  torch.full_like(p.rotation.flatten(1), math.sqrt(2)),
                                  torch.full_like(p.omega, 10.)), -1)
        return torch.cat((legacy, p.arm_length[:, None].log()/5,
                          (p.rotor_positions/p.arm_length[:, None, None]).flatten(1),
                          p.rotor_torque_constant/p.arm_length[:, None],
                          polynomial.flatten(1), p.noise_std/sensor_scale), -1)

    @staticmethod
    def task_direction(tangent: torch.Tensor) -> torch.Tensor:
        # Heading is NOT deleted from dynamics or metric. Only task dissipation
        # omits an absolute heading target. Thrust-axis tilt and its coupling stay.
        thrust_axis = tangent[..., 6:15].reshape(*tangent.shape[:-1],3,3)[...,2] * math.sqrt(2)
        return torch.cat((tangent[..., :6], thrust_axis, tangent[...,15:18]), -1)


class ContractionMetric(nn.Module):
    """Bounded diagonal + low-rank SPD metric with cross-state couplings.

    M = lo I + (hi-lo)/2 * [diag(sigmoid(d)) + B'B/(1+||B||_F^2)].
    For every network output, lo I <= M <= hi I. No eigensolver or matrix
    inversion is needed to train energies, and zero metric collapse is excluded.
    """
    def __init__(self, memory_dim: int, config: ContractionConfig = ContractionConfig()) -> None:
        super().__init__()
        self.config = config
        self.dim = memory_dim + 48
        h = config.hidden_dim
        self.context_dim = CONTEXT_DIM if config.context == 'legacy' else PHYSICS_CONTEXT_DIM
        output_dim = self.dim*(config.rank+1)
        if config.fusion == 'concat':
            # Preserve original names, initialization order and shapes in legacy mode.
            self.net = nn.Sequential(nn.Linear(self.dim+self.context_dim,h),nn.SiLU(),
                                     nn.Linear(h,h),nn.SiLU(),nn.Linear(h,output_dim))
            head = self.net[-1]
        else:
            self.state_encoder = nn.Sequential(nn.Linear(self.dim,h),nn.SiLU())
            self.conditioner = nn.Sequential(nn.Linear(self.context_dim,h),nn.SiLU(),nn.Linear(h,2*h))
            self.gate = nn.Linear(h+self.context_dim,h)
            self.head = nn.Sequential(nn.Linear(h,h),nn.SiLU(),nn.Linear(h,output_dim))
            nn.init.normal_(self.conditioner[-1].weight, std=.005)
            nn.init.zeros_(self.conditioner[-1].bias)
            nn.init.normal_(self.gate.weight, std=.005)
            nn.init.ones_(self.gate.bias)
            head = self.head[-1]
        nn.init.normal_(head.weight, std=.005)
        nn.init.zeros_(head.bias)

    def factors(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[:-1] != context.shape[:-1] or context.shape[-1] != self.context_dim:
            raise ValueError('metric state/context shape mismatch')
        if self.config.fusion == 'concat':
            raw = self.net(torch.cat((x, context), -1))
        else:
            hidden = self.state_encoder(x)
            gamma, beta = self.conditioner(context).chunk(2, -1)
            gate = self.gate(torch.cat((hidden, context), -1)).sigmoid()
            # Bounded residual FiLM. Keep absolute physical scale information;
            # no per-sample LayerNorm or clipping of signed/log context values.
            hidden = hidden + gate * (.5*gamma.tanh()*hidden + .5*beta.tanh())
            raw = self.head(hidden)
        d = raw[..., :self.dim].sigmoid()
        b = raw[..., self.dim:].reshape(*x.shape[:-1], self.config.rank, self.dim)
        b = b / (1 + b.square().sum((-1,-2), keepdim=True)).sqrt()
        return d, b

    def energy(self, x: torch.Tensor, context: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        d,b = self.factors(x,context)
        lo,hi = self.config.metric_min,self.config.metric_max
        return lo*tangent.square().sum(-1) + .5*(hi-lo)*(
            (d*tangent.square()).sum(-1) + (b@tangent[...,None]).squeeze(-1).square().sum(-1))

    def matrix(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        d,b = self.factors(x,context)
        lo,hi = self.config.metric_min,self.config.metric_max
        eye = torch.eye(self.dim,device=x.device,dtype=x.dtype)
        return lo*eye + .5*(hi-lo)*(torch.diag_embed(d)+b.transpose(-1,-2)@b)



class _UnfusedGRU(nn.Module):
    """Same PyTorch GRU equations/parameters, without the CUDA fused primitive.

    The latter has no forward AD on some supported PyTorch releases (#174355).
    This is an auxiliary differentiation backend, never a deployed layer swap.
    """
    def __init__(self, cell: nn.GRUCell) -> None:
        super().__init__()
        self.cell = cell

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        ir, iz, inn = F.linear(x, self.cell.weight_ih, self.cell.bias_ih).chunk(3, -1)
        hr, hz, hn = F.linear(h, self.cell.weight_hh, self.cell.bias_hh).chunk(3, -1)
        reset, update = (ir + hr).sigmoid(), (iz + hz).sigmoid()
        candidate = (inn + reset * hn).tanh()
        return (h - candidate) * update + candidate


def _differentiation_policy(policy):
    # A shallow module view shares ALL original Parameters. Do not mutate the
    # original module registry, optimizer, deployment forward, or RNG stream.
    if not isinstance(policy.response_memory, nn.GRUCell):
        raise TypeError('contraction backend expects the reference GRUCell')
    view = copy.copy(policy)
    view._modules = policy._modules.copy()
    view.response_memory = _UnfusedGRU(policy.response_memory)
    return view


def local_flow(policy, simulator, closed, geometry, steps):
    """Use the production step/observation/termination path, without decay.

    Frozen rows are returned only for storage. Valid masks identify the genuine
    prefix including the crossing transition; diagnostics never score padding.
    """
    policy = _differentiation_policy(policy)
    values, valid = [geometry.pack(closed)], []
    for _ in range(steps):
        trace = rollout(policy, simulator, closed, 1, time_decay=0.0)
        closed = trace.end
        if simulator.params.protocol == 'raptor' and any(
            bool((getattr(closed.physical, name).abs() >= 100000).any())
            for name in ('position', 'velocity', 'omega')
        ):
            # This is the simulator's numerical guard, not a task boundary.
            # Its clipped derivative is not evidence of physical contraction.
            raise FloatingPointError('contraction probe reached numerical state clamp')
        values.append(geometry.pack(closed))
        valid.append(trace.valid[0])
    return torch.stack(values), torch.stack(valid)


@torch.no_grad()
def sample_boundaries(record, count: int, steps: int, seed: int, *, sampling: str = 'uniform') -> tuple[ResponseClosedLoopState | None, dict]:
    """Choose scenes, then a live boundary per scene; no survivor filter."""
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    n = record.valid.shape[1]
    if sampling == 'uniform':
        ids = torch.randperm(n, generator=generator)[:min(count,n)].tolist()
    elif sampling == 'mass_quantiles':
        # Cover the batch's mass range at the same probe count, including failed
        # scenes. This is auxiliary coverage, NOT balanced task/size updates.
        order = record.boundaries[0].physical.mass.detach().cpu().argsort(stable=True)
        bins = torch.tensor_split(order, min(count,n))
        ids = [int(group[torch.randint(group.numel(), (), generator=generator)]) for group in bins]
    else:
        raise ValueError('unknown contraction sampling')
    starts = sorted(t for t in record.boundaries if t+steps<=record.horizon)
    if not starts:
        raise ValueError('contraction interval exceeds episode horizon')
    chosen, metadata = [], []
    for idx in ids:
        available = [t for t in starts if bool(record.valid[t,idx])]
        if not available:
            continue
        t = available[int(torch.randint(len(available),(1,),generator=generator))]
        z = record.boundaries[t]
        row = torch.tensor([idx],device=z.physical.position.device)
        chosen.append(ResponseClosedLoopState(_select_rows(z.physical,row),_select_rows(z.policy,row)))
        metadata.append({'scene':idx,'start':t})
    if not chosen:
        return None, {'requested':len(ids),'selected':0,'initially_inactive':len(ids),'boundaries':[],
                      'sampling':sampling}
    combined = ResponseClosedLoopState(*(
        type(getattr(chosen[0],name))(**{
            f.name: torch.cat([getattr(getattr(c,name),f.name) for c in chosen],0)
            for f in fields(getattr(chosen[0],name))
        }) for name in ('physical','policy')
    ))
    return _detached(combined), {'requested':len(ids),'selected':len(chosen),
                                 'initially_inactive':len(ids)-len(chosen),'boundaries':metadata,
                                 'sampling':sampling}


def _direction_penalties(ratio, prefix_gain_squared, valid, config):
    """Endpoint metric objective plus a metric-independent transient budget."""
    endpoint = torch.relu(ratio.clamp_min(torch.finfo(ratio.dtype).eps).log()).square()
    real = torch.cat((torch.ones_like(valid[:1]), valid), 0)
    peak = torch.where(real, prefix_gain_squared, torch.zeros_like(prefix_gain_squared)).amax(0)
    prefix = torch.relu((peak/config.prefix_max_gain_squared).clamp_min(
        torch.finfo(peak.dtype).eps).log()).square()
    return endpoint + config.prefix_weight*prefix, endpoint, prefix, peak


def _reduce_penalties(penalties, fraction):
    if fraction == 1.0:
        return penalties.mean()
    return penalties.topk(max(1, math.ceil(fraction*penalties.numel()))).values.mean()


def contraction_loss(policy, simulator, metric, closed, config, *, seed, differentiable=True):
    """Directional penalty: V_end - V_start + rate*duration*||delta_task||^2.

    This is differential dissipativity/non-expansion with task contraction,
    NOT strict contraction of every state (heading-neutral modes are allowed).
    A direction terminating within the interval remains a reported failure;
    only its genuinely executed prefix is differentiated and penalized.
    """
    if metric.config != config:
        raise ValueError('metric and contraction configuration disagree')
    geo = StateGeometry(closed,policy.config.integral_limit)
    zero = closed.physical.position.new_zeros(closed.physical.position.shape[0],geo.tangent_dim)
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    context = geo.context(geo.base, config.context)
    terms, ratios, euclidean, transient = [], [], [], []
    endpoint_terms, prefix_terms, raw_transient = [], [], []
    lengths = None
    def function(delta):
        return local_flow(policy,simulator,geo.retract(delta),geo,config.steps)
    with torch.set_grad_enabled(differentiable):
        for direction_index in range(config.directions):
            direction = torch.randn(zero.shape,generator=generator,dtype=zero.dtype).to(zero.device)
            # Alternate full-state and plant/command probes: large hidden states
            # must not drown out physical directions. Both remain sampled checks.
            if direction_index % 2:
                direction[:,20:] = 0
            direction = direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            values, tangent, valid = torch.func.jvp(function,(zero,),(direction,),has_aux=True)
            if not bool(torch.isfinite(values).all() & torch.isfinite(tangent).all()):
                raise FloatingPointError('nonfinite true contraction trajectory or tangent')
            lengths = valid.sum(0)
            rows = torch.arange(zero.shape[0],device=zero.device)
            end = values[lengths,rows]
            end_tangent = tangent[lengths,rows]
            start_energy = metric.energy(values[0],context,tangent[0])
            end_energy = metric.energy(end,context,end_tangent)
            task_energy = geo.task_direction(tangent[0]).square().sum(-1)
            duration = lengths*simulator.params.dt
            ratio = (end_energy+config.rate*duration*task_energy)/start_energy
            active = lengths > 0
            raw_gains = tangent.square().sum(-1)/tangent[0].square().sum(-1)[None, :]
            penalty, endpoint_penalty, prefix_penalty, raw_peak = _direction_penalties(
                ratio, raw_gains, valid, config)
            terms.append(penalty[active])
            endpoint_terms.append(endpoint_penalty.detach()[active])
            prefix_terms.append(prefix_penalty.detach()[active])
            raw_transient.append(raw_peak.detach()[active])
            ratios.append(ratio.detach()[active])
            raw = end_tangent.square().sum(-1)/tangent[0].square().sum(-1)
            euclidean.append(raw.detach()[active])
            # Examine ALL real prefixes, never repeated/frozen padding.
            with torch.no_grad():
                energies = metric.energy(values,context.expand(values.shape[0],-1,-1),tangent)
                gains = energies/start_energy[None,:]
            real = torch.cat((torch.ones_like(valid[:1]),valid),0)
            transient.append(torch.where(real,gains,torch.zeros_like(gains)).amax(0).detach()[active])
    collected = torch.cat(terms)
    loss = _reduce_penalties(collected, config.tail_fraction) if collected.numel() else sum(p.sum()*0 for p in metric.parameters())
    r, raw, peak = torch.cat(ratios),torch.cat(euclidean),torch.cat(transient)
    # Inspect the actual terminal state; failure on the last step is still failure.
    with torch.no_grad():
        final = rollout(policy,simulator,geo.base,config.steps,time_decay=0.0)
        failed = simulator.terminated(final.end.physical)
        # Same equations need not round identically across fused/unfused backends.
        # Never silently certify a different trajectory or termination branch.
        if not torch.equal(final.valid, valid):
            raise RuntimeError('GRU differentiation backend changed termination mask')
        native_end = geo.pack(final.end)
        backend_error = float((native_end-values[-1].detach()).abs().max())
        atol, rtol = (1e-5, 2e-4) if native_end.dtype == torch.float32 else (1e-10, 1e-9)
        if not torch.allclose(native_end, values[-1].detach(), atol=atol, rtol=rtol):
            raise RuntimeError('GRU differentiation backend changed local trajectory')
    report = {
        'version':CONTRACTION_VERSION,'certified':False,'loss':float(loss.detach()),
        'gru_backend_max_error':backend_error,
        'scenes':zero.shape[0],'evaluated_directions':r.numel(),
        'complete_intervals':int(((lengths==config.steps)&~failed).sum()),
        'terminal_intervals':int(failed.sum()),'inactive_intervals':int((lengths==0).sum()),
        'mean_ratio':float(r.mean()) if r.numel() else None,
        'max_ratio':float(r.max()) if r.numel() else None,
        'direction_violation_fraction':float((r>1).double().mean()) if r.numel() else None,
        'max_euclidean_gain_squared':float(raw.max()) if raw.numel() else None,
        'max_prefix_metric_gain_squared':float(peak.max()) if peak.numel() else None,
        'max_prefix_euclidean_gain_squared':float(torch.cat(raw_transient).max()) if r.numel() else None,
        'endpoint_loss_mean':float(torch.cat(endpoint_terms).mean()) if r.numel() else None,
        'prefix_loss_mean':float(torch.cat(prefix_terms).mean()) if r.numel() else None,
        'tail_fraction':config.tail_fraction,
        'context':config.context,'fusion':config.fusion,
        'true_forward_transitions':int(lengths.sum())*(config.directions+1),
    }
    if not math.isfinite(report['loss']):
        raise FloatingPointError('nonfinite contraction loss')
    return loss,report


def worst_direction(policy, simulator, metric, closed, config):
    """Exact local generalized eigenproblem for ONE state, not a region proof.

    Intended for bounded audits, not every production update. The full legal
    tangent basis is used; no heading or controller-history row is discarded.
    """
    if closed.physical.position.shape[0] != 1:
        raise ValueError('worst-direction audit expects one scene')
    if bool(simulator.terminated(closed.physical).any()):
        raise ValueError('cannot audit an already terminal state')
    geo = StateGeometry(closed,policy.config.integral_limit)
    zero = closed.physical.position.new_zeros(geo.tangent_dim)
    def endpoints(d):
        values,_ = local_flow(policy,simulator,geo.retract(d[None,:]),geo,config.steps)
        return values[0,0],values[-1,0]
    with torch.enable_grad():
        C,A = torch.autograd.functional.jacobian(endpoints,zero,create_graph=False)
    with torch.no_grad():
        values,valid = local_flow(policy,simulator,geo.base,geo,config.steps)
        elapsed = int(valid.sum())*simulator.params.dt
        context = geo.context(geo.base, config.context)
        m0 = metric.matrix(values[0],context)[0]
        m1 = metric.matrix(values[-1],context)[0]
        E = geo.task_direction(C.T).T
        h0 = C.T@m0@C
        h1 = A.T@m1@A + config.rate*elapsed*(E.T@E)
        factor = torch.linalg.cholesky((h0+h0.T)*.5)
        tmp = torch.linalg.solve_triangular(factor,h1,upper=False)
        normalized = torch.linalg.solve_triangular(factor,tmp.T,upper=False).T
        eigenvalues,vectors = torch.linalg.eigh((normalized+normalized.T)*.5)
        direction = torch.linalg.solve_triangular(factor.T,vectors[:,-1:],upper=True).squeeze(-1)
        direction = direction/direction.norm()
        eigen = float(eigenvalues[-1])
        if not math.isfinite(eigen):
            raise FloatingPointError('nonfinite worst-direction audit')
    return {'max_ratio':eigen,'direction':direction,'certified':False,
            'true_interval_steps':int(valid.sum())}
