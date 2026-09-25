"""Reference environment regressions; no long training or external services."""
from dataclasses import fields, replace
import copy
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env_l2f import ENVIRONMENT_VERSION, L2FParams, L2FSimulator
from response_task import sample_scenarios, observation, rollout, TaskLossConfig, reference_episode_metrics
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_adjoints import collect_boundary_rollout, backward_actor


def make(protocol='raptor', n=64, horizon=20, seed=7, dtype=torch.float64):
    return sample_scenarios(n, seed=seed, dtype=dtype, scenario_mode=protocol, horizon=horizon)


@pytest.mark.parametrize('protocol', ['l2f', 'raptor'])
def test_initial_state_inside_own_boundary(protocol):
    s = make(protocol, 4096, horizon=1)
    assert ((s.position.abs() <= s.initial_position_limit[:, None])).all()
    assert ((s.position.abs() < s.position_limit[:, None])).all()
    assert s.velocity.abs().max() <= 1
    assert s.omega.abs().max() <= 1
    angles = 2 * s.orientation[:, 0].clamp(-1, 1).acos()
    assert angles.max() <= math.pi / 2 + 1e-12
    assert angles.max() > 1.5
    if protocol == 'raptor':
        torch.testing.assert_close(s.initial_position_limit, 10 * s.arm_length)
        torch.testing.assert_close(s.position_limit, 20 * s.arm_length)
        assert (s.velocity[s.guidance.bool()] == 0).all()
        assert (s.omega[s.guidance.bool()] == 0).all()
    assert 0.07 < s.guidance.double().mean() < 0.13


def test_hover_not_pre_normalized():
    s = make()
    r = L2FSimulator.thrust(s, L2FSimulator.motor_command(s, torch.zeros_like(s.motor))).sum(-1) / (s.mass * 9.81)
    assert r.std() > 0.05
    assert not torch.allclose(r, torch.ones_like(r))
    assert (s.thrust_coefficients[..., 2] > 0).all()
    assert s.motor.std() > 0.05
    assert s.motor.min() >= 0 and s.motor.max() <= 0.5


def test_x_mixer_signs():
    s = make('l2f', 1)
    F = torch.eye(4, dtype=s.position.dtype)
    # +roll torques from left rotors; +pitch from rear rotors.
    expected = torch.tensor([[-1., -1., -1.], [-1., 1., 1.],
                             [1., 1., -1.], [1., -1., 1.]])
    s = make('l2f', 4)
    tau = L2FSimulator.body_torque(s, F)
    torch.testing.assert_close(tau.sign(), expected.to(tau))


def test_noise_profiles_and_consistency():
    for profile in ['l2f', 'raptor']:
        s = make(profile, 256, horizon=20)
        expected = s.position.new_tensor([0.001]*3 + [0.002]*3 + [0.001]*9 + [0.002]*3)
        if profile == 'l2f':
            torch.testing.assert_close(s.noise_std, expected.expand(256, -1))
            torch.testing.assert_close(s.noise_tape.std((0,1)), expected, atol=1e-4, rtol=0.08)
            assert s.external_force.abs().max() > 0
            assert s.external_torque.abs().max() > 0
        else:
            assert (s.noise_tape == 0).all()
            assert (s.external_torque == 0).all()
            assert s.external_force.abs().max() > 0
        obs = observation(s)
        torch.testing.assert_close(obs, observation(s), rtol=0, atol=0)
        # Memory is built from noisy observations, never privileged motor states.
        p = ResponseMotorPolicy().double()
        hidden = p.initial_state(obs).memory
        assert hidden.shape == (256, p.config.memory_dim) and not hidden.any()
        expected_hidden = p.response_memory(p.control_features(obs), hidden)
        torch.testing.assert_close(p(obs).next_state.memory, expected_hidden, rtol=0, atol=0)


@pytest.mark.parametrize('protocol', ['l2f', 'raptor'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_sampler_determinism_and_rng_isolation(protocol, dtype):
    rng = torch.get_rng_state().clone()
    a, b = make(protocol, dtype=dtype), make(protocol, dtype=dtype)
    assert torch.equal(rng, torch.get_rng_state())
    for f in fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        assert torch.equal(x, y), f.name
        assert x.device.type == 'cpu'
        if x.is_floating_point():
            assert x.dtype == dtype
    assert not torch.equal(a.position, make(protocol, seed=8, dtype=dtype).position)


def test_joint_distribution_and_airframe_scaling():
    s = make('raptor', 16000, horizon=1)
    root = s.mass.pow(1/3)
    assert root.min() >= 0.02**(1/3) and root.max() <= 5**(1/3)
    assert abs(float(root.mean()) - (0.02**(1/3)+5**(1/3))/2) < .012
    assert 1.5 <= s.thrust_to_weight.min() <= s.thrust_to_weight.max() <= 5
    assert 40 <= s.torque_to_inertia.min() <= s.torque_to_inertia.max() <= 1200
    assert 0.03 <= s.motor_time_rising.min() <= s.motor_time_rising.max() <= 0.1
    assert 0.03 <= s.motor_time_falling.min() <= s.motor_time_falling.max() <= 0.3
    assert (s.motor_time_falling < s.motor_time_rising).any()  # No fake fall>=rise constraint.
    base_j = s.inertia.new_tensor((9.416556729130406e-6, 9.644051701582312e-6, 1.745951732253285e-5))
    torch.testing.assert_close(s.inertia[:,1]/s.inertia[:,0], (base_j[1]/base_j[0]).expand(16000))
    torch.testing.assert_close(s.inertia[:,2]/s.inertia[:,0], (base_j[2]/base_j[0]).expand(16000))
    thrust = L2FSimulator.thrust(s, torch.ones_like(s.motor))
    torch.testing.assert_close(thrust.sum(-1)/(s.mass*9.81), s.thrust_to_weight)
    # Upstream's torque/inertia root is sqrt(2)*xy*F/Jx, not the two-rotor max roll authority.
    torch.testing.assert_close(s.arm_length*thrust[:,0]/s.inertia[:,0], s.torque_to_inertia)
    assert (s.force_std <= .3*(s.thrust_to_weight-1)*s.thrust_to_weight*s.mass/3 + 1e-12).all()


@pytest.mark.parametrize('protocol', ['l2f','raptor'])
def test_rotations_and_motor_endpoints(protocol):
    s = make(protocol)
    R = s.rotation
    torch.testing.assert_close(R.transpose(1,2)@R, torch.eye(3,dtype=R.dtype).expand(len(R),3,3))
    torch.testing.assert_close(torch.linalg.det(R), torch.ones(len(R),dtype=R.dtype))
    for a in [-1,1]:
        command = torch.full_like(s.motor, a)
        actual = L2FSimulator.motor_command(s, command)
        target = (s.motor_min if a == -1 else s.motor_max)[:,None].expand_as(actual)
        torch.testing.assert_close(actual, target)


def _numpy_step(s, action):
    """Independent scalar/NumPy port of upstream RK4; does not call our RHS."""
    import numpy as np
    def arr(name):
        return getattr(s,name).detach().numpy()[0]
    a = np.clip(action.detach().numpy()[0], -1,1)
    target = arr('motor_min')+(a+1)/2*(arr('motor_max')-arr('motor_min'))
    z = np.concatenate([arr('position'), arr('velocity'), arr('orientation'), arr('omega'), arr('motor')])
    def rhs(z):
        p,v,q,w,m = z[:3],z[3:6],z[6:10],z[10:13],z[13:17]
        th = (arr('thrust_coefficients')*np.stack([np.ones(4),m,m*m],-1)).sum(-1)
        tf = np.zeros((4,3));tf[:,2]=th
        torque = np.cross(arr('rotor_positions'),tf).sum(0)
        torque[2] += (np.array([-1,1,-1,1])*arr('rotor_torque_constant')*th).sum()
        thrust = tf.sum(0)
        # Upstream rotate_vector_by_quaternion uses this double-cross form.
        t = 2*np.cross(q[1:],thrust)
        thrust_world = thrust + q[0]*t + np.cross(q[1:],t)
        acc = thrust_world/arr('mass')+np.array([0,0,-9.81])+arr('external_force')/arr('mass')
        qdot = 0.5*np.concatenate([[-np.dot(q[1:],w)],q[0]*w+np.cross(q[1:],w)])
        wdot = (torque+arr('external_torque')-np.cross(w,arr('inertia')*w))/arr('inertia')
        tau = np.where(target>=m,arr('motor_time_rising'),arr('motor_time_falling'))
        return np.concatenate([v,acc,qdot,wdot,(target-m)/tau])
    k1=rhs(z);k2=rhs(z+.005*k1);k3=rhs(z+.005*k2);k4=rhs(z+.01*k3)
    out=z+.01/6*(k1+2*k2+2*k3+k4)
    out[6:10]/=np.linalg.norm(out[6:10])
    out[13:17]=np.clip(out[13:17],arr('motor_min'),arr('motor_max'))
    return torch.from_numpy(out)


@pytest.mark.parametrize('protocol', ['l2f','raptor'])
def test_rk4_matches_independent_reference(protocol):
    s=make(protocol,1,seed=230)
    sim=L2FSimulator(L2FParams(protocol=protocol))
    for i in range(25):
        a=s.motor.new_tensor([[.15*math.sin(i), .12, -.2, .35]])
        reference=_numpy_step(s,a)
        s=sim.step(s,a)
        actual=torch.cat([s.position[0],s.velocity[0],s.orientation[0],s.omega[0],s.motor[0]])
        torch.testing.assert_close(actual,reference,rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('protocol', ['l2f','raptor'])
def test_action_gradient_finite_difference(protocol):
    s=make(protocol,2)
    sim=L2FSimulator(L2FParams(protocol=protocol))
    a=s.motor.new_full(s.motor.shape,.15).requires_grad_()
    def fn(a):
        out=sim.step(s,a)
        return torch.cat((out.position, out.velocity, out.orientation, out.omega,
                          out.motor/s.motor_max[:,None]),-1)
    assert torch.autograd.gradcheck(fn,(a,),eps=1e-6,atol=1e-5,rtol=1e-3)


@pytest.mark.parametrize('protocol', ['l2f','raptor'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_full_and_windowed_bptt_share_noise_and_gradients(protocol,dtype):
    torch.manual_seed(7)
    config=ResponsePolicyConfig(memory_dim=8)
    policies=[ResponseMotorPolicy(config).to(dtype=dtype)]
    policies.append(copy.deepcopy(policies[0]))
    s=make(protocol,8,horizon=20,dtype=dtype)
    sim=L2FSimulator(L2FParams(protocol=protocol)); loss=TaskLossConfig()
    records=[];grads=[]
    for policy,mode in zip(policies,['full','windowed']):
        r=collect_boundary_rollout(policy,sim,s,loss,horizon=20,window_steps=5,backprop_mode=mode)
        checks=backward_actor(policy,sim,r,loss)
        records.append(r)
        grads.append(torch.cat([p.grad.flatten() for p in policy.parameters() if p.grad is not None]))
        if mode=='windowed':
            assert len(checks['boundaries'])==4
            assert max(x['max_error'] for x in checks['boundaries'])==0
    torch.testing.assert_close(records[0].costs,records[1].costs,rtol=0,atol=0)
    tol=2e-5 if dtype==torch.float32 else 1e-10
    torch.testing.assert_close(grads[0],grads[1],rtol=tol,atol=tol)
    assert records[1].boundaries[0].physical.noise_tape.data_ptr()==s.noise_tape.data_ptr()
    for p in policies:
        torch.optim.Adam(p.parameters(),lr=3e-4).step()
    for x,y in zip(policies[0].parameters(),policies[1].parameters()):
        torch.testing.assert_close(x,y,rtol=tol,atol=tol)


def test_noise_horizon_and_protocol_errors():
    s=make('l2f',1,horizon=3)
    p=ResponseMotorPolicy().double()
    with pytest.raises(ValueError,match='noise tape'):
        rollout(p,L2FSimulator(L2FParams(protocol='l2f')),s,4)
    with pytest.raises(ValueError,match='protocol'):
        rollout(p,L2FSimulator(),s,2)
    with pytest.raises(ValueError,match='100 Hz'):
        L2FParams(dt=.02)


def test_reference_metrics_use_per_airframe_thresholds():
    s=make('raptor',2,horizon=3)
    s=replace(s,position_limit=s.mass.new_tensor([.6,4]),initial_position_limit=s.mass.new_tensor([.3,2]),position=torch.zeros_like(s.position))
    p=ResponseMotorPolicy().double()
    t=rollout(p,L2FSimulator(),s,3)
    positions=torch.zeros_like(t.positions);positions[0,:,0]=.8
    report=reference_episode_metrics(replace(t,positions=positions,velocities=torch.zeros_like(t.velocities),omegas=torch.zeros_like(t.omegas)))
    assert report['raptor_share_terminated']==.5
    assert report['raptor_episode_length_mean']==2  # 1 and 3, despite later recovery.
    assert not any(k.startswith('l2f_') for k in report)
    bad=replace(s,position=torch.ones_like(s.position)*10)
    with pytest.raises(ValueError,match='starts outside'):
        reference_episode_metrics(replace(t,initial=bad))


def test_native_motor_action_has_no_default_slew_projection():
    p=ResponseMotorPolicy().double()
    for x in p.parameters():
        x.data.zero_()
    p.readout.bias.data.fill_(10)
    s=make('raptor',1)
    obs=observation(s);obs[:,18:22]=-1
    assert (p(obs).action>.99).all()
