from dataclasses import fields, replace
import math

import pytest
import torch

from env_raptor import RaptorSimulator
from response_noise import DisturbanceConfig, capacities, sample_bounded, measured_observation, executed_command, REFERENCE_TIME
from response_task import _select_rows, rollout
from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_training import sample_pool


@pytest.mark.parametrize('budget', [-1., .100001, float('nan'), float('inf')])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError): DisturbanceConfig(budget=budget)


@pytest.mark.parametrize('pool', [(0,0,0,0,0), (1,2), (1,1,1,1,-1), (1,1,1,1,float('nan'))])
def test_invalid_pool_rejected(pool):
    with pytest.raises(ValueError): DisturbanceConfig(pool=pool)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_exact_joint_budget_bounds_and_feasible_trim(dtype):
    sim = RaptorSimulator()
    s = sim.reset(2048, seed=17, horizon=8, dtype=dtype,
                  disturbances=DisturbanceConfig(pool=(0,0,0,0,1)))
    cap = capacities(s)
    rho = s.noise_fraction.double()
    assert (rho >= 0).all() and (rho.sum(-1) <= .1).all()
    assert ((s.noise_tape.abs() <= s.noise_bounds[:, None, :])).all()
    assert (s.velocity_delay >= 0).all() and (s.velocity_delay <= .003).all()
    assert s.external_torque.abs().sum() > 0
    torch.testing.assert_close(cap['allocation'] @ cap['inverse'],
                               torch.eye(4, dtype=torch.float64).expand(2048,4,4), atol=1e-12, rtol=1e-12)
    # All axes, collective, and action uncertainty share the SAME rotor margins.
    gravity = torch.zeros_like(s.external_force.double()); gravity[:,2] = s.mass.double()*9.81
    total = (gravity-s.external_force.double()).norm(dim=-1)
    wrench = torch.cat((total[:,None], -s.external_torque.double()), -1)
    trim = (cap['inverse'] @ wrench[...,None]).squeeze(-1)
    normalized = (trim-cap['hover']).abs()/cap['reserve']
    assert (normalized <= (rho[:,0]+rho[:,1])[:,None] + 2e-12).all()
    for t in (0,3,7):
        state = replace(s, step_index=torch.full_like(s.step_index,t))
        u = sample_bounded(torch.ones_like(s.motor), torch.Generator().manual_seed(t))
        actual = executed_command(state, u)
        delta = (sim.thrust(s, sim.motor_command(s, actual))-
                 sim.thrust(s, sim.motor_command(s, u))).abs().double()
        assert ((delta/cap['reserve']) <= rho[:,2,None]+2e-6).all()
    assert (normalized + rho[:,2,None] <= .1+1e-12).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_reference_sensor_and_delay_loads_obey_the_joint_bound(dtype):
    s = RaptorSimulator().reset(256, seed=3, horizon=4, dtype=dtype,
          disturbances=DisturbanceConfig(pool=(0,0,0,0,1)))
    c = capacities(s); rho = s.noise_fraction.double(); eps = s.noise_tape[:,0,:12].double()
    force = c['force'][:,0]*math.sqrt(3)
    kp,kv = 1/REFERENCE_TIME**2,2/REFERENCE_TIME
    loads = torch.zeros_like(rho)
    loads[:,0] = s.external_force.double().norm(dim=-1)/force
    def torque_load(torque):
        return ((c['inverse'][...,1:] @ torque[...,None]).squeeze(-1).abs()/c['reserve']).amax(-1)
    loads[:,1] = torque_load(s.external_torque.double())
    loads[:,2] = rho[:,2]  # Analytic derivative bound for all command/tape values.
    loads[:,3] = s.mass.double()*kp*eps[:,:3].norm(dim=-1)/force
    loads[:,4] = s.mass.double()*kv*eps[:,3:6].norm(dim=-1)/force
    fmax = RaptorSimulator.thrust(s,torch.ones_like(s.motor)).sum(-1).double()
    loads[:,5] = (torque_load(s.inertia.double()*kp*eps[:,6:9]) +
                  fmax*eps[:,6:9].norm(dim=-1)/force)
    loads[:,6] = torque_load(s.inertia.double()*kv*eps[:,9:12])
    loads[:,7] = s.mass.double()*kv*c['acceleration']*s.velocity_delay.double()/force
    assert (loads <= rho+2e-7).all()
    assert (loads.sum(-1) <= .1+2e-7).all()
    # No own-mass scaling of a sensor: the unit-percent sensor bounds are common.
    for i,section in enumerate((slice(0,3),slice(3,6),slice(6,9),slice(9,12))):
        unit = s.noise_bounds[:,section].double()/rho[:,3+i,None]
        torch.testing.assert_close(unit,unit[0].expand_as(unit),atol=2e-6,rtol=2e-6)


def test_airframe_and_initial_state_rng_independent_of_noise_config():
    sim = RaptorSimulator(); before=torch.get_rng_state().clone()
    a=sim.reset(64,seed=88,horizon=8,disturbances=DisturbanceConfig(budget=0))
    b=sim.reset(64,seed=88,horizon=8)
    c=sim.reset(64,seed=88,horizon=8)
    assert torch.equal(before,torch.get_rng_state())
    nuisance={'external_force','external_torque','noise_tape','rotation_tape','noise_bounds','noise_fraction','velocity_delay'}
    for f in fields(a):
        if f.name not in nuisance: assert torch.equal(getattr(a,f.name),getattr(b,f.name)),f.name
        assert torch.equal(getattr(b,f.name),getattr(c,f.name)),f.name
    assert a.noise_tape.shape==(64,1,16)
    assert not a.external_force.any() and not a.external_torque.any()


def test_pool_weights_can_request_zero_or_full_budget_without_airframe_groups():
    for pool,expected in [((1,0,0,0,0),0.),((0,0,0,0,1),.1)]:
        s=RaptorSimulator().reset(1024,seed=71,horizon=1,disturbances=DisturbanceConfig(pool=pool))
        torch.testing.assert_close(s.noise_fraction.sum(-1),s.mass.new_full((1024,),expected),atol=1e-6,rtol=1e-5)


def test_one_sampler_handles_static_and_temporal_bounds():
    bounds=torch.tensor([[1.,2.,3.],[.1,.2,.3]],dtype=torch.float64)
    a=sample_bounded(bounds,torch.Generator().manual_seed(12))
    b=sample_bounded(bounds,torch.Generator().manual_seed(12),steps=1)[:,0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert (a.abs()<=bounds).all()


def test_so3_measurements_replay_and_do_not_mutate_truth():
    s=RaptorSimulator().reset(64,seed=2,horizon=8,dtype=torch.float64)
    q=s.orientation.clone(); old=s.noise_tape.clone()
    a=measured_observation(s);b=measured_observation(s)
    assert torch.equal(a,b) and torch.equal(s.orientation,q) and torch.equal(s.noise_tape,old)
    r=a[:,6:15].reshape(-1,3,3)
    torch.testing.assert_close(r.transpose(1,2)@r,torch.eye(3,dtype=r.dtype).expand(64,3,3),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(torch.linalg.det(r),torch.ones(64,dtype=r.dtype),atol=1e-12,rtol=1e-12)
    changed=replace(s,mass=s.mass*5,inertia=s.inertia*3,motor=s.motor+.2,
                    external_force=s.external_force+10,external_torque=s.external_torque+4)
    assert torch.equal(a,measured_observation(changed))


def test_fractional_delay_uses_acquisition_time_noise_and_preserves_history_gradient():
    s=RaptorSimulator().reset(1,seed=2,horizon=4,dtype=torch.float64)
    tape=torch.zeros_like(s.noise_tape);tape[:,0,3:6]=.1;tape[:,1,3:6]=.2
    current=torch.full_like(s.velocity,2.,requires_grad=True)
    previous=torch.full_like(s.velocity,1.,requires_grad=True)
    s=replace(s,velocity=current,previous_velocity=previous,velocity_delay=s.mass.new_full((1,),.003),
              step_index=torch.ones_like(s.step_index),noise_tape=tape)
    observed=measured_observation(s)[:,3:6]
    torch.testing.assert_close(observed,torch.full_like(observed,.7*2.2+.3*1.1))
    grads=torch.autograd.grad(observed.sum(),(current,previous))
    torch.testing.assert_close(grads[0],torch.full_like(current,.7))
    torch.testing.assert_close(grads[1],torch.full_like(previous,.3))


def test_compaction_shares_tape_and_keeps_original_noise_row_ids():
    s=RaptorSimulator().reset(64,seed=7,horizon=8,dtype=torch.float64)
    ids=torch.tensor([3,7,61])
    compact=_select_rows(s,ids)
    assert compact.noise_tape.data_ptr()==s.noise_tape.data_ptr()
    assert compact.rotation_tape.data_ptr()==s.rotation_tape.data_ptr()
    assert compact.noise_row.tolist()==[3,7,61]
    torch.testing.assert_close(measured_observation(compact),measured_observation(s)[ids],rtol=0,atol=0)
    empty=_select_rows(compact,torch.tensor([],dtype=torch.long))
    assert empty.noise_tape.data_ptr()==s.noise_tape.data_ptr()
    assert empty.noise_row.numel()==0


def test_commands_not_hidden_execution_are_returned_to_actor_and_loss():
    sim=RaptorSimulator();s=sim.reset(8,seed=3,horizon=3,dtype=torch.float64,
                disturbances=DisturbanceConfig(pool=(0,0,0,0,1)))
    u=torch.zeros_like(s.motor)
    assert not torch.equal(executed_command(s,u),u)
    result=sim.step(s,u)
    assert torch.equal(result.previous_action,u)
    assert torch.equal(result.previous_velocity,s.velocity)
    assert torch.equal(measured_observation(result)[:,18:],u)
    assert torch.equal(result.external_torque,s.external_torque)
    assert torch.equal(result.external_force,s.external_force)


def test_pooled_sampler_transfers_one_shared_tape_and_is_repeatable():
    a=sample_pool(8,[31,32,33,34],horizon=8,dtype=torch.float64)
    b=sample_pool(8,[31,32,33,34],horizon=8,dtype=torch.float64)
    assert a.noise_row.tolist()==list(range(32))
    for f in fields(a): assert torch.equal(getattr(a,f.name),getattr(b,f.name))


def test_tape_horizon_guard():
    s=RaptorSimulator().reset(4,seed=7,horizon=3,dtype=torch.float64)
    p=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).double()
    with pytest.raises(ValueError,match='noise tape'): rollout(p,RaptorSimulator(),s,4)


def test_precomputed_attitude_tape_equals_exponential_and_no_tape_copy_at_reset():
    from response_noise import rotation_error
    s=RaptorSimulator().reset(32,seed=91,horizon=7,dtype=torch.float64)
    expected=rotation_error(s.noise_tape[:,:,6:9].reshape(-1,3)).reshape(32,8,3,3)
    torch.testing.assert_close(s.rotation_tape,expected,rtol=0,atol=0)


def test_worst_case_torque_box_corners_obey_rotor_margin():
    import itertools
    s=RaptorSimulator().reset(128,horizon=1,dtype=torch.float64)
    c=capacities(s)
    corners=torch.tensor(list(itertools.product((-1.,1.),repeat=3)),dtype=torch.float64)
    for sign in corners:
        torque=.1*c['torque']*sign
        load=(c['inverse'][...,1:]@torque[...,None]).squeeze(-1).abs()/c['reserve']
        assert (load<=.1+1e-12).all()


def test_selected_aircraft_metrics_do_not_broadcast_original_pool_tapes():
    from response_task import trajectory_metrics, TaskLossConfig
    s=RaptorSimulator().reset(32,seed=91,horizon=7,dtype=torch.float64)
    selected=_select_rows(s,torch.tensor([1,7,11]))
    p=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).double()
    trace=rollout(p,RaptorSimulator(),selected,7)
    result=trajectory_metrics(trace,TaskLossConfig())
    assert result['scenario_count']==3 and result['finite']
