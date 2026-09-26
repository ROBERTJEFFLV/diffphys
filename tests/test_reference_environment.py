import math
import pytest
import torch
from env_raptor import RaptorSimulator,RaptorParams
from response_noise import DisturbanceConfig,executed_command

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


@pytest.mark.parametrize('noisy',[False,True])
def test_rk4_matches_independent_numpy(noisy):
    sim=RaptorSimulator();s=sim.reset(1,seed=230,horizon=25,dtype=torch.float64,
             disturbances=DisturbanceConfig(budget=.1 if noisy else 0,pool=(0,0,0,0,1)))
    for i in range(25):
        a=s.motor.new_tensor([[.15*math.sin(i),.12,-.2,.35]])
        reference=_numpy_step(s,executed_command(s,a))
        s=sim.step(s,a)
        actual=torch.cat([s.position[0],s.velocity[0],s.orientation[0],s.omega[0],s.motor[0]])
        torch.testing.assert_close(actual,reference,rtol=2e-12,atol=2e-12)


@pytest.mark.parametrize('noisy',[False,True])
def test_actual_motor_action_jacobian_matches_finite_differences(noisy):
    sim=RaptorSimulator();s=sim.reset(2,seed=7,horizon=2,dtype=torch.float64,
                         disturbances=DisturbanceConfig(budget=.1 if noisy else 0))
    action=s.motor.new_full(s.motor.shape,.15).requires_grad_()
    def output(u):
        out=sim.step(s,u)
        return torch.cat((out.position,out.velocity,out.orientation,out.omega,out.motor),-1)
    assert torch.autograd.gradcheck(output,(action,),eps=1e-6,atol=1e-5,rtol=1e-3)


def test_joint_physical_distribution_and_axis_authorities():
    s=RaptorSimulator().reset(4096,seed=7,horizon=1,dtype=torch.float64)
    assert .02<=s.mass.min()<=s.mass.max()<=5
    assert 1.5<=s.thrust_to_weight.min()<=s.thrust_to_weight.max()<=5
    assert 40<=s.torque_to_inertia.min()<=s.torque_to_inertia.max()<=1200
    assert .03<=s.motor_time_rising.min()<=s.motor_time_rising.max()<=.1
    assert .03<=s.motor_time_falling.min()<=s.motor_time_falling.max()<=.3
    thrust=RaptorSimulator.thrust(s,torch.ones_like(s.motor))
    torch.testing.assert_close(thrust.sum(-1)/(s.mass*9.81),s.thrust_to_weight)
    torch.testing.assert_close(s.arm_length*thrust[:,0]/s.inertia[:,0],s.torque_to_inertia)
    assert (s.motor_time_falling<s.motor_time_rising).any()
    root=s.mass.pow(1/3)
    assert abs(float(root.mean())-(.02**(1/3)+5**(1/3))/2)<.025


def test_only_reference_frequency_allowed():
    with pytest.raises(ValueError,match='100 Hz'):RaptorParams(.02)
