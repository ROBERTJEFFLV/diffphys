"""The production B architecture retains the confirmed adapted loss."""
from dataclasses import fields
from pathlib import Path

import torch

from response_policy import ARCHITECTURE, ResponseMotorPolicy, ResponsePolicyConfig
from tools.train_response_control import parse_args


def test_mainline_keeps_only_the_original_gru_and_hidden_linear_readout():
    policy = ResponseMotorPolicy()
    assert ARCHITECTURE == 'gru16-hidden-only-readout-absolute-motor-policy-v1'
    assert [type(module) for module in policy.children()] == [torch.nn.GRUCell, torch.nn.Linear]
    assert (policy.readout.in_features, policy.readout.out_features) == (64, 4)
    assert sum(parameter.numel() for parameter in policy.parameters()) == 16004
    assert {field.name for field in fields(ResponsePolicyConfig)} == {'memory_dim', 'dt'}
    assert not hasattr(policy, 'base_feedback')


def test_gru_only_run_config_has_only_confirmed_loss_and_no_geometry_flags():
    root = Path(__file__).resolve().parents[1]
    args = parse_args(['@'+str(root/'configs/response_raptor_multi_airframe.args')])
    assert (args.epsilon_p, args.epsilon_a, args.lambda_R) == (.01, .01, .2)
    assert not any(hasattr(args, name) for name in ('residual_amplitude', 'residual_gain',
                                                  'horizontal_accel_limit', 'vertical_fraction',
                                                  'antipodal_epsilon'))
    assert str(args.work_dir) == 'runs/attitude_delta_gru_only_no_cvar/seed7'
