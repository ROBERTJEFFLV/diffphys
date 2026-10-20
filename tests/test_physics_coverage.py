"""Fixed physical TRAIN coverage without modifying dynamics or fixed EVAL."""

from loss_fixtures import test_loss
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from response_training import sample_training_scenarios
from loss_fixtures import parse_loss_args as parse_args


def test_production_config_selects_2048_coverage_without_changing_eval():
    root = Path(__file__).resolve().parents[1]
    args = parse_args(['@' + str(root / 'configs/response_raptor_multi_airframe.args')])
    assert getattr(args, 'train_sampling', None) == 'coverage128'
    assert args.scenarios == 512 and args.eval_scenarios == 128
    assert args.group_max_groups == 128


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_all_128_cells_have_exactly_16_distinct_airframes(dtype):
    from response_sampling import physics_cell_ids
    report = {}
    state, _ = sample_training_scenarios(512, 0, dtype=dtype, horizon=5,
                                        sampling='coverage128', sampling_report=report)
    assert torch.bincount(physics_cell_ids(state), minlength=128).tolist() == [16] * 128
    assert report['cell_counts'] == [16] * 128
    physical = torch.stack((state.mass, state.torque_to_inertia,
                            state.motor_time_rising[:, 0], state.motor_time_falling[:, 0]), -1)
    assert torch.unique(physical, dim=0).shape[0] == 2048
    assert state.noise_tape.shape == (2048, 6, 12)
    assert torch.equal(state.noise_row, torch.arange(2048))


def assert_state_equal(a, b, excluded=()):
    for field in fields(a):
        if field.name not in excluded:
            torch.testing.assert_close(getattr(a, field.name), getattr(b, field.name), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_yaw_measure_uses_thrust_and_inertia_not_extra_arm_division(dtype):
    from dataclasses import replace
    from env_raptor import RaptorSimulator
    from response_noise import DisturbanceConfig
    from response_sampling import yaw_authority
    s = RaptorSimulator().reset(256, seed=201, dtype=dtype, disturbances=DisturbanceConfig.clean())
    actual = yaw_authority(s)
    expected = (s.torque_to_inertia.double() * s.rotor_torque_constant[:, 0].double()
                / s.arm_length.double() * s.inertia[:, 0].double() / s.inertia[:, 2].double())
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-10)
    # Doubling all moments with all thrusts retained halves yaw authority.
    torch.testing.assert_close(yaw_authority(replace(s, inertia=2*s.inertia)), actual/2)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_every_cell_id_and_boundary_convention(dtype, monkeypatch):
    import itertools
    from dataclasses import replace
    from env_raptor import RaptorSimulator
    from response_noise import DisturbanceConfig
    import response_sampling as sampling
    definition = sampling.sampling_contract('coverage128')
    grid = list(itertools.product(range(4), range(4), range(4), range(2)))
    s = RaptorSimulator().reset(128, seed=5, dtype=dtype, disturbances=DisturbanceConfig.clean())
    tti = [40., 330., 620., 910.]
    rise = [.03, .0475, .065, .0825]
    fall = [.03, .0975, .165, .2325]
    s = replace(s, torque_to_inertia=s.mass.new_tensor([tti[i] for i,_,_,_ in grid]),
                motor_time_rising=s.mass.new_tensor([rise[j] for _,j,_,_ in grid])[:,None].repeat(1,4),
                motor_time_falling=s.mass.new_tensor([fall[k] for _,_,k,_ in grid])[:,None].repeat(1,4))
    authority = torch.tensor([definition['yaw_medians_by_tti_bin'][i]*(.5 if y==0 else 1.)
                              for i,_,_,y in grid], dtype=torch.float64)
    monkeypatch.setattr(sampling, 'yaw_authority', lambda state: authority)
    assert torch.equal(sampling.physics_cell_ids(s), torch.arange(128))


def test_coverage_keeps_original_candidate_rows_and_clean_physical_coupling(monkeypatch):
    from dataclasses import replace
    import response_sampling as sampling
    from env_raptor import RaptorSimulator
    captured = []
    original = RaptorSimulator.reset
    def recording_reset(self, *args, **kwargs):
        assert kwargs['horizon'] == 1 and not kwargs['disturbances'].enabled
        state = original(self, *args, **kwargs)
        captured.append(state)
        return state
    monkeypatch.setattr(RaptorSimulator, 'reset', recording_reset)
    s, report = sampling.sample_coverage(2048, (1,2,3,4), dtype=torch.float64)
    # Mass is unique in this deterministic fixture; locate the unchanged rows.
    candidate_mass = torch.cat([state.mass for state in captured])
    sorted_mass, order = candidate_mass.sort()
    assert sorted_mass.unique().numel() == sorted_mass.numel()
    rows = order[torch.searchsorted(sorted_mass, s.mass)]
    for field in fields(s):
        if field.name == 'noise_row':
            continue
        expected = torch.cat([getattr(state, field.name) for state in captured])[rows]
        torch.testing.assert_close(getattr(s, field.name), expected, rtol=0, atol=0)
    assert torch.equal(s.noise_row, torch.arange(2048))
    assert report['candidates_drawn'] == candidate_mass.numel()
    assert torch.equal(s.previous_action, torch.zeros_like(s.previous_action))
    assert len(report['candidate_seeds']) == len(set(report['candidate_seeds']))
    # Sorting/shuffling must not introduce new hidden state fields into the Actor.
    from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
    from response_noise import measured_observation
    p = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).double()
    obs = measured_observation(s)
    changed = replace(s, torque_to_inertia=s.torque_to_inertia+100, inertia=s.inertia*2,
                      rotor_torque_constant=s.rotor_torque_constant*3)
    assert torch.equal(p(obs).action, p(measured_observation(changed)).action)


@pytest.mark.parametrize('count', [128, 512, 2048])
def test_repeatability_refresh_and_no_global_rng_consumption(count):
    import response_sampling as sampling
    from response_noise import DisturbanceConfig
    report_a, report_b = {}, {}
    torch.manual_seed(21)
    before = torch.get_rng_state().clone()
    a, seeds = sample_training_scenarios(count//4, 12, horizon=6, sampling='coverage128', sampling_report=report_a)
    assert torch.equal(before, torch.get_rng_state())
    torch.rand(31)  # Global RNG calls cannot alter this sampler.
    b, _ = sample_training_scenarios(count//4, 12, horizon=6, sampling='coverage128', sampling_report=report_b)
    assert_state_equal(a,b)
    assert report_a == report_b
    assert torch.bincount(sampling.physics_cell_ids(a), minlength=128).tolist() == [count//128]*128
    c, _ = sample_training_scenarios(count//4, 13, horizon=6, sampling='coverage128')
    assert not torch.equal(a.mass,c.mass)
    clean, _ = sample_training_scenarios(count//4, 12, horizon=12, sampling='coverage128',
                                        disturbances=DisturbanceConfig.clean())
    for name in ('mass','inertia','position','velocity','orientation','omega','motor','previous_action',
                 'rotor_positions','thrust_coefficients','rotor_torque_constant','motor_time_rising','motor_time_falling'):
        assert torch.equal(getattr(a,name),getattr(clean,name))
    assert all(seed >= 1 << 62 for seed in report_a['candidate_seeds'])


def test_noise_attached_once_after_selection_and_original_row_ids_survive_compaction(monkeypatch):
    import response_training as training
    from response_noise import pulse_at
    from response_task import _select_rows
    calls = []
    original = training.attach_disturbances
    def attach(state, config, **kwargs):
        calls.append((state.mass.numel(), state.noise_tape.shape[1], kwargs['horizon']))
        return original(state, config, **kwargs)
    monkeypatch.setattr(training, 'attach_disturbances', attach)
    state, _ = sample_training_scenarios(512, 5, horizon=500, sampling='coverage128')
    assert calls == [(2048,1,500)]
    assert state.noise_tape.shape == (2048,501,12)
    assert state.pulse_tape.shape == (2048,501,6)
    ids = torch.tensor([93,0,2047,301])
    selected = _select_rows(state,ids)
    assert selected.pulse_tape.data_ptr() == state.pulse_tape.data_ptr()
    for part, full in zip(pulse_at(selected), pulse_at(state)):
        torch.testing.assert_close(part,full[ids],rtol=0,atol=0)


def test_original_random_and_eval_pools_do_not_change():
    from response_training import sample_pool, DEVELOPMENT_SEEDS
    a, seeds = sample_training_scenarios(8, 2, horizon=4)
    b = sample_pool(8,seeds,horizon=4)
    assert_state_equal(a,b)
    before = sample_pool(4,DEVELOPMENT_SEEDS,horizon=4)
    sample_training_scenarios(32, 2, horizon=4, sampling='coverage128')
    after = sample_pool(4,DEVELOPMENT_SEEDS,horizon=4)
    assert_state_equal(before,after)
    assert parse_args(['--scenarios','8']).train_sampling == 'random'


def test_legacy_adaptive_api_still_available_but_production_uses_fixed_cells():
    from response_groups import GroupBalanceConfig, physics_group_layout
    state, _ = sample_training_scenarios(512,0,horizon=1,sampling='coverage128')
    _, counts = physics_group_layout(state, GroupBalanceConfig())
    assert counts.tolist() == [128]*16


@pytest.mark.parametrize('scenarios', [1,8,31,33,511])
def test_invalid_coverage_size_fails_before_writing(scenarios, tmp_path):
    with pytest.raises(SystemExit):
        parse_args(['--train-sampling','coverage128','--scenarios',str(scenarios),'--work-dir',str(tmp_path/'bad')])
    assert not (tmp_path/'bad').exists()
    with pytest.raises(ValueError,match='divisible by 128'):
        sample_training_scenarios(scenarios,0,sampling='coverage128')


def test_incomplete_coverage_raises_instead_of_duplicate_or_random_fallback(monkeypatch):
    import response_sampling as sampling
    monkeypatch.setattr(sampling, 'MAX_CANDIDATE_ROUNDS', 2)
    monkeypatch.setattr(sampling, 'physics_cell_ids', lambda state, definition: torch.zeros(len(state.mass),dtype=torch.long))
    with pytest.raises(RuntimeError,match='candidate budget exhausted'):
        sampling.sample_coverage(128,[1,2,3,4])


def test_invalid_yaw_and_reference_features_are_not_sanitized():
    from dataclasses import replace
    from env_raptor import RaptorSimulator
    from response_sampling import physics_cell_ids
    s = RaptorSimulator().reset(2,seed=3)
    with pytest.raises(ValueError,match='yaw authority'):
        physics_cell_ids(replace(s,inertia=torch.zeros_like(s.inertia)))
    with pytest.raises(ValueError,match='reference range'):
        physics_cell_ids(replace(s,torque_to_inertia=torch.full_like(s.mass,1201.)))


def test_coverage_contract_is_frozen_and_bound_in_checkpoint(tmp_path):
    from response_sampling import sampling_contract
    from response_training import binding, SOURCE_FILES
    from response_policy import ResponsePolicyConfig
    from response_task import TaskLossConfig
    args = parse_args(['--device','cpu','--train-sampling','coverage128','--scenarios','512'])
    contract = binding(args,ResponsePolicyConfig(),test_loss())
    assert contract['training_sampling'] == sampling_contract('coverage128')
    assert 'response_sampling.py' in SOURCE_FILES and 'configs/physics_coverage.json' in SOURCE_FILES
    assert contract['training_sampling']['calibration']['seed_base'] == 6500000000


def test_coverage_resume_repeats_actor_adam_and_next_selected_pool(tmp_path):
    import json
    from test_reference_training import args_for, run
    from response_training import evaluate_checkpoint
    def args(path,updates,extra=()):
        return args_for(path,updates,('--scenarios','32','--horizon','2','--train-sampling','coverage128','--group-max-groups','128','--group-min-scenarios','1',*extra))
    full, split = tmp_path/'full',tmp_path/'split'
    run(args(full,2)); run(args(split,1))
    run(args(split,2,('--resume',str(split/'latest.pt'))))
    a = torch.load(full/'latest.pt',weights_only=True)
    b = torch.load(split/'latest.pt',weights_only=True)
    assert a['model_sha256'] == b['model_sha256']
    for i,state in a['optimizer']['state'].items():
        for key,value in state.items():
            torch.testing.assert_close(value,b['optimizer']['state'][i][key],rtol=0,atol=0)
    ha = [json.loads(x) for x in (full/'history.jsonl').read_text().splitlines()]
    hb = [json.loads(x) for x in (split/'history.jsonl').read_text().splitlines()]
    assert [x['training_sampling'] for x in ha] == [x['training_sampling'] for x in hb]
    assert ha[0]['training_sampling']['cell_counts'] == [1]*128
    assert ha[0]['training_sampling']['candidate_seeds'] != ha[1]['training_sampling']['candidate_seeds']
    # EVAL ignores a TRAIN sampler/count that would be invalid for new training.
    ev = parse_args(['--mode','evaluate','--device','cpu','--dtype','float64',
                     '--train-sampling','coverage128','--scenarios','8',
                     '--checkpoint',str(split/'latest.pt'),'--work-dir',str(tmp_path/'eval')])
    result=evaluate_checkpoint(ev)
    expected=json.loads((split/'evaluation.jsonl').read_text().splitlines()[-1])
    assert result['scenario_count'] == 8
    assert result['task_objective'] == expected['task_objective']
    with pytest.raises(ValueError,match='configuration'):
        run(args(split,3,('--resume',str(split/'latest.pt'),'--train-sampling','random','--group-max-groups','16','--group-min-scenarios','32')))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_coverage_cpu_draws_equal_cuda_transfer_and_short_backward():
    from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
    from response_task import rollout, task_loss, TaskLossConfig
    from env_raptor import RaptorSimulator
    a,_=sample_training_scenarios(32,0,horizon=5,sampling='coverage128')
    b,_=sample_training_scenarios(32,0,horizon=5,sampling='coverage128',device='cuda')
    assert_state_equal(a,b.to('cpu',torch.float32))
    policy=ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=8)).cuda()
    trace=rollout(policy,RaptorSimulator(),b,5,time_decay=1.)
    task_loss(trace,test_loss()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())
