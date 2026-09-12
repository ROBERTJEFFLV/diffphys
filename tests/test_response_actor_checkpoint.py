import copy
import json

import pytest
import torch

from response_policy import ResponseMotorPolicy, ResponsePolicyConfig
from response_task import TaskLossConfig
from response_training import (
    capture_rng,
    restore_rng,
    safe_global_clip,
    migrate_actor_weights,
    migrate_named_adam,
    train,
)
from tools.train_response_control import parse_args
import response_training as training


def test_explicit_legacy_loader_handles_numpy_module_rename(tmp_path):
    import numpy as np
    import zipfile
    original, renamed = tmp_path/'original.pt', tmp_path/'renamed.pt'
    rng = np.random.get_state()
    torch.save({'numpy_rng':rng,'tensor':torch.ones(2)},original,pickle_protocol=2)
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(renamed,'w') as target:
        for name in source.namelist():
            content = source.read(name)
            if name.endswith('data.pkl'):
                content = content.replace(b'numpy.core.multiarray\n',b'numpy._core.multiarray\n')
            target.writestr(name,content)
    loaded = training.load_legacy_checkpoint(renamed)
    assert np.array_equal(loaded['numpy_rng'][1],rng[1])
    assert torch.equal(loaded['tensor'],torch.ones(2))


def small_args(path, updates):
    return parse_args(['--device','cpu','--horizon','6','--window-steps','2','--scenarios','16',
                       '--updates',str(updates),'--development-every','1','--work-dir',str(path)])


def test_finite_development_deterioration_does_not_veto_adam_and_best_refreshes(tmp_path, monkeypatch):
    real_evaluate = training.evaluate
    scores = iter([1., .999, .998, 2.])
    def observed(*args):
        report = real_evaluate(*args)
        report['task_objective'] = next(scores)
        return report
    monkeypatch.setattr(training,'evaluate',observed)
    result = train(small_args(tmp_path,3),ResponsePolicyConfig(memory_dim=4,hidden_dim=8),TaskLossConfig())
    assert result['updates'] == 3 and result['best_update'] == 2
    best = torch.load(tmp_path/'best.pt',weights_only=True)
    latest = torch.load(tmp_path/'latest.pt',weights_only=True)
    assert best['next_update'] == 2 and latest['next_update'] == 3
    assert not all(torch.equal(best['model'][k],latest['model'][k]) for k in best['model'])


def test_nonfinite_adam_restores_actor_optimizer_rng_and_stops(tmp_path, monkeypatch):
    original = torch.optim.Adam.step
    def broken(optimizer,*args,**kwargs):
        original(optimizer,*args,**kwargs)
        with torch.no_grad():
            optimizer.param_groups[0]['params'][0].fill_(float('nan'))
    monkeypatch.setattr(torch.optim.Adam,'step',broken)
    with pytest.raises(FloatingPointError,match='Adam'):
        train(small_args(tmp_path,3),ResponsePolicyConfig(memory_dim=4,hidden_dim=8),TaskLossConfig())
    before = torch.load(tmp_path/'latest.pt',weights_only=True)
    restored = torch.load(tmp_path/'failure.pt',weights_only=True)
    assert restored['next_update'] == 0 and restored['optimizer']['state'] == before['optimizer']['state']
    assert all(torch.equal(before['model'][k],restored['model'][k]) for k in before['model'])
    assert torch.equal(before['rng']['torch'],restored['rng']['torch'])
    assert not (tmp_path/'history.jsonl').exists()


def test_main_import_is_independent_of_archived_algorithms():
    import subprocess, sys
    subprocess.run([sys.executable,'-c', '''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'response_critic','response_value','response_value_training',
                                      'response_proposals','response_shooting','petsc4py','petsc_kkt_solver',
                                      'full_space_shooting'}:
            raise AssertionError('production imported '+fullname)
sys.meta_path.insert(0,Block())
from tools.train_response_control import parse_args
assert parse_args([]).gradient_scale == .1
'''],check=True)


def test_fp64_clip_and_unused_parameter_stays_none():
    p = torch.nn.Parameter(torch.ones(2))
    unused = torch.nn.Parameter(torch.ones(1))
    p.grad = torch.full_like(p, 1e30)
    norm = safe_global_clip([p, unused], 10.0)
    assert norm > 1e30 and torch.isfinite(p.grad).all()
    assert float(p.grad.double().norm()) == pytest.approx(10.0)
    assert unused.grad is None
    p.grad[0] = float("inf")
    with pytest.raises(FloatingPointError):
        safe_global_clip([p], 10.0)


def test_weights_migration_rejects_unknown_or_missing_control_keys():
    actor = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4, hidden_dim=8))
    state = copy.deepcopy(actor.state_dict())
    state.update(
        {
            "response_predictor.0.weight": torch.zeros(8, 27),
            "response_predictor.0.bias": torch.zeros(8),
            "response_predictor.2.weight": torch.zeros(6, 8),
            "response_predictor.2.bias": torch.zeros(6),
        }
    )
    migrate_actor_weights(actor, state)
    bad = dict(state, mystery=torch.ones(1))
    with pytest.raises(ValueError):
        migrate_actor_weights(actor, bad)
    del state["response_memory.bias_hh"]
    with pytest.raises(ValueError):
        migrate_actor_weights(actor, state)


def test_adam_migration_uses_names_not_indices():
    actor = ResponseMotorPolicy(ResponsePolicyConfig(memory_dim=4, hidden_dim=8))
    old = torch.optim.Adam(reversed(list(actor.parameters())), lr=0.0003)
    for i, p in enumerate(actor.parameters()):
        p.grad = torch.full_like(p, 0.01 * (i + 1))
    old.step()
    new = torch.optim.Adam(actor.parameters(), lr=0.0003)
    names = list(reversed([n for n, p in actor.named_parameters()]))
    migrate_named_adam(new, actor, old.state_dict(), [names])
    for p in actor.parameters():
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(old.state[p][key], new.state[p][key])
    with pytest.raises(ValueError):
        migrate_named_adam(new, actor, old.state_dict(), None)


def test_ten_updates_equal_four_plus_six_and_recover_log_tail(tmp_path):
    def run(name, updates, resume=None):
        argv = [
            "--device",
            "cpu",
            "--horizon",
            "6",
            "--window-steps",
            "2",
            "--scenarios",
            "16",
            "--updates",
            str(updates),
            "--development-every",
            "5",
            "--checkpoint-every",
            "1",
            "--work-dir",
            str(tmp_path / name),
            "--max-seconds",
            "120",
        ]
        if resume:
            argv += ["--resume", str(resume)]
        args = parse_args(argv)
        return train(
            args, ResponsePolicyConfig(memory_dim=4, hidden_dim=8), TaskLossConfig(steady_steps=2)
        )

    assert run("full", 10)["updates"] == 10
    assert run("resume", 4)["updates"] == 4
    logfile = tmp_path / "resume" / "history.jsonl"
    with logfile.open("a") as f:
        f.write(json.dumps({"update": 5, "uncommitted": True}) + "\n{partial")
    assert run("resume", 10, tmp_path / "resume" / "latest.pt")["updates"] == 10
    a, b = [
        torch.load(tmp_path / name / "latest.pt", weights_only=True) for name in ("full", "resume")
    ]

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                equal(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        else:
            assert a == b

    for key in ("model", "optimizer", "rng", "optimizer_parameter_names", "next_update"):
        equal(a[key], b[key])
    rows = [json.loads(line) for line in logfile.read_text().splitlines()]
    assert [r["update"] for r in rows] == list(range(1, 11))
    assert "history" not in b and "critic" not in b
