"""Stored loss settings are explicit; legacy v3 evaluation keeps zero costs."""
from dataclasses import fields
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from response_policy import ResponsePolicyConfig
from response_task import TaskLossConfig
from response_training import train, evaluate_checkpoint
from tools.train_response_control import parse_args


def training_args(path, updates=0, extra=()):
    return parse_args(['--device', 'cpu', '--dtype', 'float64', '--scenarios', '1',
                      '--eval-scenarios', '2', '--horizon', '8', '--window-steps', '4',
                      '--updates', str(updates), '--max-seconds', '60',
                      '--memory-dim', '8',
                      '--work-dir', str(path), *extra])


def run(args):
    pc = ResponsePolicyConfig(**{f.name: getattr(args, f.name) for f in fields(ResponsePolicyConfig)})
    lc = TaskLossConfig(**{f.name: getattr(args, f.name) for f in fields(TaskLossConfig)})
    return train(args, pc, lc)


def test_loss_config_binding_and_old_evaluation_defaults(tmp_path):
    path = tmp_path / 'run'
    run(training_args(path))
    saved = torch.load(path / 'latest.pt', weights_only=True)
    assert saved['binding']['protocol']['loss']['dead_cost'] == 3.
    assert saved['binding']['protocol']['loss']['terminal_cost'] == 200.
    altered = training_args(path, extra=('--resume', str(path / 'latest.pt'), '--terminal-cost', '100'))
    with pytest.raises(ValueError, match='configuration'):
        run(altered)
    # Legacy v3 loss dictionaries predate these fields: rescoring must not
    # silently add today's defaults to the checkpoint's stored objective.
    del saved['binding']['protocol']['loss']['dead_cost']
    del saved['binding']['protocol']['loss']['terminal_cost']
    torch.save(saved, tmp_path / 'legacy.pt')
    args = training_args(tmp_path / 'eval'); args.checkpoint = tmp_path / 'legacy.pt'
    result = evaluate_checkpoint(args)
    assert result['loss_config']['dead_cost'] == 0.
    assert result['loss_config']['terminal_cost'] == 0.
    assert result['task_components']['dead'] == 0.
    assert result['task_components']['terminal'] == 0.


def test_new_objective_resume_matches_uninterrupted(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    run(training_args(a, updates=2))
    run(training_args(b, updates=1))
    run(training_args(b, updates=2, extra=('--resume', str(b / 'latest.pt'))))
    first = torch.load(a / 'latest.pt', weights_only=True)
    second = torch.load(b / 'latest.pt', weights_only=True)
    assert first['model_sha256'] == second['model_sha256']
    for idx, values in first['optimizer']['state'].items():
        for key, value in values.items():
            torch.testing.assert_close(value, second['optimizer']['state'][idx][key], rtol=0, atol=0)
    assert torch.equal(first['rng']['torch'], second['rng']['torch'])
