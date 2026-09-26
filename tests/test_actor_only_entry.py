"""The Time Decay trainer runs without any training-only network package."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_reference_training import args_for, run
from response_training import evaluate_checkpoint


def test_training_and_resume_without_teacher_module(tmp_path):
    # Simulate installing only the production Actor/physics modules.
    code = """
import sys
sys.modules['response_contraction'] = None
from test_reference_training import args_for, run
from pathlib import Path
path = Path(sys.argv[1])
run(args_for(path, updates=1))
run(args_for(path, updates=2, extra=('--resume', str(path/'latest.pt'))))
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, '-c', "import sys; sys.path.insert(0, 'tests');\n" + code,
         str(tmp_path / 'run')], cwd=root, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    saved = torch.load(tmp_path / 'run/latest.pt', weights_only=True)
    assert saved['next_update'] == 2
    assert saved['binding']['algorithm'] == 'time-decayed-bptt-adam+physics-group-gradient-median'
    assert saved['optimizer']['state']
    rows = [json.loads(line) for line in (tmp_path / 'run/history.jsonl').read_text().splitlines()]
    assert [row['update'] for row in rows] == [1, 2]
    assert all(row['finite'] for row in rows)


def test_archived_auxiliary_payload_is_irrelevant_to_actor_evaluation(tmp_path):
    run(args_for(tmp_path / 'run', updates=0))
    original = tmp_path / 'run/latest.pt'
    args = args_for(tmp_path / 'eval_original')
    args.checkpoint = original
    expected = evaluate_checkpoint(args)
    saved = torch.load(original, weights_only=True)
    # Archived auxiliary metadata must not instantiate another model at EVAL.
    saved['binding']['source_sha256'] = 'archived-source'
    saved['binding']['contraction'] = {'version': 'archived-auxiliary-format'}
    saved['contraction'] = {'model': {'obsolete': torch.tensor(float('nan'))}}
    archived = tmp_path / 'archived.pt'
    torch.save(saved, archived)
    args = args_for(tmp_path / 'eval_archived')
    args.checkpoint = archived
    actual = evaluate_checkpoint(args)
    for name in ('task_objective', 'position_rms', 'velocity_rms', 'omega_rms'):
        assert actual[name] == expected[name]
    assert not actual['source_match']
    with pytest.raises(ValueError, match='configuration'):
        run(args_for(tmp_path / 'resume', extra=('--resume', str(archived))))
