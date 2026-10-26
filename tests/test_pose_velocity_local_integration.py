"""Keep the supplied region task connected to the real local training CLI."""
from __future__ import annotations

import torch

from tools.train_response_control import main


def test_positive_time_decay_runs_the_region_training_entry(tmp_path):
    work = tmp_path / 'decayed'
    main(['@configs/agile_flight.args', '--device', 'cpu', '--dtype', 'float64',
           '--train-sampling', 'random',
          '--scenarios', '8', '--eval-scenarios', '2', '--horizon', '8',
          '--group-max-groups', '1', '--group-min-scenarios', '32',
          '--time-decay', '1', '--memory-dim', '8', '--updates', '1',
          '--development-every', '1', '--checkpoint-every', '1',
          '--work-dir', str(work)])
    saved = torch.load(work / 'latest.pt', weights_only=True)
    assert saved['progress']['updates'] == 1
    assert saved['binding']['time_decay'] == 1.
    assert saved['model']['response_memory.weight_ih'].shape == (24, 69)
    assert saved['binding']['protocol']['loss']['mission']['velocities'] == (
        (0., 0., 0.), (2., 0., 0.), (0., 0., 0.))
