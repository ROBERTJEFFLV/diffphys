from __future__ import annotations

import numpy as np
import torch

from tools.run_diverse_multiple_shooting_screen import balanced_update_batch_ids
from tools.run_timed_multiple_shooting_value_screen import (
    paired_bootstrap_extra_improvement,
)


def test_paired_bootstrap_error_sign_is_positive_when_ms_is_better() -> None:
    tbptt = np.linspace(0.10, 0.20, 128)
    multiple_shooting = tbptt - 0.02
    point, low, high = paired_bootstrap_extra_improvement(
        tbptt,
        multiple_shooting,
        mode="error",
        replicates=1000,
        seed=7,
    )
    assert np.isclose(point, 0.02)
    assert low > 0.0
    assert high > 0.0


def test_paired_bootstrap_success_sign_is_positive_when_ms_is_better() -> None:
    tbptt = np.zeros(128)
    multiple_shooting = np.ones(128)
    point, low, high = paired_bootstrap_extra_improvement(
        tbptt,
        multiple_shooting,
        mode="success",
        replicates=1000,
        seed=7,
    )
    assert point == 1.0
    assert low == 1.0
    assert high == 1.0


def test_diverse_schedule_visits_scenarios_eight_or_nine_times() -> None:
    update_batch_ids = balanced_update_batch_ids(
        16, 8, extra_batch_id=0, seed=7011
    )
    counts = torch.bincount(update_batch_ids, minlength=16)
    assert update_batch_ids.numel() == 129
    assert int(counts.min()) == 8
    assert int(counts.max()) == 9
    assert int((counts == 9).sum()) == 1
