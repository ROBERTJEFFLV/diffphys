from __future__ import annotations

import pytest

from train import (
    _active_steps_in_optimization_block,
    _resolve_optimization_block_horizon,
)
from training_schedule import (
    COMPRESSED_T2_10PCT,
    COMPRESSED_T2_10PCT_SEQUENCE,
    select_episode_horizon,
)


def test_compressed_t2_sequence_has_preregistered_counts() -> None:
    assert len(COMPRESSED_T2_10PCT_SEQUENCE) == 67
    assert COMPRESSED_T2_10PCT_SEQUENCE.count(500) == 59
    assert COMPRESSED_T2_10PCT_SEQUENCE.count(1000) == 8
    assert COMPRESSED_T2_10PCT_SEQUENCE.count(2000) == 0
    assert sum(COMPRESSED_T2_10PCT_SEQUENCE) == 37_500


def test_compressed_t2_schedule_replays_exact_9p6m_budget() -> None:
    batch_size = 256
    budget = 9_600_000
    completed = 0
    selected: list[int] = []
    for episode_index in range(len(COMPRESSED_T2_10PCT_SEQUENCE)):
        decision = select_episode_horizon(
            completed_physical_steps=completed,
            physical_step_budget=budget,
            checkpoint_physical_steps=(budget,),
            batch_size=batch_size,
            segment_steps=250,
            episode_index=episode_index,
            mode=COMPRESSED_T2_10PCT,
        )
        selected.append(decision.horizon_steps)
        completed += batch_size * decision.horizon_steps

    assert tuple(selected) == COMPRESSED_T2_10PCT_SEQUENCE
    assert completed == budget


def test_compressed_t2_rejects_budget_or_checkpoint_truncation() -> None:
    with pytest.raises(ValueError, match="requires a physical-step budget"):
        select_episode_horizon(
            completed_physical_steps=0,
            physical_step_budget=9_600_000 - 128_000,
            checkpoint_physical_steps=(),
            batch_size=256,
            segment_steps=250,
            episode_index=0,
            mode=COMPRESSED_T2_10PCT,
        )

    # Episode 42 is the first H1000 in the compressed mixed phase. This
    # checkpoint would silently turn it into H500 without the strict guard.
    completed_before_episode_42 = 5_376_000
    with pytest.raises(ValueError, match="would truncate"):
        select_episode_horizon(
            completed_physical_steps=completed_before_episode_42,
            physical_step_budget=9_600_000,
            checkpoint_physical_steps=(5_504_000, 9_600_000),
            batch_size=256,
            segment_steps=250,
            episode_index=42,
            mode=COMPRESSED_T2_10PCT,
        )


def test_optimization_block_resolution_and_reset_relative_burn_in() -> None:
    assert _resolve_optimization_block_horizon(0, 500) == 500
    assert _resolve_optimization_block_horizon(0, 1000) == 1000
    assert _resolve_optimization_block_horizon(500, 1000) == 500
    with pytest.raises(ValueError, match="does not divide"):
        _resolve_optimization_block_horizon(750, 1000)

    assert _active_steps_in_optimization_block(
        block_start_episode_step=0,
        block_horizon=500,
        burn_in=15,
    ) == 485
    assert _active_steps_in_optimization_block(
        block_start_episode_step=500,
        block_horizon=500,
        burn_in=15,
    ) == 500
