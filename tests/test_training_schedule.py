from __future__ import annotations

import unittest

import torch

from training_schedule import (
    FIXED_H500,
    MIXED_HORIZON_CURRICULUM,
    normalized_episode_value,
    select_episode_horizon,
)
from training_objectives import time_normalized_segment_sum


class EpisodeHorizonScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.common = {
            "physical_step_budget": 256_000_000,
            "checkpoint_physical_steps": (
                32_000_000,
                64_000_000,
                96_000_000,
                128_000_000,
                192_000_000,
                256_000_000,
            ),
            "batch_size": 256,
            "segment_steps": 250,
        }

    def test_fixed_schedule_is_h500(self) -> None:
        decision = select_episode_horizon(
            completed_physical_steps=0,
            episode_index=0,
            mode=FIXED_H500,
            **self.common,
        )
        self.assertEqual(decision.horizon_steps, 500)
        self.assertEqual(decision.phase, "fixed_h500")

    def test_curriculum_uses_requested_deterministic_mixtures(self) -> None:
        phase_two = [
            select_episode_horizon(
                completed_physical_steps=60_000_000,
                episode_index=index,
                mode=MIXED_HORIZON_CURRICULUM,
                **self.common,
            ).nominal_horizon_steps
            for index in range(10)
        ]
        self.assertEqual(phase_two.count(500), 7)
        self.assertEqual(phase_two.count(1000), 3)

        phase_three = [
            select_episode_horizon(
                completed_physical_steps=160_000_000,
                episode_index=index,
                mode=MIXED_HORIZON_CURRICULUM,
                **self.common,
            ).nominal_horizon_steps
            for index in range(10)
        ]
        self.assertEqual(phase_three.count(500), 5)
        self.assertEqual(phase_three.count(1000), 3)
        self.assertEqual(phase_three.count(2000), 2)

    def test_schedule_lands_exactly_on_checkpoint_boundary(self) -> None:
        # Only H500 (128k physical steps at batch 256) fits before 32M.
        decision = select_episode_horizon(
            completed_physical_steps=31_872_000,
            episode_index=9,
            mode=MIXED_HORIZON_CURRICULUM,
            **self.common,
        )
        self.assertEqual(decision.horizon_steps, 500)

    def test_schedule_does_not_consume_random_state(self) -> None:
        import torch

        torch.manual_seed(91)
        before = torch.get_rng_state().clone()
        select_episode_horizon(
            completed_physical_steps=160_000_000,
            episode_index=7,
            mode=MIXED_HORIZON_CURRICULUM,
            **self.common,
        )
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_full_256m_schedule_hits_every_boundary_exactly(self) -> None:
        completed = 0
        episode_index = 0
        observed: set[int] = set()
        checkpoints = set(self.common["checkpoint_physical_steps"])
        horizons: list[int] = []
        while completed < self.common["physical_step_budget"]:
            decision = select_episode_horizon(
                completed_physical_steps=completed,
                episode_index=episode_index,
                mode=MIXED_HORIZON_CURRICULUM,
                **self.common,
            )
            horizons.append(decision.horizon_steps)
            completed += self.common["batch_size"] * decision.horizon_steps
            episode_index += 1
            if completed in checkpoints:
                observed.add(completed)
        self.assertEqual(completed, self.common["physical_step_budget"])
        self.assertEqual(observed, checkpoints)
        self.assertIn(500, horizons)
        self.assertIn(1000, horizons)
        self.assertIn(2000, horizons)


class EpisodeLossNormalizationTests(unittest.TestCase):
    def test_repeated_dense_trajectory_does_not_grow_with_horizon(self) -> None:
        once = normalized_episode_value([2.0], 0.3)
        twice = normalized_episode_value([2.0, 2.0], 0.3)
        four_times = normalized_episode_value([2.0] * 4, 0.3)
        self.assertEqual(once, twice)
        self.assertEqual(once, four_times)

    def test_episode_event_is_not_divided_by_segment_count(self) -> None:
        self.assertAlmostEqual(
            normalized_episode_value([1.0, 1.0, 1.0, 1.0], 0.001),
            1.001,
        )

    def test_burn_in_shortened_auxiliary_is_an_exact_time_mean(self) -> None:
        first = torch.tensor(235.0, requires_grad=True)
        second = torch.tensor(500.0, requires_grad=True)
        # Batch one, H500, burn-in 15: 485 eligible time points total.
        first_objective = time_normalized_segment_sum(
            first,
            total_episode_count=485,
            segments_per_episode=2,
        )
        second_objective = time_normalized_segment_sum(
            second,
            total_episode_count=485,
            segments_per_episode=2,
        )
        accumulated_then_averaged = (first_objective + second_objective) / 2.0
        self.assertAlmostEqual(accumulated_then_averaged.item(), 735.0 / 485.0, places=6)


if __name__ == "__main__":
    unittest.main()
