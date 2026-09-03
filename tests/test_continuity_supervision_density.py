from __future__ import annotations

import unittest

from tools.analyze_continuity_supervision_density import extract_optimizer_blocks


def _row(
    step: int,
    *,
    first: int,
    boundary: int,
    reset: float,
    value: float,
) -> dict[str, str]:
    row = {
        "step": str(step),
        "first_optimization_segment": str(first),
        "optimization_block_boundary": str(boundary),
        "reset_mask": str(reset),
        "episode_target_steps": "4",
        "update_applied": str(boundary),
    }
    for name in (
        "early_position_cvar",
        "early_omega_cvar",
        "position",
        "velocity",
        "omega",
        "omega_decay_active_fraction",
        "final_position_cvar",
        "final_omega_cvar",
        "grad_norm_fp64_before_clip",
        "max_abs_param_delta",
    ):
        row[name] = str(value)
    return row


class ContinuitySupervisionDensityTest(unittest.TestCase):
    def test_classifies_complete_fresh_and_continuation_blocks(self) -> None:
        rows = [
            _row(1, first=1, boundary=0, reset=1.0, value=4.0),
            _row(2, first=0, boundary=1, reset=0.0, value=3.0),
            _row(3, first=1, boundary=0, reset=0.0, value=2.0),
            _row(4, first=0, boundary=1, reset=0.0, value=1.0),
        ]
        blocks = extract_optimizer_blocks(rows, seed=7)
        self.assertEqual([block["block_type"] for block in blocks], ["fresh_reset", "continuation"])
        self.assertEqual([block["segment_count"] for block in blocks], [2, 2])
        self.assertEqual(blocks[0]["first_velocity"], 4.0)
        self.assertEqual(blocks[1]["boundary_max_abs_param_delta"], 1.0)

    def test_rejects_incomplete_block(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete"):
            extract_optimizer_blocks(
                [_row(1, first=1, boundary=0, reset=1.0, value=1.0)],
                seed=7,
            )


if __name__ == "__main__":
    unittest.main()
