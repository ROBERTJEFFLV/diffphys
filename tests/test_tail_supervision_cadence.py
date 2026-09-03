from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TailSupervisionCadenceTest(unittest.TestCase):
    def _command(self, root: Path, *extra: str) -> tuple[str, ...]:
        return (
            sys.executable,
            "train.py",
            "--device", "cpu",
            "--sim-backend", "torch",
            "--optimizer-updates", "1",
            "--horizon", "2",
            "--training-episode-steps", "8",
            "--persistent-episode-training",
            "--update-timing", "episode-boundary",
            "--batch-size", "2",
            "--encoder-dim", "16",
            "--hidden-dim", "12",
            "--tail-window-steps", "2",
            "--steady-window-steps", "2",
            "--tail-selection-mode", "independent",
            "--w-position-cvar", "0.001",
            "--w-omega-cvar", "0.001",
            "--early-tail-weight", "0.25",
            "--final-tail-weight", "1.0",
            "--correct-episode-boundary-weighting",
            "--training-diagnostics-every-updates", "1000",
            "--log-every", "1000",
            "--save-every", "0",
            "--post-update-check", "off",
            "--log-path", str(root / "train.csv"),
            "--checkpoint-path", str(root / "model.pt"),
            *extra,
        )

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, str]]:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_two_tail_blocks_accumulate_into_one_optimizer_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(
                self._command(root, "--tail-supervision-block-horizon", "4"),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            rows = self._read_rows(root / "train.csv")

            self.assertEqual(len(rows), 4)
            self.assertEqual(
                [int(row["first_optimization_segment"]) for row in rows],
                [1, 0, 0, 0],
            )
            self.assertEqual(
                [int(row["optimization_block_boundary"]) for row in rows],
                [0, 0, 0, 1],
            )
            self.assertEqual(
                [int(row["first_tail_supervision_segment"]) for row in rows],
                [1, 0, 1, 0],
            )
            self.assertEqual(
                [int(row["tail_supervision_block_boundary"]) for row in rows],
                [0, 1, 0, 1],
            )
            self.assertEqual(
                [int(row["update_applied"]) for row in rows],
                [0, 0, 0, 1],
            )
            self.assertEqual(
                [int(row["grad_accum_segments"]) for row in rows],
                [1, 2, 3, 4],
            )
            self.assertTrue(float(rows[0]["early_position_cvar"]) > 0.0)
            self.assertTrue(float(rows[1]["final_position_cvar"]) > 0.0)
            self.assertTrue(float(rows[2]["early_position_cvar"]) > 0.0)
            self.assertTrue(float(rows[3]["final_position_cvar"]) > 0.0)

    def test_zero_preserves_optimizer_coupled_tail_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(
                self._command(root),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            rows = self._read_rows(root / "train.csv")

            self.assertEqual(
                [int(row["first_tail_supervision_segment"]) for row in rows],
                [1, 0, 0, 0],
            )
            self.assertEqual(
                [int(row["tail_supervision_block_boundary"]) for row in rows],
                [0, 0, 0, 1],
            )
            self.assertTrue(float(rows[0]["early_position_cvar"]) > 0.0)
            self.assertEqual(float(rows[1]["final_position_cvar"]), 0.0)
            self.assertEqual(float(rows[2]["early_position_cvar"]), 0.0)
            self.assertTrue(float(rows[3]["final_position_cvar"]) > 0.0)

    def test_fixed_schedule_uses_configured_h8_reset_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(
                self._command(root, "--tail-supervision-block-horizon", "8"),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            rows = self._read_rows(root / "train.csv")

            self.assertEqual(len(rows), 4)
            self.assertEqual(
                [int(row["tail_supervision_block_horizon"]) for row in rows],
                [8, 8, 8, 8],
            )
            self.assertEqual(
                [int(row["first_tail_supervision_segment"]) for row in rows],
                [1, 0, 0, 0],
            )
            self.assertEqual(
                [int(row["tail_supervision_block_boundary"]) for row in rows],
                [0, 0, 0, 1],
            )
            self.assertEqual(
                [int(row["update_applied"]) for row in rows],
                [0, 0, 0, 1],
            )

    def test_invalid_tail_cadence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nondivisible = subprocess.run(
                self._command(root, "--tail-supervision-block-horizon", "3"),
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(nondivisible.returncode, 0)
            self.assertIn(
                "must be divisible by --horizon",
                nondivisible.stdout + nondivisible.stderr,
            )

            combined = subprocess.run(
                self._command(
                    root,
                    "--tail-supervision-block-horizon", "4",
                    "--tail-selection-mode", "combined",
                ),
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(combined.returncode, 0)
            self.assertIn(
                "supports only independent CVaR events",
                combined.stdout + combined.stderr,
            )


if __name__ == "__main__":
    unittest.main()
