from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from train import _base_fieldnames, _raptor_fieldnames, _status_fieldnames, open_log


ROOT = Path(__file__).resolve().parents[1]


class OptimizerUpdateScheduleTest(unittest.TestCase):
    def assert_nested_equal(self, expected, actual) -> None:
        if torch.is_tensor(expected):
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                with self.subTest(key=key):
                    self.assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for expected_item, actual_item in zip(expected, actual):
                self.assert_nested_equal(expected_item, actual_item)
        else:
            self.assertEqual(expected, actual)

    def test_checkpoint_and_auxiliary_ramp_use_optimizer_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pt"
            log = root / "train.csv"
            command = (
                sys.executable,
                "train.py",
                "--device", "cpu",
                "--sim-backend", "torch",
                "--optimizer-updates", "2",
                "--horizon", "2",
                "--training-episode-steps", "4",
                "--persistent-episode-training",
                "--update-timing", "episode-boundary",
                "--batch-size", "2",
                "--encoder-dim", "16",
                "--hidden-dim", "12",
                "--lambda-motor-aux", "0.03",
                "--aux-weight-ramp-updates", "2",
                "--tail-window-steps", "2",
                "--steady-window-steps", "2",
                "--checkpoint-updates", "1,2",
                "--save-every", "0",
                "--post-update-check", "off",
                "--log-path", str(log),
                "--checkpoint-path", str(checkpoint),
            )
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)

            update_one = checkpoint.with_name("model_update_1.pt")
            update_two = checkpoint.with_name("model_update_2.pt")
            self.assertTrue(update_one.exists())
            self.assertTrue(update_two.exists())
            self.assertEqual(torch.load(update_one, map_location="cpu")["optimizer_update"], 1)
            self.assertEqual(torch.load(update_two, map_location="cpu")["optimizer_update"], 2)
            with log.open(newline="") as handle:
                accepted = [row for row in csv.DictReader(handle) if row["update_applied"] == "1"]
            self.assertEqual([int(row["optimizer_update"]) for row in accepted], [1, 2])
            self.assertEqual(
                [float(row["lambda_motor_aux_effective"]) for row in accepted],
                [0.015, 0.03],
            )

    def test_resume_rejects_pending_episode_gradients_but_boundary_is_bitwise(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            uninterrupted_checkpoint = root / "uninterrupted.pt"
            uninterrupted_log = root / "uninterrupted.csv"
            common = (
                sys.executable,
                "train.py",
                "--device", "cpu",
                "--sim-backend", "torch",
                "--horizon", "2",
                "--training-episode-steps", "4",
                "--persistent-episode-training",
                "--update-timing", "episode-boundary",
                "--batch-size", "2",
                "--encoder-dim", "16",
                "--hidden-dim", "12",
                "--tail-window-steps", "2",
                "--steady-window-steps", "2",
                "--training-diagnostics-every-updates", "1000",
                "--log-every", "1000",
                "--save-every", "0",
                "--post-update-check", "off",
            )
            uninterrupted_command = common + (
                "--steps", "4",
                "--checkpoint-steps", "1,2",
                "--log-path", str(uninterrupted_log),
                "--checkpoint-path", str(uninterrupted_checkpoint),
            )
            subprocess.run(
                uninterrupted_command,
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            mid_episode_checkpoint = root / "uninterrupted_step_1.pt"
            boundary_checkpoint = root / "uninterrupted_step_2.pt"
            self.assertTrue(mid_episode_checkpoint.exists())
            self.assertTrue(boundary_checkpoint.exists())

            rejected = subprocess.run(
                common + (
                    "--optimizer-updates", "2",
                    "--init-checkpoint-path", str(mid_episode_checkpoint),
                    "--resume-training-state",
                    "--log-path", str(root / "mid_resume.csv"),
                    "--checkpoint-path", str(root / "mid_resume.pt"),
                ),
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(
                "cannot resume an unfinished persistent episode with episode-boundary updates",
                rejected.stdout + rejected.stderr,
            )

            resumed_checkpoint = root / "resumed.pt"
            subprocess.run(
                common + (
                    "--optimizer-updates", "2",
                    "--init-checkpoint-path", str(boundary_checkpoint),
                    "--resume-training-state",
                    "--log-path", str(root / "resumed.csv"),
                    "--checkpoint-path", str(resumed_checkpoint),
                ),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            uninterrupted = torch.load(uninterrupted_checkpoint, map_location="cpu")
            resumed = torch.load(resumed_checkpoint, map_location="cpu")
            self.assertEqual(uninterrupted["optimizer_update"], 2)
            self.assertEqual(resumed["optimizer_update"], 2)
            self.assert_nested_equal(uninterrupted["model"], resumed["model"])
            self.assert_nested_equal(uninterrupted["optimizer"], resumed["optimizer"])
            self.assert_nested_equal(
                uninterrupted["training_state"],
                resumed["training_state"],
            )

    def test_append_rejects_historical_training_log_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "historical.csv"
            current_fields = (
                _base_fieldnames()
                + _status_fieldnames()
                + _raptor_fieldnames(10.0)
            )
            historical_fields = tuple(
                field
                for field in current_fields
                if field
                not in {
                    "tail_supervision_block_horizon",
                    "tail_supervision_block_boundary",
                    "first_tail_supervision_segment",
                }
            )
            with path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(historical_fields)

            with self.assertRaisesRegex(
                ValueError,
                "cannot append to a training log with a different schema",
            ):
                open_log(path, 10.0, append=True)


if __name__ == "__main__":
    unittest.main()
