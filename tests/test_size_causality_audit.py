from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.audit_size_causality import run_invariance_experiment


ROOT = Path(__file__).resolve().parents[1]


class SizeCausalityAuditTest(unittest.TestCase):
    def test_matched_normalized_dynamics_hide_absolute_size(self) -> None:
        rows, mass_rows, coverage_rows = run_invariance_experiment(
            groups=8,
            steps=20,
            masses=(0.02, 0.2, 2.0),
            seed=1007,
        )
        self.assertEqual(len(mass_rows), 3)
        self.assertTrue(all(int(row["passed"]) == 1 for row in rows))
        self.assertLess(max(float(row["max_abs_difference"]) for row in rows), 1.0e-10)
        self.assertTrue(all(int(row["passed"]) == 1 for row in coverage_rows))
        coverage = {row["diagnostic"]: row for row in coverage_rows}
        self.assertGreater(float(coverage["motor_rising_branch"]["value"]), 0.0)
        self.assertGreater(float(coverage["motor_falling_branch"]["value"]), 0.0)
        self.assertGreater(float(coverage["raw_thrust_clamp_branch"]["value"]), 0.0)
        self.assertGreater(float(coverage["yaw_differential_commands"]["value"]), 0.0)

    def test_cli_provenance_limits_the_claim_to_constructed_tracked_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit"
            subprocess.run(
                (
                    sys.executable,
                    "tools/audit_size_causality.py",
                    "--groups",
                    "8",
                    "--steps",
                    "20",
                    "--masses",
                    "0.02,0.2,2.0",
                    "--output-dir",
                    str(output),
                ),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            provenance = json.loads((output / "RUN_PROVENANCE.json").read_text())
            self.assertEqual(
                provenance["experiment"],
                "constructive scale-equivalent counterexample",
            )
            self.assertTrue(provenance["all_tracked_dynamic_fields_invariant"])
            self.assertTrue(provenance["excitation_coverage_passed"])
            self.assertNotIn("all_state_fields_invariant", provenance)
            summary = (output / "SUMMARY.md").read_text()
            self.assertIn("constructed counterexample", summary)
            self.assertIn("does **not** show", summary)
            self.assertTrue((output / "EXCITATION_COVERAGE.csv").is_file())


if __name__ == "__main__":
    unittest.main()
