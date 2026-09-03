from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PhysicalFitSamplerAuditTest(unittest.TestCase):
    def test_small_audit_has_no_hard_constraint_violations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit"
            subprocess.run(
                (
                    sys.executable,
                    "tools/audit_physical_fit_sampler.py",
                    "--seeds", "7,17",
                    "--samples-per-seed", "256",
                    "--output-dir", str(output),
                ),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            with (output / "HARD_CONSTRAINTS.csv").open(newline="") as handle:
                constraints = list(csv.DictReader(handle))
            self.assertTrue(constraints)
            self.assertTrue(all(row["violations"] == "0" for row in constraints))
            self.assertTrue(all(row["passed"] == "1" for row in constraints))
            checks = {row["check"] for row in constraints}
            self.assertIn("all_returned_dynamics_fields_finite", checks)
            self.assertIn("fall_time_not_faster_than_rise", checks)
            self.assertIn("fall_to_rise_ratio_not_above_2p6", checks)
            self.assertIn("realized_inertia_coefficient_in_0p045_0p50", checks)
            self.assertIn(
                "realized_rotor_torque_constant_in_0p006_0p035", checks
            )
            self.assertIn("thrust_curve_matches_declared_thrust_to_weight", checks)
            self.assertIn(
                "derived_capabilities_reconstruct_from_primitive_fields", checks
            )
            self.assertIn("static_translation_thrust_magnitude_feasible", checks)
            self.assertIn("balanced_roots_cover_all_4pow4_joint_cells", checks)

            with (output / "SAMPLES.csv").open(newline="") as handle:
                samples = list(csv.DictReader(handle))
            self.assertEqual(len(samples), 512)
            self.assertTrue(all(row["inertia_triangle_valid"] == "1" for row in samples))
            self.assertTrue(
                all(row["finite_all_returned_fields"] == "1" for row in samples)
            )
            self.assertTrue(
                all(row["static_translation_trim_feasible"] == "1" for row in samples)
            )

            summary = (output / "SUMMARY.md").read_text()
            self.assertIn("does **not** establish", summary)
            self.assertIn("construction assumptions", summary)

            provenance = json.loads((output / "RUN_PROVENANCE.json").read_text())
            self.assertEqual(provenance["total_samples"], 512)
            self.assertTrue(provenance["hard_constraints_passed"])
            self.assertEqual(provenance["joint_root_grid"]["cells"], 256)


if __name__ == "__main__":
    unittest.main()
