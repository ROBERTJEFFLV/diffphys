from __future__ import annotations

import unittest

from env_l2f import L2FLossConfig
from l2f_full_cuda_backend import METRIC_NAMES


class FullCudaLossSemanticsTest(unittest.TestCase):
    def test_cuda_full_metrics_have_no_orientation_objective(self) -> None:
        self.assertEqual(
            METRIC_NAMES,
            (
                "loss",
                "tracking",
                "position",
                "velocity",
                "omega",
                "clf",
                "outward",
                "tail",
                "du",
                "ddu",
                "sat",
            ),
        )

    def test_loss_schema_only_has_position_velocity_and_omega_tracking(self) -> None:
        fields = set(L2FLossConfig.__dataclass_fields__)
        self.assertTrue({"w_p", "w_v", "w_omega"}.issubset(fields))
        self.assertFalse(any("attitude" in name or "yaw" in name for name in fields))


if __name__ == "__main__":
    unittest.main()
