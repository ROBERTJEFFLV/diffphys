from __future__ import annotations

import unittest

from tools.analyze_physical_failure_axes import (
    SCENARIO_FIELDS,
    assert_scenario_pairing,
    equal_rank_groups,
)


class PhysicalFailureAxisAuditTest(unittest.TestCase):
    def test_equal_rank_groups_are_stable_and_complete(self) -> None:
        values = [3.0, 1.0, 4.0, 2.0, 0.0, 5.0, 7.0, 6.0]
        groups = equal_rank_groups(values)
        self.assertEqual(groups, [[4, 1], [3, 0], [2, 5], [7, 6]])
        self.assertEqual(
            sorted(index for group in groups for index in group),
            list(range(8)),
        )

    def test_scenario_pairing_rejects_field_mismatch(self) -> None:
        row = {field: "1" for field in SCENARIO_FIELDS}
        changed = dict(row)
        changed["thrust_to_weight"] = "2"
        with self.assertRaisesRegex(ValueError, "thrust_to_weight"):
            assert_scenario_pairing([row], [changed])


if __name__ == "__main__":
    unittest.main()
