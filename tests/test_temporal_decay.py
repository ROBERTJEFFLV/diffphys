from __future__ import annotations

import math
import unittest

from temporal_decay import current_decay_equivalent_alpha, resolve_step_gradient_decay


class TemporalDecayTest(unittest.TestCase):
    def test_current_mode_is_unchanged(self) -> None:
        self.assertEqual(
            resolve_step_gradient_decay(
                mode="current", dt=0.01, current_base=0.5, alpha=None
            ),
            0.5**0.01,
        )

    def test_current_and_equivalent_nmi_match(self) -> None:
        for base in (0.5, 0.7, 1.0):
            current = resolve_step_gradient_decay(
                mode="current", dt=0.01, current_base=base, alpha=None
            )
            nmi = resolve_step_gradient_decay(
                mode="nmi",
                dt=0.01,
                current_base=base,
                alpha=current_decay_equivalent_alpha(base),
            )
            self.assertAlmostEqual(current, nmi, places=15)

    def test_requested_nmi_form(self) -> None:
        value = resolve_step_gradient_decay(
            mode="nmi", dt=0.01, current_base=0.5, alpha=4.0
        )
        self.assertAlmostEqual(value, math.exp(-0.04), places=15)

    def test_invalid_combinations_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_step_gradient_decay(
                mode="nmi", dt=0.01, current_base=0.5, alpha=None
            )
        with self.assertRaises(ValueError):
            resolve_step_gradient_decay(
                mode="current", dt=0.01, current_base=0.5, alpha=1.0
            )


if __name__ == "__main__":
    unittest.main()
