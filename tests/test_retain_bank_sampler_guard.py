from __future__ import annotations

import unittest

import torch

from retain_bank import (
    PHYSICAL_FIT_SAMPLER_SOURCE_METADATA_KEY,
    RetainBank,
    physical_fit_sampler_source_sha256,
    rigid_body_inertia_violation_mask,
    validate_retain_bank_for_sampler,
)
from env_l2f import L2FState


def _bank(
    *,
    source: str | None,
    inertia_z: float,
    source_sha: str | None = None,
    override: tuple[str, float] | None = None,
) -> RetainBank:
    metadata = {} if source is None else {"broad_sampler": source}
    if source_sha is not None:
        metadata[PHYSICAL_FIT_SAMPLER_SOURCE_METADATA_KEY] = source_sha
    state = {
        name: torch.ones(1, dtype=torch.float32)
        for name in L2FState.__dataclass_fields__
    }
    state.update(
        {
            "force_std": torch.zeros(1),
            "inertia_x": torch.tensor([1.0]),
            "inertia_y": torch.tensor([1.0]),
            "inertia_z": torch.tensor([inertia_z]),
            "motor_time_rising": torch.tensor([0.1]),
            "motor_time_falling": torch.tensor([0.2]),
        }
    )
    if override is not None:
        name, value = override
        state[name] = torch.tensor([value], dtype=torch.float32)
    return RetainBank(
        state=state,
        scenario_id=torch.tensor([0]),
        baseline_h500_success=torch.tensor([True]),
        baseline_h10000_success=torch.tensor([True]),
        metadata=metadata,
    )


class RetainBankSamplerGuardTest(unittest.TestCase):
    def test_historical_sampler_behavior_is_unchanged(self) -> None:
        bank = _bank(source="physical", inertia_z=3.0)
        validate_retain_bank_for_sampler(bank, "physical")
        self.assertEqual(int(rigid_body_inertia_violation_mask(bank).sum().item()), 1)

    def test_physical_fit_rejects_mismatched_or_missing_provenance(self) -> None:
        for source in (None, "physical"):
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "dedicated retain bank"):
                    validate_retain_bank_for_sampler(
                        _bank(source=source, inertia_z=1.5),
                        "physical-fit",
                    )

    def test_physical_fit_enforces_rigid_body_inertia_constraint(self) -> None:
        with self.assertRaisesRegex(ValueError, "inertia-infeasible"):
            validate_retain_bank_for_sampler(
                _bank(
                    source="physical-fit",
                    inertia_z=3.0,
                    source_sha=physical_fit_sampler_source_sha256(),
                ),
                "physical-fit",
            )
        validate_retain_bank_for_sampler(
            _bank(
                source="physical-fit",
                inertia_z=1.5,
                source_sha=physical_fit_sampler_source_sha256(),
            ),
            "physical-fit",
        )

    def test_physical_fit_requires_exact_sampler_source_hash(self) -> None:
        for source_sha in (None, "0" * 64):
            with self.subTest(source_sha=source_sha):
                with self.assertRaisesRegex(ValueError, "source identity"):
                    validate_retain_bank_for_sampler(
                        _bank(
                            source="physical-fit",
                            inertia_z=1.5,
                            source_sha=source_sha,
                        ),
                        "physical-fit",
                    )

    def test_physical_fit_rejects_non_finite_floating_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite.*position"):
            validate_retain_bank_for_sampler(
                _bank(
                    source="physical-fit",
                    inertia_z=1.5,
                    source_sha=physical_fit_sampler_source_sha256(),
                    override=("position", float("nan")),
                ),
                "physical-fit",
            )

    def test_physical_fit_rejects_non_positive_physical_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-positive.*mass"):
            validate_retain_bank_for_sampler(
                _bank(
                    source="physical-fit",
                    inertia_z=1.5,
                    source_sha=physical_fit_sampler_source_sha256(),
                    override=("mass", 0.0),
                ),
                "physical-fit",
            )

    def test_physical_fit_rejects_fall_time_below_rise_time(self) -> None:
        with self.assertRaisesRegex(ValueError, "motor_time_falling >= motor_time_rising"):
            validate_retain_bank_for_sampler(
                _bank(
                    source="physical-fit",
                    inertia_z=1.5,
                    source_sha=physical_fit_sampler_source_sha256(),
                    override=("motor_time_falling", 0.05),
                ),
                "physical-fit",
            )


if __name__ == "__main__":
    unittest.main()
