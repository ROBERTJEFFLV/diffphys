from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch

from env_l2f import L2FState


PHYSICAL_FIT_SAMPLER_SOURCE_METADATA_KEY = "physical_fit_sampler_source_sha256"
_PHYSICAL_FIT_SAMPLER_SOURCE = Path(__file__).with_name("env_l2f.py")
_PHYSICAL_FIT_STRICTLY_POSITIVE_FIELDS = (
    "mass",
    "thrust_coeff_c0",
    "thrust_coeff_c1",
    "thrust_to_weight",
    "torque_to_inertia",
    "rotor_distance_factor",
    "inertia_factor",
    "motor_time_rising",
    "motor_time_falling",
    "rotor_torque_constant",
    "cbrt_mass",
    "arm_length",
    "inertia_x",
    "inertia_y",
    "inertia_z",
    "alpha_roll_max",
    "alpha_pitch_max",
    "alpha_yaw_max",
    "eta_yaw",
    "jz_over_jxy",
    "dt_alpha_roll_max",
    "dt_alpha_yaw_max",
)


def physical_fit_sampler_source_sha256() -> str:
    """Return the exact ``env_l2f.py`` identity used for a physical-fit bank.

    The hash intentionally covers the complete source module rather than a
    hand-maintained sampler version string.  This is conservative: unrelated
    edits to the module can require rebuilding a physical-fit retain bank, but
    a bank can never be silently treated as originating from unverified code.
    """

    digest = hashlib.sha256()
    with _PHYSICAL_FIT_SAMPLER_SOURCE.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RetainBank:
    state: dict[str, torch.Tensor]
    scenario_id: torch.Tensor
    baseline_h500_success: torch.Tensor
    baseline_h10000_success: torch.Tensor
    metadata: dict[str, Any]

    def __len__(self) -> int:
        return int(self.scenario_id.numel())

    @property
    def baseline_h500_rate(self) -> float:
        return float(self.baseline_h500_success.float().mean().item())

    @property
    def baseline_h10000_rate(self) -> float:
        return float(self.baseline_h10000_success.float().mean().item())


def load_retain_bank(path: str | Path) -> RetainBank:
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict) or "state" not in payload:
        raise ValueError("retain bank must contain a state tensor mapping")
    state = payload["state"]
    expected = set(L2FState.__dataclass_fields__)
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise ValueError(f"retain bank state mismatch: missing={missing}, unexpected={unexpected}")
    lengths = {int(value.shape[0]) for value in state.values()}
    if len(lengths) != 1:
        raise ValueError("retain bank tensors must share their first dimension")
    count = lengths.pop()
    scenario_id = torch.as_tensor(payload.get("scenario_id"), dtype=torch.long)
    h500 = torch.as_tensor(payload.get("baseline_h500_success"), dtype=torch.bool)
    h10000 = torch.as_tensor(payload.get("baseline_h10000_success"), dtype=torch.bool)
    if any(value.numel() != count for value in (scenario_id, h500, h10000)):
        raise ValueError("retain bank metadata length does not match state tensors")
    return RetainBank(
        state={name: value.detach().cpu() for name, value in state.items()},
        scenario_id=scenario_id.cpu(),
        baseline_h500_success=h500.cpu(),
        baseline_h10000_success=h10000.cpu(),
        metadata=dict(payload.get("metadata", {})),
    )


def rigid_body_inertia_violation_mask(bank: RetainBank) -> torch.Tensor:
    """Return retain samples that cannot be principal moments of a rigid body."""

    inertia_x = bank.state["inertia_x"]
    inertia_y = bank.state["inertia_y"]
    inertia_z = bank.state["inertia_z"]
    finite = torch.isfinite(inertia_x) & torch.isfinite(inertia_y) & torch.isfinite(inertia_z)
    positive = (inertia_x > 0.0) & (inertia_y > 0.0) & (inertia_z > 0.0)
    triangle = (
        (inertia_x <= inertia_y + inertia_z)
        & (inertia_y <= inertia_x + inertia_z)
        & (inertia_z <= inertia_x + inertia_y)
    )
    return ~(finite & positive & triangle)


def validate_retain_bank_for_sampler(bank: RetainBank, broad_sampler: str) -> None:
    """Prevent corrected physical-fit runs from silently replaying legacy banks.

    Historical ``physical`` runs remain reproducible. New ``physical-fit`` runs
    require an explicitly matching bank and enforce the rigid-body inertia hard
    constraint that motivated the corrected sampler.
    """

    if broad_sampler != "physical-fit":
        return
    source_sampler = bank.metadata.get("broad_sampler")
    if source_sampler != "physical-fit":
        source_label = "missing" if source_sampler is None else repr(source_sampler)
        raise ValueError(
            "physical-fit training requires a dedicated retain bank with metadata "
            f"broad_sampler='physical-fit'; bank source is {source_label}"
        )

    expected_source_sha = physical_fit_sampler_source_sha256()
    bank_source_sha = bank.metadata.get(PHYSICAL_FIT_SAMPLER_SOURCE_METADATA_KEY)
    if bank_source_sha != expected_source_sha:
        source_label = "missing" if bank_source_sha is None else repr(bank_source_sha)
        raise ValueError(
            "physical-fit retain bank sampler source identity does not match the "
            f"current env_l2f.py: bank {PHYSICAL_FIT_SAMPLER_SOURCE_METADATA_KEY}="
            f"{source_label}, expected {expected_source_sha!r}; rebuild the bank"
        )

    non_finite_fields: list[str] = []
    for name, value in bank.state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"physical-fit retain bank state field {name!r} is not a tensor")
        if (torch.is_floating_point(value) or torch.is_complex(value)) and not bool(
            torch.isfinite(value).all().item()
        ):
            non_finite_fields.append(name)
    if non_finite_fields:
        raise ValueError(
            "physical-fit retain bank contains non-finite floating state values in "
            + ", ".join(sorted(non_finite_fields))
        )

    missing_positive_fields = [
        name for name in _PHYSICAL_FIT_STRICTLY_POSITIVE_FIELDS if name not in bank.state
    ]
    if missing_positive_fields:
        raise ValueError(
            "physical-fit retain bank is missing required physical state fields: "
            + ", ".join(missing_positive_fields)
        )
    non_positive_fields = [
        name
        for name in _PHYSICAL_FIT_STRICTLY_POSITIVE_FIELDS
        if not bool((bank.state[name] > 0.0).all().item())
    ]
    if non_positive_fields:
        raise ValueError(
            "physical-fit retain bank contains non-positive values in required "
            "physical fields: "
            + ", ".join(non_positive_fields)
        )
    if "force_std" not in bank.state or not bool((bank.state["force_std"] >= 0.0).all().item()):
        raise ValueError("physical-fit retain bank requires finite force_std >= 0")
    if not bool(
        (bank.state["motor_time_falling"] >= bank.state["motor_time_rising"]).all().item()
    ):
        raise ValueError("physical-fit retain bank requires motor_time_falling >= motor_time_rising")

    violation_count = int(rigid_body_inertia_violation_mask(bank).sum().item())
    if violation_count > 0:
        raise ValueError(
            "physical-fit retain bank contains "
            f"{violation_count}/{len(bank)} non-finite, non-positive, or "
            "rigid-body-inertia-infeasible samples"
        )


def apply_retain_bank_samples(
    state: L2FState,
    bank: RetainBank,
    reset_mask: torch.Tensor,
    *,
    fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace a fraction of reset samples and return mask/bank indices.

    Sampling uses the caller's active torch RNG.  Runs with the same seed and
    reset sequence therefore receive byte-identical retain scenarios.
    """
    if reset_mask.dtype != torch.bool or reset_mask.ndim != 1:
        raise ValueError("reset_mask must be a boolean [batch] tensor")
    if reset_mask.shape[0] != state.position.shape[0]:
        raise ValueError("reset_mask batch does not match state")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("retain fraction must be in [0, 1]")
    if len(bank) == 0 and fraction > 0.0:
        raise ValueError("cannot sample from an empty retain bank")

    retain_mask = torch.zeros_like(reset_mask)
    selected_bank_indices = torch.full(
        reset_mask.shape,
        -1,
        device=reset_mask.device,
        dtype=torch.long,
    )
    reset_indices = torch.nonzero(reset_mask, as_tuple=False).flatten()
    retain_count = int(round(float(fraction) * int(reset_indices.numel())))
    if retain_count == 0:
        return retain_mask, selected_bank_indices
    permutation = torch.randperm(reset_indices.numel(), device=reset_mask.device)
    target_indices = reset_indices[permutation[:retain_count]]
    bank_indices = torch.randint(
        len(bank),
        (retain_count,),
        device=reset_mask.device,
    )
    retain_mask[target_indices] = True
    selected_bank_indices[target_indices] = bank_indices
    bank_indices_cpu = bank_indices.detach().cpu()
    for name in L2FState.__dataclass_fields__:
        destination = getattr(state, name)
        if any(stride == 0 for stride in destination.stride()):
            destination = destination.clone()
            setattr(state, name, destination)
        source = bank.state[name][bank_indices_cpu].to(
            device=destination.device,
            dtype=destination.dtype,
        )
        destination[target_indices] = source
    return retain_mask, selected_bank_indices
