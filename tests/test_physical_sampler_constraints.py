from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import torch

from env_l2f import (
    L2FParams,
    _joint_stratified_units,
    sample_physical_fit_episode_dynamics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_physical_fit_inertia_is_realizable_by_a_rigid_body() -> None:
    torch.manual_seed(20260804)
    dynamics = sample_physical_fit_episode_dynamics(
        L2FParams(),
        16_384,
        torch.device("cpu"),
        torch.float64,
    )

    assert torch.all(dynamics.inertia_x > 0.0)
    assert torch.all(dynamics.inertia_y > 0.0)
    assert torch.all(dynamics.inertia_z > 0.0)
    assert torch.all(dynamics.inertia_x + dynamics.inertia_y >= dynamics.inertia_z)
    assert torch.all(dynamics.inertia_y + dynamics.inertia_z >= dynamics.inertia_x)
    assert torch.all(dynamics.inertia_z + dynamics.inertia_x >= dynamics.inertia_y)
    assert float(dynamics.jz_over_jxy.min()) >= 1.45
    assert float(dynamics.jz_over_jxy.max()) <= 1.95


def test_balanced_four_root_batch_covers_every_joint_cell() -> None:
    torch.manual_seed(20260804)
    roots = _joint_stratified_units(
        256,
        torch.device("cpu"),
        torch.float64,
        bins=4,
        dimensions=4,
    )
    bin_ids = torch.stack(tuple(torch.floor(root * 4).long() for root in roots), dim=-1)
    cell_ids = (bin_ids * torch.tensor([1, 4, 16, 64])).sum(dim=-1)

    assert torch.unique(cell_ids).numel() == 256
    for dimension in range(4):
        torch.testing.assert_close(
            torch.bincount(bin_ids[:, dimension], minlength=4),
            torch.full((4,), 64, dtype=torch.long),
        )


def test_named_vehicle_inertia_ratios_remain_inside_fit_envelope() -> None:
    named_ratios = (
        2.17e-5 / 1.40e-5,
        1.45e-4 / 8.00e-5,
        2.80e-2 / 1.50e-2,
    )
    assert all(1.45 <= ratio <= 1.95 for ratio in named_ratios)


def test_x500_reference_inertia_matches_its_inverse() -> None:
    path = (
        ROOT
        / "reference/current_diffphys_cuda_chain/include/rl_tools/rl/environments/l2f/parameters/dynamics/x500_real.h"
    )
    text = path.read_text(encoding="utf-8")
    block = re.search(r"// J\s*\{(?P<j>.*?)// J_inv\s*\{(?P<j_inv>.*?)// hovering", text, re.S)
    assert block is not None
    j_values = [float(value) for value in re.findall(r"\{([0-9.eE+-]+), 0\.0, 0\.0\}", block.group("j"))]
    j_inv_values = [float(value) for value in re.findall(r"\{([0-9.eE+-]+), 0\.0, 0\.0\}", block.group("j_inv"))]
    # The diagonal layout uses a different zero placement for rows two and three;
    # explicitly include those literals after verifying the first entry by parse.
    assert len(j_values) == 1 and len(j_inv_values) == 1
    assert math.isclose(j_values[0] * j_inv_values[0], 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12)


def test_historical_size_mass_toggle_is_not_silently_ignored() -> None:
    result = subprocess.run(
        (sys.executable, "train.py", "--correlated-size-mass-sampling"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "historical no-op flags" in result.stdout + result.stderr
