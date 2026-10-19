"""Force-only controls preserve sensor acquisition, latency and legacy defaults."""
from dataclasses import asdict, fields
import json

import pytest
import torch

from diffphys.env_raptor import RaptorSimulator
from diffphys.response_noise import DisturbanceConfig, disturbance_report, measured_observation, pulse_at
from tools.train_response_control import parse_args


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_force_only_switch_preserves_every_sensor_and_physical_draw(dtype):
    simulator = RaptorSimulator()
    kwargs = {"seed": 719, "horizon": 120, "dtype": dtype}
    noisy = simulator.reset(12, **kwargs)
    force_free = simulator.reset(12, disturbances=DisturbanceConfig(force_enabled=False), **kwargs)
    changed = {"external_force", "force_std", "pulse_tape", "pulse_active_tape"}
    for field in fields(noisy):
        if field.name not in changed:
            assert torch.equal(getattr(noisy, field.name), getattr(force_free, field.name)), field.name
    assert noisy.external_force.any() and noisy.pulse_active_tape.any()
    for value in (force_free.external_force, force_free.external_torque,
                  force_free.force_std, force_free.pulse_tape, force_free.pulse_active_tape):
        assert not value.any()
    force, lever = pulse_at(force_free)
    assert not force.any() and not lever.any()  # Hence no induced r x R^T F torque.
    assert force_free.noise_tape.shape == (12, 121, 12)
    assert force_free.noise_tape.any() and force_free.velocity_delay.min() >= .010 - 1e-8
    assert torch.equal(measured_observation(noisy), measured_observation(force_free))
    assert not torch.equal(measured_observation(force_free)[:, :3], force_free.position)
    report = disturbance_report(force_free)
    assert report["force_model"] == "disabled"
    assert report["force_norm_N"] == [0., 0.] and not report["transient_force_schedule"]


@pytest.mark.parametrize("kwargs", [{"force_enabled": 1}, {"force_enabled": None},
                                    {"force_enabled": "False"}])
def test_force_switch_requires_a_boolean(kwargs):
    with pytest.raises(ValueError, match="force_enabled"):
        DisturbanceConfig(**kwargs)


def test_effective_switches_and_config_round_trip_are_unambiguous():
    config = DisturbanceConfig(force_enabled=False)
    assert config.enabled and not config.force_enabled and not config.pulse_enabled
    assert DisturbanceConfig(**json.loads(json.dumps(asdict(config)))) == config
    clean = DisturbanceConfig.clean()
    assert not clean.enabled and not clean.force_enabled and not clean.pulse_enabled
    legacy = asdict(DisturbanceConfig())
    legacy.pop("force_enabled")
    assert DisturbanceConfig(**legacy) == DisturbanceConfig()


@pytest.mark.parametrize("task, flags, enabled, force, pulses", [
    ("hover", [], True, True, True),
    ("agile", [], True, False, False),
    ("hover", ["--disable-external-forces"], True, False, False),
    ("agile", ["--enable-external-forces"], True, True, True),
    ("hover", ["--disable-pulses"], True, True, False),
    ("agile", ["--enable-external-forces", "--disable-pulses"], True, True, False),
    ("hover", ["--disable-disturbances"], False, False, False),
    ("agile", ["--disable-disturbances", "--enable-external-forces"], False, False, False),
])
def test_cli_resolves_task_defaults_and_explicit_force_overrides(task, flags, enabled, force, pulses):
    args = parse_args(["--task", task, "--epsilon-p", ".01", "--epsilon-a", ".01",
                       "--lambda-R", "25", *flags])
    config = DisturbanceConfig.from_args(args)
    assert (config.enabled, config.force_enabled, config.pulse_enabled) == (enabled, force, pulses)
    assert (config.position_std, config.velocity_std, config.attitude_std, config.omega_std) == (
        .001, .002, .001, .002)
    assert (config.velocity_delay_min, config.velocity_delay_max) == (.010, .030)


def test_cli_rejects_contradictory_force_overrides():
    with pytest.raises(SystemExit):
        parse_args(["--task", "agile", "--enable-external-forces", "--disable-external-forces"])
