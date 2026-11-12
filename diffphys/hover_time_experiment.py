"""Historical opt-in ablation wrapper around the shared production time loss.

The PASS/STOP time proxy specializes to reverse-cummin hover quality when PASS
is already fulfilled. Stable force rejection may require tilt, so hover quality
uses true position, speed and angular speed, without an absolute attitude target.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace

import torch

from .hover_time_loss import (HoverTimeConfig, settling_proxy, settling_metrics,
                              _quality_samples, _report)
from .response_task import TASK_OBJECTIVE_VERSION as BASE_OBJECTIVE_VERSION

VERSION = "hover-settling-time-ablation-v1"
_ACTIVE = False


@contextmanager
def experiment(config: HoverTimeConfig):
    """Scope one explicit experiment to the existing single-process trainer.

    The spawn sampler is unchanged. Checkpoints bind this code and the time
    configuration; ordinary CLI loading rejects the experimental objective.
    """
    from . import response_adjoints as adjoints
    from . import response_training as training

    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError("hover time experiment contexts cannot be nested")
    _ACTIVE = True
    names = ("collect_rollout", "evaluate", "binding", "require_reference_checkpoint",
             "TASK_OBJECTIVE_VERSION", "SOURCE_FILES")
    original = {name: getattr(training, name) for name in names}
    task = {"version": VERSION, "base_objective": BASE_OBJECTIVE_VERSION,
            **asdict(config)}

    def collect(*args, **kwargs):
        # This wrapper measures the base loss before adding its explicit treatment.
        kwargs.pop("hover_time", None)
        samples, masks = [], []
        native_rollout = adjoints.rollout
        record = None

        def observe(*a, **kw):
            trace = native_rollout(*a, **kw)
            with torch.set_grad_enabled(torch.is_grad_enabled() and config.weight > 0):
                if not samples:
                    ratio, alive = _quality_samples(trace, config, birth=True)
                    samples.append(ratio); masks.append(alive)
                ratio, alive = _quality_samples(trace, config)
                samples.append(ratio); masks.append(alive)
            return trace

        adjoints.rollout = observe
        try:
            record = original["collect_rollout"](*args, **kwargs)
            ratio, alive = torch.cat(samples), torch.cat(masks)
            simulator = args[1] if len(args) > 1 else kwargs["simulator"]
            dt = simulator.params.dt
            costs = record.costs
            if config.weight:
                costs = costs + config.weight * settling_proxy(ratio, alive, dt=dt) / ((len(ratio)-1)*dt)
            if not bool(torch.isfinite(costs).all()):
                raise FloatingPointError("nonfinite experimental hover objective")
            report = _report(record.metrics, ratio, alive, config, dt)
            if config.weight:
                report["task_objective"] = float(costs.detach().mean())
            return replace(record, costs=costs, metrics=report)
        except BaseException:
            if record is not None and record.probe is not None:
                record.probe.close()
            raise
        finally:
            adjoints.rollout = native_rollout

    @torch.no_grad()
    def evaluate(*args, **kwargs):
        kwargs["hover_time"] = None
        native_rollout = training.rollout
        traces = []

        def observe(*a, **kw):
            trace = native_rollout(*a, **kw)
            traces.append(trace)
            return trace

        training.rollout = observe
        try:
            base = original["evaluate"](*args, **kwargs)
        finally:
            training.rollout = native_rollout
        if len(traces) != 1:
            raise RuntimeError("experiment must observe exactly one normal EVAL")
        start, start_alive = _quality_samples(traces[0], config, birth=True)
        rest, rest_alive = _quality_samples(traces[0], config)
        simulator = args[1] if len(args) > 1 else kwargs["simulator"]
        return _report(base, torch.cat((start, rest)), torch.cat((start_alive, rest_alive)),
                       config, simulator.params.dt)

    def binding(*args, **kwargs):
        value = original["binding"](*args, **kwargs)
        value["protocol"]["hover_time_experiment"] = task
        # The wrapper supplies the time term; native TRAIN must contribute zero.
        value["protocol"]["hover_time"] = asdict(replace(config, weight=0.))
        value["algorithm"] += "+hover-time-ablation"
        return value

    def require(value):
        params = original["require_reference_checkpoint"](value)
        if value["binding"]["protocol"].get("hover_time_experiment") != task:
            raise ValueError("checkpoint hover time configuration mismatch")
        return params

    try:
        training.collect_rollout = collect
        training.evaluate = evaluate
        training.binding = binding
        training.require_reference_checkpoint = require
        training.TASK_OBJECTIVE_VERSION = VERSION
        yield
    finally:
        for name, value in original.items():
            setattr(training, name, value)
        _ACTIVE = False
