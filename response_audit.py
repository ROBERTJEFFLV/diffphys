"""Bounded, replayable update evidence. Observes training; never vetoes finite steps."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import uuid
import zipfile

import torch

AUDIT_VERSION = "actor-adam-update-capsule-v1"


def cpu_copy(value):
    """No live tensor aliases or retained autograd graphs in evidence."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return value


def json_report(value):
    if torch.is_tensor(value):
        return json_report(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {k: json_report(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_report(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # Failure evidence, not sanitized training data.
    return value


@dataclass(frozen=True)
class AuditConfig:
    history: int = 64
    max_events: int = 8
    raw_factor: float = 100.0
    step_factor: float = 5.0
    eval_ratio: float = 1.5

    def __post_init__(self):
        for key in ('history', 'max_events'):
            v = getattr(self, key)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError('audit-' + key.replace('_', '-') + ' must be a positive integer')
        for key in ('raw_factor', 'step_factor', 'eval_ratio'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 1:
                raise ValueError('audit-' + key.replace('_', '-') + ' must exceed one')

    @classmethod
    def from_args(cls, args):
        return cls(**{key: getattr(args, 'audit_' + key, default)
                      for key, default in asdict(cls()).items()})


@torch.no_grad()
def parameter_changes(policy, before: dict) -> dict:
    layers, total, maximum, previous = {}, 0.0, 0.0, 0.0
    for name, parameter in policy.named_parameters():
        old = before[name].detach().to(parameter)
        delta = parameter.detach().double() - old.double()
        norm = float(delta.norm())
        peak = float(delta.abs().max())
        base = float(old.double().norm())
        layers[name] = {'l2': norm, 'max_abs': peak, 'relative_l2': norm/max(base, 1e-12)}
        total += norm*norm
        previous += base*base
        maximum = max(maximum, peak)
    return {'l2': math.sqrt(total), 'max_abs': maximum,
            'relative_l2': math.sqrt(total)/max(math.sqrt(previous), 1e-12), 'layers': layers}


class UpdateAudit:
    """Keep recent capsules and pin bounded context windows on alerts.

    Thresholds trigger evidence ONLY, not sample filtering/optimizer rejection.
    Pinning uses hard links when available. Eviction is explicit in events.jsonl.
    Unique names preserve forensic branches when resuming an older checkpoint.
    """
    def __init__(self, work, config, *, evaluation_interval, binding, source_root,
                 source_files, torch_writer, json_writer):
        self.root = Path(work) / 'audit'
        self.config = config
        self.history = max(config.history, evaluation_interval + 1)
        self.interval = evaluation_interval
        self.binding = binding
        self.write_torch, self.write_json = torch_writer, json_writer
        self.updates = self.root / 'updates'
        self.events = self.root / 'events'
        self.updates.mkdir(parents=True, exist_ok=True)
        self.events.mkdir(parents=True, exist_ok=True)
        source = self.root / 'source.zip'
        if not source.exists():
            temporary = self.root / ('source.' + uuid.uuid4().hex + '.zip')
            try:
                with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
                    for name in source_files:
                        archive.write(Path(source_root)/name, name)
                os.replace(temporary, source)
            finally:
                temporary.unlink(missing_ok=True)
        with zipfile.ZipFile(source) as archive:
            for name in source_files:
                if archive.read(name) != (Path(source_root)/name).read_bytes():
                    raise ValueError('audit source archive mismatch; use a new run directory')
        self.write_json(self.root/'manifest.json', {
            'version': AUDIT_VERSION, 'config': asdict(config), 'effective_history': self.history,
            'binding': binding, 'source_archive_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'limits': 'CPU/CUDA/backend must match for bitwise replay. Alerts are not safety guarantees.'})

    def _log_event(self, event):
        with (self.root/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(json_report(event), allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def _pin(self, update, reasons, progress, *, force=False):
        last = progress.get('audit_last_event_update', -self.interval)
        if not force and update-last < self.interval:
            return
        destination = self.events / f'u{update:08d}_{uuid.uuid4().hex[:12]}'
        destination.mkdir()
        paths = sorted(self.updates.glob('*.pt'), key=lambda p: (p.stat().st_mtime_ns, p.name))[-self.history:]
        for path in paths:
            try:
                os.link(path, destination/path.name)
            except OSError:
                shutil.copy2(path, destination/path.name)
        self.write_json(destination/'event.json', {'update': update, 'reasons': reasons,
                                                  'capsules': [p.name for p in paths]})
        progress['audit_last_event_update'] = update
        self._log_event({'action': 'pinned', 'update': update, 'path': str(destination.relative_to(self.root)),
                         'reasons': reasons, 'capsule_count': len(paths)})
        events = sorted((p for p in self.events.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime_ns)
        for old in events[:-self.config.max_events]:
            self._log_event({'action': 'retention_eviction', 'path': str(old.relative_to(self.root))})
            shutil.rmtree(old)

    def record(self, *, update, before, policy, optimizer, names, rng_after,
               progress, sampling, gradients, groups, changes, failure=None):
        """Called after Adam (before rollback on failure); before is the existing backup."""
        pre_model, pre_adam, pre_rng = before
        capsule = {
            'version': AUDIT_VERSION, 'update': update, 'failure': failure,
            'binding': self.binding, 'policy_config': asdict(policy.config),
            'optimizer_parameter_names': names,
            'before': {'model': pre_model, 'optimizer': pre_adam, 'rng': pre_rng,
                       'sampling_index': update-1},
            'after': {'model': policy.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': rng_after},
            'sampling': sampling, 'groups': groups, 'gradients_entering_adam': gradients,
            'parameter_changes': changes,
        }
        path = self.updates/f'u{update:08d}_{uuid.uuid4().hex[:12]}.pt'
        self.write_torch(path, cpu_copy(capsule))
        reasons = []
        report = groups.get('gradient') if groups else None
        if report is not None:
            raw = report['values'][:, 0]
            if bool((raw > self.config.raw_factor*report['clip_norm']).any()):
                reasons.append('raw_group_above_alert_multiple_of_fixed_cap')
        old = progress.get('audit_step_norm_ema')
        current = changes.get('l2', float('nan')) if changes else float('nan')
        if old is not None and math.isfinite(current) and current > self.config.step_factor*max(old, 1e-12):
            reasons.append('actual_adam_step_jump')
        if failure is not None:
            reasons.append(failure)
        if reasons:
            self._pin(update, reasons, progress, force=failure is not None)
        if math.isfinite(current) and failure is None:
            progress['audit_step_norm_ema'] = current if old is None else .95*old + .05*current
        paths = sorted(self.updates.glob('*.pt'), key=lambda p: (p.stat().st_mtime_ns, p.name))
        for old_path in paths[:-self.history]:
            old_path.unlink()
        return str(path.relative_to(self.root.parent))

    def evaluation(self, report, previous_best, progress):
        if (previous_best is not None and previous_best > 0
                and report['task_objective'] > self.config.eval_ratio*previous_best):
            self._pin(progress['updates'], ['fixed_eval_regression'], progress, force=True)
