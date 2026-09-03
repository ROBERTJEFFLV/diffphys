from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from scipy import signal as scipy_signal
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - exercised only in minimal environments
    scipy_signal = None
    scipy_stats = None


@dataclass(frozen=True)
class PhaseAnalysisConfig:
    period_steps: int = 250
    dt: float = 0.01
    nperseg: int = 1000
    overlap_fraction: float = 0.5
    null_permutations: int = 1000
    seed: int = 1007


@dataclass(frozen=True)
class CvarAlignmentConfig:
    position_threshold: float = 0.05
    velocity_threshold: float = 0.10
    omega_threshold: float = 0.20
    required_passes: int = 95
    window_steps: int = 100
    scenario_cvar_fraction: float = 0.20


def _linear_detrend(values: np.ndarray) -> np.ndarray:
    if values.size < 2:
        return values - values.mean() if values.size else values
    if scipy_signal is not None:
        return scipy_signal.detrend(values, type="linear")
    time = np.arange(values.size, dtype=np.float64)
    design = np.stack((time, np.ones_like(time)), axis=1)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def _welch(values: np.ndarray, config: PhaseAnalysisConfig) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(config.nperseg, values.size)
    if nperseg < 8:
        return np.array([]), np.array([])
    noverlap = min(int(nperseg * config.overlap_fraction), nperseg - 1)
    if scipy_signal is not None:
        return scipy_signal.welch(
            values,
            fs=1.0 / config.dt,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="linear",
            scaling="density",
        )
    window = np.hanning(nperseg)
    stride = max(1, nperseg - noverlap)
    powers = []
    for start in range(0, values.size - nperseg + 1, stride):
        segment = _linear_detrend(values[start : start + nperseg]) * window
        powers.append(np.square(np.abs(np.fft.rfft(segment))) / max(np.sum(np.square(window)), 1e-12))
    if not powers:
        return np.array([]), np.array([])
    return np.fft.rfftfreq(nperseg, d=config.dt), np.mean(powers, axis=0)


def _phase_lock_null_test(
    profiles: np.ndarray,
    *,
    harmonic: int,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    if profiles.ndim != 2:
        raise ValueError("profiles must have shape [scenario, phase]")
    if profiles.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    period = profiles.shape[1]
    coefficients = np.fft.rfft(profiles, axis=1)[:, harmonic]
    observed = float(np.abs(np.nanmean(coefficients)) / period)
    offsets = rng.integers(0, period, size=(permutations, profiles.shape[0]))
    shift_phase = np.exp(-2j * np.pi * harmonic * offsets / period)
    null = np.abs(np.nanmean(coefficients[None, :] * shift_phase, axis=1)) / period
    p_value = float((1 + np.sum(null >= observed)) / (permutations + 1))
    return observed, float(null.mean()), p_value


def _unit_columns(frame: pd.DataFrame, scenario_column: str, group_columns: Sequence[str]) -> list[str]:
    return [column for column in (*group_columns, scenario_column) if column in frame.columns]


def _phase_group_summary(
    scenario_profiles: pd.DataFrame,
    *,
    scenario_column: str,
    group_columns: Sequence[str],
    config: PhaseAnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    group_keys = [column for column in group_columns if column in scenario_profiles.columns]
    phase_records: list[dict[str, object]] = []
    lock_records: list[dict[str, object]] = []
    grouping = [*group_keys, "signal"]
    grouped = scenario_profiles.groupby(grouping, dropna=False, observed=True) if grouping else [((), scenario_profiles)]
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        metadata = dict(zip(grouping, group_key))
        unit_index = [column for column in _unit_columns(group, scenario_column, group_keys) if column not in group_keys]
        index = unit_index or [scenario_column]
        for profile_name in ("raw_phase_mean", "detrended_phase_mean"):
            if profile_name not in group.columns:
                continue
            pivot = group.pivot_table(index=index, columns="phase", values=profile_name, aggfunc="first")
            profiles = pivot.reindex(columns=np.arange(config.period_steps)).to_numpy(dtype=np.float64)
            mean = np.nanmean(profiles, axis=0)
            std = np.nanstd(profiles, axis=0, ddof=1) if profiles.shape[0] > 1 else np.zeros(config.period_steps)
            count = np.sum(np.isfinite(profiles), axis=0)
            for phase in range(config.period_steps):
                phase_records.append({
                    **metadata,
                    "profile": profile_name[: -len("_phase_mean")],
                    "phase": phase,
                    "scenario_count": int(profiles.shape[0]),
                    "mean": float(mean[phase]),
                    "std": float(std[phase]),
                    "standard_error": float(std[phase] / np.sqrt(max(int(count[phase]), 1))),
                })
            if profile_name == "detrended_phase_mean":
                for harmonic in (1, 2, 3):
                    observed, null_mean, p_value = _phase_lock_null_test(
                        profiles,
                        harmonic=harmonic,
                        permutations=config.null_permutations,
                        rng=rng,
                    )
                    lock_records.append({
                        **metadata,
                        "harmonic": harmonic,
                        "frequency_hz": harmonic / (config.period_steps * config.dt),
                        "phase_locked_amplitude": observed,
                        "null_mean_amplitude": null_mean,
                        "null_p_value": p_value,
                        "scenario_count": int(profiles.shape[0]),
                    })
    return pd.DataFrame(phase_records), pd.DataFrame(lock_records)


def analyze_h250_phase(
    frame: pd.DataFrame,
    *,
    signal_columns: Sequence[str],
    scenario_column: str = "scenario_uid",
    step_column: str = "step",
    group_columns: Sequence[str] = ("checkpoint", "seed", "failure_group"),
    config: PhaseAnalysisConfig = PhaseAnalysisConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {scenario_column, step_column, *signal_columns}
    if missing := required.difference(frame.columns):
        raise KeyError(f"Missing H250 phase columns: {sorted(missing)}")
    working = frame.copy()
    unit_columns = _unit_columns(working, scenario_column, group_columns)
    working = working.sort_values([*unit_columns, step_column])
    profile_frames: list[pd.DataFrame] = []
    frequency_records: list[dict[str, object]] = []
    target = 1.0 / (config.period_steps * config.dt)
    for unit_key, unit in working.groupby(unit_columns, sort=False, observed=True, dropna=False):
        if not isinstance(unit_key, tuple):
            unit_key = (unit_key,)
        metadata = dict(zip(unit_columns, unit_key))
        phase = (
            unit["phase"].to_numpy(dtype=np.int64)
            if "phase" in unit.columns
            else (unit[step_column].to_numpy(dtype=np.int64) - 1) % config.period_steps
        )
        for signal_name in signal_columns:
            raw = unit[signal_name].to_numpy(dtype=np.float64)
            detrended = _linear_detrend(raw)
            records = []
            for phase_value in range(config.period_steps):
                mask = phase == phase_value
                records.append({
                    **metadata,
                    "signal": signal_name,
                    "phase": phase_value,
                    "raw_phase_mean": float(raw[mask].mean()) if mask.any() else float("nan"),
                    "detrended_phase_mean": float(detrended[mask].mean()) if mask.any() else float("nan"),
                    "phase_sample_count": int(mask.sum()),
                })
            profile_frames.append(pd.DataFrame(records))

            for window_name, values in (
                ("all", raw),
                ("early", raw[: max(raw.size // 3, 1)]),
                ("middle", raw[raw.size // 3 : 2 * raw.size // 3]),
                ("late", raw[2 * raw.size // 3 :]),
            ):
                frequencies, power = _welch(values, config)
                record: dict[str, object] = {**metadata, "signal": signal_name, "window": window_name}
                if frequencies.size:
                    median = float(np.median(power[1:])) if power.size > 1 else float("nan")
                    for harmonic in (1, 2, 3):
                        index = int(np.argmin(np.abs(frequencies - harmonic * target)))
                        record[f"harmonic_{harmonic}_nearest_frequency_hz"] = float(frequencies[index])
                        record[f"harmonic_{harmonic}_psd"] = float(power[index])
                    record["median_non_dc_psd"] = median
                    record["target_to_median_psd_ratio"] = float(
                        power[int(np.argmin(np.abs(frequencies - target)))] / max(median, 1e-18)
                    )
                if values.size > config.period_steps:
                    centered = _linear_detrend(values)
                    left = centered[:-config.period_steps]
                    right = centered[config.period_steps:]
                    record["lag250_autocorrelation"] = (
                        float(np.corrcoef(left, right)[0, 1])
                        if left.std() > 0 and right.std() > 0
                        else float("nan")
                    )
                frequency_records.append(record)
    scenario_profiles = pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame()
    phase_summary, lock_frame = _phase_group_summary(
        scenario_profiles,
        scenario_column=scenario_column,
        group_columns=group_columns,
        config=config,
    )
    frequency = pd.DataFrame(frequency_records)
    if not lock_frame.empty:
        frequency = pd.concat((frequency, lock_frame.assign(window="phase_lock_null")), ignore_index=True, sort=False)
    return scenario_profiles, phase_summary, frequency


def analyze_streaming_phase_profiles(
    profiles: pd.DataFrame,
    online_frequency: pd.DataFrame,
    *,
    scenario_column: str = "scenario_uid",
    group_columns: Sequence[str] = ("checkpoint", "seed", "failure_group"),
    config: PhaseAnalysisConfig = PhaseAnalysisConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_required = {scenario_column, "phase"}
    if missing := base_required.difference(profiles.columns):
        raise KeyError(f"Missing streaming profile columns: {sorted(missing)}")
    if {"signal", "raw_phase_mean", "detrended_phase_mean"}.issubset(profiles.columns):
        summary, locks = _phase_group_summary(
            profiles,
            scenario_column=scenario_column,
            group_columns=group_columns,
            config=config,
        )
    else:
        suffix = "_detrended_phase_mean"
        signals = [column[: -len(suffix)] for column in profiles.columns if column.endswith(suffix)]
        if not signals:
            raise KeyError("No wide streaming phase signal columns were found")
        summary_frames = []
        lock_frames = []
        passthrough = [
            column for column in (*group_columns, scenario_column, "phase") if column in profiles.columns
        ]
        for signal in signals:
            long = profiles[passthrough].copy()
            long["signal"] = signal
            long["raw_phase_mean"] = profiles[f"{signal}_raw_phase_mean"]
            long["detrended_phase_mean"] = profiles[f"{signal}_detrended_phase_mean"]
            signal_summary, signal_locks = _phase_group_summary(
                long,
                scenario_column=scenario_column,
                group_columns=group_columns,
                config=config,
            )
            summary_frames.append(signal_summary)
            lock_frames.append(signal_locks)
        summary = pd.concat(summary_frames, ignore_index=True)
        locks = pd.concat(lock_frames, ignore_index=True)
    frequency = online_frequency.copy()
    if not locks.empty:
        frequency = pd.concat((frequency, locks.assign(window="phase_lock_null")), ignore_index=True, sort=False)
    return profiles.copy(), summary, frequency


def summarize_phase_group_profiles(phase_summary: pd.DataFrame) -> pd.DataFrame:
    """Reduce phase curves to one measured row per group/signal/profile."""
    required = {"phase", "mean", "standard_error", "signal", "profile"}
    if missing := required.difference(phase_summary.columns):
        raise KeyError(f"Missing phase summary columns: {sorted(missing)}")
    group_columns = [
        column
        for column in ("checkpoint", "seed", "failure_group", "signal", "profile")
        if column in phase_summary.columns
    ]
    records: list[dict[str, object]] = []
    for key, group in phase_summary.groupby(group_columns, dropna=False, observed=True, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        ordered = group.sort_values("phase")
        values = ordered["mean"].to_numpy(dtype=np.float64)
        phases = ordered["phase"].to_numpy(dtype=np.int64)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        center = float(np.nanmean(values))
        maximum = int(np.nanargmax(values))
        minimum = int(np.nanargmin(values))
        records.append({
            **dict(zip(group_columns, key)),
            "scenario_count": int(ordered["scenario_count"].max()) if "scenario_count" in ordered else 0,
            "phase_mean": center,
            "phase_rms_deviation": float(np.sqrt(np.nanmean(np.square(values - center)))),
            "phase_peak_to_peak": float(np.nanmax(values) - np.nanmin(values)),
            "phase_max_abs_deviation": float(np.nanmax(np.abs(values - center))),
            "phase_of_max": int(phases[maximum]),
            "phase_of_min": int(phases[minimum]),
            "boundary_jump_phase0_minus_last": float(values[0] - values[-1]),
            "mean_standard_error": float(np.nanmean(ordered["standard_error"])),
        })
    return pd.DataFrame(records)


def _violation(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.square(np.maximum(values / threshold - 1.0, 0.0))


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    finite = np.isfinite(scores)
    labels = labels[finite].astype(bool)
    scores = scores[finite]
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float("nan")
    if scipy_stats is not None:
        ranks = scipy_stats.rankdata(scores, method="average")
    else:
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(scores.size, dtype=np.float64)
        ranks[order] = np.arange(1, scores.size + 1)
    positive_ranks = ranks[labels]
    positives = int(labels.sum())
    negatives = labels.size - positives
    return float((positive_ranks.sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _classification(labels: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum(predicted & labels))
    fp = int(np.sum(predicted & ~labels))
    fn = int(np.sum(~predicted & labels))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-18)
    return float(f1), float(precision), float(recall)


def _best_f1(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    finite = np.isfinite(scores)
    labels = labels[finite].astype(bool)
    scores = scores[finite]
    if labels.size == 0:
        return (float("nan"),) * 4
    best = (-1.0, float("nan"), float("nan"), float("nan"))
    for threshold in np.unique(scores):
        f1, precision, recall = _classification(labels, scores >= threshold)
        if f1 > best[0]:
            best = (f1, float(threshold), precision, recall)
    return best


def compute_cvar_alignment(
    frame: pd.DataFrame,
    *,
    scenario_column: str = "scenario_uid",
    step_column: str = "step",
    position_column: str = "position_norm",
    velocity_column: str = "velocity_norm",
    omega_column: str = "omega_norm",
    group_columns: Sequence[str] = ("checkpoint", "seed"),
    config: CvarAlignmentConfig = CvarAlignmentConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {scenario_column, step_column, position_column, velocity_column, omega_column}
    if missing := required.difference(frame.columns):
        raise KeyError(f"Missing CVaR alignment columns: {sorted(missing)}")
    group_keys = [column for column in group_columns if column in frame.columns]
    unit_columns = [*group_keys, scenario_column]
    working = frame.sort_values([*unit_columns, step_column])
    channels = {
        "position": (position_column, config.position_threshold),
        "velocity": (velocity_column, config.velocity_threshold),
        "omega": (omega_column, config.omega_threshold),
    }
    records: list[dict[str, object]] = []
    for unit_key, scenario in working.groupby(unit_columns, sort=False, observed=True, dropna=False):
        if not isinstance(unit_key, tuple):
            unit_key = (unit_key,)
        tail = scenario.tail(config.window_steps)
        if tail.shape[0] != config.window_steps:
            continue
        record: dict[str, object] = dict(zip(unit_columns, unit_key))
        pass_vectors: dict[str, np.ndarray] = {}
        for name, (column, threshold) in channels.items():
            values = tail[column].to_numpy(dtype=np.float64)
            passes = values < threshold
            violation = _violation(values, threshold)
            sorted_violation = np.sort(violation)[::-1]
            pass_vectors[name] = passes
            record.update({
                f"{name}_pass_count": int(passes.sum()),
                f"{name}_failure_count": int((~passes).sum()),
                f"{name}_failure": bool(passes.sum() < config.required_passes),
                f"{name}_temporal_mean_violation": float(violation.mean()),
                f"{name}_temporal_max_violation": float(sorted_violation[0]),
                f"{name}_temporal_top5_mean": float(sorted_violation[:5].mean()),
                f"{name}_temporal_cvar5": float(sorted_violation[:5].mean()),
                f"{name}_sixth_largest_violation": float(sorted_violation[5]),
                f"{name}_active_violation_steps": int(np.count_nonzero(violation > 0)),
            })
        simultaneous = pass_vectors["position"] & pass_vectors["velocity"] & pass_vectors["omega"]
        record["simultaneous_pass_count"] = int(simultaneous.sum())
        record["steady_success"] = bool(simultaneous.sum() >= config.required_passes)
        record["overall_failure"] = not record["steady_success"]
        records.append(record)
    metrics = pd.DataFrame(records)

    ranking_records: list[dict[str, object]] = []
    grouped = [((), metrics)] if not group_keys else metrics.groupby(group_keys, dropna=False, observed=True)
    proxies = (
        "temporal_mean_violation", "temporal_max_violation", "temporal_top5_mean",
        "temporal_cvar5", "sixth_largest_violation",
    )
    for group_key, group in grouped:
        if group.empty:
            continue
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        metadata = dict(zip(group_keys, group_key))
        for channel in channels:
            labels = group[f"{channel}_failure"].to_numpy(dtype=bool)
            for proxy in proxies:
                score_column = f"{channel}_{proxy}"
                scores = group[score_column].to_numpy(dtype=np.float64)
                f1, threshold, precision, recall = _best_f1(labels, scores)
                top_count = max(1, int(np.ceil(config.scenario_cvar_fraction * scores.size)))
                top_indices = np.argsort(np.nan_to_num(scores, nan=-np.inf))[-top_count:]
                top_predicted = np.zeros(scores.size, dtype=bool)
                top_predicted[top_indices] = True
                top_f1, top_precision, top_recall = _classification(labels, top_predicted)
                spearman = kendall = float("nan")
                if scipy_stats is not None and np.unique(scores[np.isfinite(scores)]).size > 1:
                    spearman = float(scipy_stats.spearmanr(scores, -group[f"{channel}_pass_count"]).statistic)
                    kendall = float(scipy_stats.kendalltau(scores, -group[f"{channel}_pass_count"]).statistic)
                ranking_records.append({
                    **metadata,
                    "channel": channel,
                    "proxy": proxy,
                    "scenario_count": int(group.shape[0]),
                    "failure_count": int(labels.sum()),
                    "auroc": _binary_auroc(labels, scores),
                    "best_f1": f1,
                    "best_threshold": threshold,
                    "precision_at_best_f1": precision,
                    "recall_at_best_f1": recall,
                    "hard_top20_count": top_count,
                    "hard_top20_f1": top_f1,
                    "hard_top20_precision": top_precision,
                    "hard_top20_recall": top_recall,
                    "spearman_failure_ranking": spearman,
                    "kendall_failure_ranking": kendall,
                })
    return metrics, pd.DataFrame(ranking_records)
