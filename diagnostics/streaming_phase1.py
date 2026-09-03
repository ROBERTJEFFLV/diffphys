from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .phase1_fast_common import atomic_write_dataframe, atomic_write_json


@dataclass(frozen=True)
class StreamingConfig:
    batch_size: int
    signal_names: tuple[str, ...]
    period_steps: int = 250
    tail_steps: int = 100
    early_steps: int = 500
    selection_tail_steps: int = 500
    detrend_steps: int = 1000
    welch_steps: int = 1000
    welch_overlap_steps: int = 500
    dt: float = 0.01


def _ordered_ring(buffer: np.ndarray, next_index: int, valid_count: int) -> np.ndarray:
    if valid_count <= 0:
        return buffer[:, :0]
    if valid_count < buffer.shape[1]:
        return buffer[:, :valid_count]
    return np.concatenate((buffer[:, next_index:], buffer[:, :next_index]), axis=1)


class Phase1StreamingAccumulator:
    """Bounded-memory H250, Welch, tail-risk, and snapshot-plan collector.

    Raw and long-window linearly detrended phase statistics are kept separately.
    No complete trajectory or recurrent state history is retained.
    """

    def __init__(self, config: StreamingConfig) -> None:
        self.config = config
        if config.batch_size <= 0 or not config.signal_names:
            raise ValueError("batch_size and signal_names must be non-empty")
        if not 0 <= config.welch_overlap_steps < config.welch_steps:
            raise ValueError("welch overlap must be in [0, welch_steps)")
        if min(
            config.period_steps,
            config.tail_steps,
            config.early_steps,
            config.selection_tail_steps,
            config.welch_steps,
        ) <= 0:
            raise ValueError("streaming window lengths must be positive")
        if config.detrend_steps < config.period_steps or config.detrend_steps % config.period_steps:
            raise ValueError("detrend_steps must be a positive multiple of period_steps")
        shape = (config.batch_size, config.period_steps, len(config.signal_names))
        self.raw_phase_sum = np.zeros(shape, dtype=np.float64)
        self.raw_phase_sq_sum = np.zeros(shape, dtype=np.float64)
        self.detrended_phase_sum = np.zeros(shape, dtype=np.float64)
        self.detrended_phase_sq_sum = np.zeros(shape, dtype=np.float64)
        self.phase_count = np.zeros(shape[:2], dtype=np.int64)
        self.detrended_phase_count = np.zeros(shape[:2], dtype=np.int64)
        self.detrend_buffer = np.zeros(
            (config.batch_size, config.detrend_steps, len(config.signal_names)), dtype=np.float32
        )
        self.detrend_fill = 0
        self.tail_buffer = np.zeros(
            (config.batch_size, config.tail_steps, len(config.signal_names)), dtype=np.float32
        )
        self.tail_index = 0
        self.early_buffer = np.zeros(
            (config.batch_size, config.early_steps, len(config.signal_names)), dtype=np.float32
        )
        self.early_fill = 0
        self.selection_buffer = np.zeros(
            (config.batch_size, config.selection_tail_steps, len(config.signal_names)), dtype=np.float32
        )
        self.selection_index = 0
        self.total_steps = 0
        self.welch_buffer = np.zeros(
            (config.batch_size, config.welch_steps, len(config.signal_names)), dtype=np.float32
        )
        self.welch_fill = 0
        self.welch_power_sum = np.zeros(
            (config.batch_size, config.welch_steps // 2 + 1, len(config.signal_names)), dtype=np.float64
        )
        self.welch_window_count = 0
        self._hann = np.hanning(config.welch_steps).astype(np.float64)

    def update(self, signals: np.ndarray) -> None:
        values = np.asarray(signals, dtype=np.float32)
        expected = (self.config.batch_size, len(self.config.signal_names))
        if values.shape != expected:
            raise ValueError(f"signals shape is {values.shape}, expected {expected}")
        if not np.isfinite(values).all():
            raise FloatingPointError("non-finite value supplied to Phase1StreamingAccumulator")

        phase = self.total_steps % self.config.period_steps
        values64 = values.astype(np.float64)
        self.raw_phase_sum[:, phase] += values64
        self.raw_phase_sq_sum[:, phase] += np.square(values64)
        self.phase_count[:, phase] += 1

        self.detrend_buffer[:, self.detrend_fill] = values
        self.detrend_fill += 1
        if self.detrend_fill == self.config.detrend_steps:
            segment = self.detrend_buffer.astype(np.float64)
            time = np.arange(self.config.detrend_steps, dtype=np.float64)
            centered = time - time.mean()
            slopes = np.einsum("t,bts->bs", centered, segment) / np.sum(np.square(centered))
            detrended = segment - (
                segment.mean(axis=1)[:, None, :] + slopes[:, None, :] * centered[None, :, None]
            )
            for offset in range(self.config.detrend_steps):
                detrended_phase = offset % self.config.period_steps
                values_at_offset = detrended[:, offset]
                self.detrended_phase_sum[:, detrended_phase] += values_at_offset
                self.detrended_phase_sq_sum[:, detrended_phase] += np.square(values_at_offset)
                self.detrended_phase_count[:, detrended_phase] += 1
            self.detrend_fill = 0

        self.tail_buffer[:, self.tail_index] = values
        self.tail_index = (self.tail_index + 1) % self.config.tail_steps
        if self.early_fill < self.config.early_steps:
            self.early_buffer[:, self.early_fill] = values
            self.early_fill += 1
        self.selection_buffer[:, self.selection_index] = values
        self.selection_index = (self.selection_index + 1) % self.config.selection_tail_steps

        self.welch_buffer[:, self.welch_fill] = values
        self.welch_fill += 1
        if self.welch_fill == self.config.welch_steps:
            self._consume_welch_window()
            overlap = self.config.welch_overlap_steps
            if overlap:
                self.welch_buffer[:, :overlap] = self.welch_buffer[:, -overlap:]
            self.welch_fill = overlap
        self.total_steps += 1

    def _consume_welch_window(self) -> None:
        values = self.welch_buffer.astype(np.float64)
        time = np.arange(self.config.welch_steps, dtype=np.float64)
        centered = time - time.mean()
        slopes = np.einsum("t,bts->bs", centered, values) / np.sum(np.square(centered))
        detrended = values - (values.mean(axis=1)[:, None, :] + slopes[:, None, :] * centered[None, :, None])
        spectrum = np.fft.rfft(detrended * self._hann[None, :, None], axis=1)
        self.welch_power_sum += np.square(np.abs(spectrum)) / np.sum(np.square(self._hann))
        self.welch_window_count += 1

    @staticmethod
    def _moments(total: float, square_total: float, count: int) -> tuple[float, float, float]:
        if count <= 0:
            return float("nan"), float("nan"), float("nan")
        mean = total / count
        variance = max(square_total / count - mean * mean, 0.0)
        std = float(np.sqrt(variance))
        return float(mean), std, float(std / np.sqrt(count))

    def _metadata_value(self, metadata: Mapping[str, Sequence[object]], key: str, index: int) -> object:
        values = metadata[key]
        if len(values) != self.config.batch_size:
            raise ValueError(f"metadata field {key!r} has length {len(values)}, expected {self.config.batch_size}")
        return values[index]

    def finalize(
        self,
        *,
        scenario_uid: Sequence[str],
        metadata: Mapping[str, Sequence[object]] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if len(scenario_uid) != self.config.batch_size:
            raise ValueError("scenario_uid length does not match batch_size")
        metadata = metadata or {}
        phase_records: list[dict[str, object]] = []
        for batch_index, uid in enumerate(scenario_uid):
            meta = {key: self._metadata_value(metadata, key, batch_index) for key in metadata}
            for phase in range(self.config.period_steps):
                raw_count = int(self.phase_count[batch_index, phase])
                detrended_count = int(self.detrended_phase_count[batch_index, phase])
                for signal_index, signal_name in enumerate(self.config.signal_names):
                    raw = self._moments(
                        self.raw_phase_sum[batch_index, phase, signal_index],
                        self.raw_phase_sq_sum[batch_index, phase, signal_index],
                        raw_count,
                    )
                    detrended = self._moments(
                        self.detrended_phase_sum[batch_index, phase, signal_index],
                        self.detrended_phase_sq_sum[batch_index, phase, signal_index],
                        detrended_count,
                    )
                    phase_records.append({
                        "scenario_uid": uid,
                        "phase": phase,
                        "signal": signal_name,
                        "raw_count": raw_count,
                        "raw_phase_mean": raw[0],
                        "raw_phase_std": raw[1],
                        "raw_standard_error": raw[2],
                        "detrended_count": detrended_count,
                        "detrended_phase_mean": detrended[0],
                        "detrended_phase_std": detrended[1],
                        "detrended_standard_error": detrended[2],
                        **meta,
                    })
        phase_frame = pd.DataFrame(phase_records)

        frequencies = np.fft.rfftfreq(self.config.welch_steps, d=self.config.dt)
        target = 1.0 / (self.config.period_steps * self.config.dt)
        frequency_records: list[dict[str, object]] = []
        count = self.welch_window_count
        for batch_index, uid in enumerate(scenario_uid):
            meta = {key: self._metadata_value(metadata, key, batch_index) for key in metadata}
            for signal_index, signal_name in enumerate(self.config.signal_names):
                power = self.welch_power_sum[batch_index, :, signal_index] / max(count, 1)
                record: dict[str, object] = {
                    "scenario_uid": uid,
                    "signal": signal_name,
                    "welch_window_count": count,
                    "target_frequency_hz": target,
                    **meta,
                }
                for harmonic in (1, 2, 3):
                    index = int(np.argmin(np.abs(frequencies - harmonic * target)))
                    record[f"harmonic_{harmonic}_nearest_frequency_hz"] = float(frequencies[index])
                    record[f"harmonic_{harmonic}_psd"] = float(power[index])
                median = float(np.median(power[1:])) if power.size > 1 else float("nan")
                record["median_non_dc_psd"] = median
                record["target_to_median_psd_ratio"] = float(power[np.argmin(np.abs(frequencies - target))] / max(median, 1e-18))
                frequency_records.append(record)
        frequency_frame = pd.DataFrame(frequency_records)

        tail_count = min(self.total_steps, self.config.tail_steps)
        selection_count = min(self.total_steps, self.config.selection_tail_steps)
        tail_values = _ordered_ring(self.tail_buffer, self.tail_index, tail_count)
        selection_values = _ordered_ring(self.selection_buffer, self.selection_index, selection_count)

        def records_from_window(values: np.ndarray, start_step: int) -> pd.DataFrame:
            records: list[dict[str, object]] = []
            for batch_index, uid in enumerate(scenario_uid):
                meta = {key: self._metadata_value(metadata, key, batch_index) for key in metadata}
                for offset in range(values.shape[1]):
                    records.append({
                        "scenario_uid": uid,
                        "step": start_step + offset,
                        **{
                            signal_name: float(values[batch_index, offset, signal_index])
                            for signal_index, signal_name in enumerate(self.config.signal_names)
                        },
                        **meta,
                    })
            return pd.DataFrame(records)

        tail_frame = records_from_window(tail_values, self.total_steps - tail_count + 1)
        selection_frame = records_from_window(selection_values, self.total_steps - selection_count + 1)
        return phase_frame, frequency_frame, tail_frame, selection_frame

    def finalize_wide(
        self,
        *,
        scenario_uid: Sequence[str],
        metadata: Mapping[str, Sequence[object]] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Vectorized artifact form used for formal 1024-scenario rollouts."""
        if len(scenario_uid) != self.config.batch_size:
            raise ValueError("scenario_uid length does not match batch_size")
        metadata = metadata or {}
        batch, period = self.config.batch_size, self.config.period_steps
        raw_count = self.phase_count
        detrended_count = self.detrended_phase_count
        raw_mean = self.raw_phase_sum / np.maximum(raw_count[..., None], 1)
        raw_variance = np.maximum(
            self.raw_phase_sq_sum / np.maximum(raw_count[..., None], 1) - np.square(raw_mean), 0
        )
        detrended_mean = self.detrended_phase_sum / np.maximum(detrended_count[..., None], 1)
        detrended_variance = np.maximum(
            self.detrended_phase_sq_sum / np.maximum(detrended_count[..., None], 1)
            - np.square(detrended_mean),
            0,
        )
        phase_data: dict[str, object] = {
            "scenario_uid": np.repeat(np.asarray(scenario_uid, dtype=object), period),
            "phase": np.tile(np.arange(period, dtype=np.int64), batch),
            "raw_count": raw_count.reshape(-1),
            "detrended_count": detrended_count.reshape(-1),
        }
        for signal_index, signal in enumerate(self.config.signal_names):
            raw_std = np.sqrt(raw_variance[..., signal_index])
            detrended_std = np.sqrt(detrended_variance[..., signal_index])
            phase_data[f"{signal}_raw_phase_mean"] = raw_mean[..., signal_index].reshape(-1)
            phase_data[f"{signal}_raw_phase_std"] = raw_std.reshape(-1)
            phase_data[f"{signal}_raw_standard_error"] = (
                raw_std / np.sqrt(np.maximum(raw_count, 1))
            ).reshape(-1)
            phase_data[f"{signal}_detrended_phase_mean"] = detrended_mean[..., signal_index].reshape(-1)
            phase_data[f"{signal}_detrended_phase_std"] = detrended_std.reshape(-1)
            phase_data[f"{signal}_detrended_standard_error"] = (
                detrended_std / np.sqrt(np.maximum(detrended_count, 1))
            ).reshape(-1)
        for key, values in metadata.items():
            if len(values) != batch:
                raise ValueError(f"metadata field {key!r} has invalid length")
            phase_data[key] = np.repeat(np.asarray(values), period)
        phase_frame = pd.DataFrame(phase_data)

        frequencies = np.fft.rfftfreq(self.config.welch_steps, d=self.config.dt)
        target = 1.0 / (period * self.config.dt)
        frequency_records = []
        for batch_index, uid in enumerate(scenario_uid):
            meta = {key: self._metadata_value(metadata, key, batch_index) for key in metadata}
            for signal_index, signal in enumerate(self.config.signal_names):
                power = self.welch_power_sum[batch_index, :, signal_index] / max(self.welch_window_count, 1)
                median = float(np.median(power[1:])) if power.size > 1 else float("nan")
                record: dict[str, object] = {
                    "scenario_uid": uid,
                    "signal": signal,
                    "welch_window_count": self.welch_window_count,
                    "target_frequency_hz": target,
                    "median_non_dc_psd": median,
                    **meta,
                }
                for harmonic in (1, 2, 3):
                    index = int(np.argmin(np.abs(frequencies - harmonic * target)))
                    record[f"harmonic_{harmonic}_nearest_frequency_hz"] = float(frequencies[index])
                    record[f"harmonic_{harmonic}_psd"] = float(power[index])
                record["target_to_median_psd_ratio"] = float(
                    record["harmonic_1_psd"] / max(median, 1e-18)
                )
                frequency_records.append(record)
        frequency_frame = pd.DataFrame(frequency_records)

        def wide_window(buffer: np.ndarray, index: int, length: int) -> pd.DataFrame:
            count = min(self.total_steps, length)
            values = _ordered_ring(buffer, index, count)
            data: dict[str, object] = {
                "scenario_uid": np.repeat(np.asarray(scenario_uid, dtype=object), count),
                "step": np.tile(np.arange(self.total_steps - count + 1, self.total_steps + 1), batch),
            }
            for signal_index, signal in enumerate(self.config.signal_names):
                data[signal] = values[..., signal_index].reshape(-1)
            for key, meta_values in metadata.items():
                data[key] = np.repeat(np.asarray(meta_values), count)
            return pd.DataFrame(data)

        return (
            phase_frame,
            frequency_frame,
            wide_window(self.tail_buffer, self.tail_index, self.config.tail_steps),
            wide_window(self.selection_buffer, self.selection_index, self.config.selection_tail_steps),
        )

    def write(
        self,
        output_directory: str | Path,
        *,
        scenario_uid: Sequence[str],
        metadata: Mapping[str, Sequence[object]] | None = None,
    ) -> None:
        output = Path(output_directory)
        phase, frequency, tail, selection = self.finalize_wide(scenario_uid=scenario_uid, metadata=metadata)
        early_count = self.early_fill
        early_data: dict[str, object] = {
            "scenario_uid": np.repeat(np.asarray(scenario_uid, dtype=object), early_count),
            "step": np.tile(np.arange(1, early_count + 1), self.config.batch_size),
        }
        for signal_index, signal in enumerate(self.config.signal_names):
            early_data[signal] = self.early_buffer[:, :early_count, signal_index].reshape(-1)
        for key, meta_values in (metadata or {}).items():
            if len(meta_values) != self.config.batch_size:
                raise ValueError(f"metadata field {key!r} has invalid length")
            early_data[key] = np.repeat(np.asarray(meta_values), early_count)
        early = pd.DataFrame(early_data)
        atomic_write_dataframe(phase, output / "streaming_phase_profiles.csv")
        atomic_write_dataframe(frequency, output / "streaming_frequency.csv")
        atomic_write_dataframe(tail, output / "tail_window.csv")
        atomic_write_dataframe(early, output / "early_window.csv")
        atomic_write_dataframe(selection, output / "snapshot_selection_window.csv")
        manifest = asdict(self.config)
        manifest.update({
            "total_steps": self.total_steps,
            "early_steps_recorded": self.early_fill,
            "welch_window_count": self.welch_window_count,
        })
        atomic_write_json(manifest, output / "streaming_manifest.json")
