from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tools.analyze_arm_d_screen import (
    ARMS,
    CAUSAL_ARMS,
    CAUSAL_COMPARISONS,
    HORIZONS,
    PROMOTION_ARMS,
    RESET_MAPPINGS,
    SEEDS,
    _artifact_path,
    _bootstrap_intervals,
    _decision_gates,
    _effect_tables,
    _load_training_safety,
    _roots,
    _validate_matlab_linkage,
    _validate_samples,
    _validate_training_manifest,
)
from tools.run_continuity_cadence_screen import _command, _write_run_manifest
from tools.validate_continuity_cadence_label_parity import DEFAULT_ARMS, _parse_arms


def _paired_frames(count: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios: dict[str, object] = {
        "scenario_id": np.arange(1, count + 1),
        "scenario_uid": [f"scenario-{index}" for index in range(count)],
    }
    samples: dict[str, object] = {"sample_id": np.arange(1, count + 1)}
    for index, (scenario_column, sample_column) in enumerate(RESET_MAPPINGS.items()):
        values = np.arange(count, dtype=np.float64) + 0.01 * (index + 1)
        scenarios[scenario_column] = values
        samples[sample_column] = values.copy()
    return pd.DataFrame(scenarios), pd.DataFrame(samples)


class LabelParityArmParserTest(unittest.TestCase):
    def test_accepts_arm_d_without_changing_the_legacy_default_order(self) -> None:
        self.assertEqual(DEFAULT_ARMS, ("A", "B", "C"))
        self.assertEqual(_parse_arms("A,B,D"), ("A", "B", "D"))
        self.assertEqual(_parse_arms("d"), ("D",))

    def test_rejects_unknown_or_duplicate_arms(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_arms("A,E")
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_arms("D,D")


class ArmDArtifactAndPairingTest(unittest.TestCase):
    def test_resolves_shared_and_arm_specific_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared = root / "shared/seed_7/arm_D/samples.csv"
            shared.parent.mkdir(parents=True)
            shared.write_text("sample_id\n1\n", encoding="utf-8")
            self.assertEqual(
                _artifact_path(root / "shared", seed=7, arm="D", relative="samples.csv"),
                shared.resolve(),
            )

            specific = root / "specific/seed_17/train.csv"
            specific.parent.mkdir(parents=True)
            specific.write_text("step\n1\n", encoding="utf-8")
            self.assertEqual(
                _artifact_path(root / "specific", seed=17, arm="D", relative="train.csv"),
                specific.resolve(),
            )

    def test_causal_roots_do_not_require_arm_a(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            shared = Path(temporary)
            resolved = _roots(
                shared=shared,
                overrides={"A": None, "B": None, "D": None},
                label="training",
                arms=CAUSAL_ARMS,
            )
        self.assertEqual(set(resolved), {"B", "D"})

    def test_strict_sample_id_and_parameter_pairing(self) -> None:
        scenarios, samples = _paired_frames()
        errors = _validate_samples(samples, scenarios, source=Path("fixture.csv"))
        self.assertEqual(max(errors.values()), 0.0)

        reordered = samples.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(RuntimeError, "sample IDs/order"):
            _validate_samples(reordered, scenarios, source=Path("reordered.csv"))

        fractional_id = samples.copy()
        fractional_id["sample_id"] = fractional_id["sample_id"].astype(np.float64)
        fractional_id.loc[0, "sample_id"] = 1.5
        with self.assertRaisesRegex(RuntimeError, "sample IDs/order"):
            _validate_samples(
                fractional_id, scenarios, source=Path("fractional_id.csv")
            )

        changed = samples.copy()
        changed.loc[0, "mass_kg"] += 1.0e-6
        with self.assertRaisesRegex(RuntimeError, "scenario parameter mismatch"):
            _validate_samples(changed, scenarios, source=Path("changed.csv"))

        nonfinite = samples.copy()
        nonfinite.loc[0, "mass_kg"] = np.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite paired scenario field"):
            _validate_samples(nonfinite, scenarios, source=Path("nonfinite.csv"))


class ArmDStatisticsTest(unittest.TestCase):
    @staticmethod
    def _effect_rows(
        comparisons: tuple[tuple[str, str], ...],
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for left, right in comparisons:
            for horizon in HORIZONS:
                rows.append(
                    {
                        "left_arm": left,
                        "right_arm": right,
                        "horizon": horizon,
                        "stratum": "all",
                        "point_difference": 0.0,
                        "crossed_ci_low": -0.001,
                        "crossed_ci_high": 0.001,
                        "positive_seed_count": 0,
                        "minimum_seed_difference": 0.0,
                        "crossed_equivalent_pm0p5pp": True,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _safety_rows(arms: tuple[str, ...]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "seed": seed,
                    "arm": arm,
                    "schedule_valid": True,
                    "invalid_rollouts": 0,
                    "rejected_updates": 0,
                    "gradient_skips": 0,
                }
                for seed in SEEDS
                for arm in arms
            ]
        )

    def test_bootstrap_is_deterministic_and_effect_tables_are_paired(self) -> None:
        difference = np.asarray(
            [[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]
        )
        first = _bootstrap_intervals(difference, n_bootstrap=200, seed=20260806)
        second = _bootstrap_intervals(difference, n_bootstrap=200, seed=20260806)
        self.assertEqual(first, second)

        scenario_uids = [f"scenario-{index}" for index in range(4)]
        metric_rows: list[dict[str, object]] = []
        for seed in SEEDS:
            for arm in ARMS:
                for horizon in HORIZONS:
                    for scenario_uid in scenario_uids:
                        metric_rows.append(
                            {
                                "seed": seed,
                                "arm": arm,
                                "horizon": horizon,
                                "scenario_uid": scenario_uid,
                                "success": int(arm == "D"),
                            }
                        )
        effects, seed_effects, flips = _effect_tables(
            pd.DataFrame(metric_rows),
            scenario_uids=scenario_uids,
            feasible_mask=np.asarray([True, True, False, False]),
            n_bootstrap=100,
            bootstrap_seed=20260806,
        )
        self.assertEqual(len(effects), 8)
        self.assertEqual(len(seed_effects), 24)
        self.assertEqual(len(flips), 32)
        db500 = effects[
            effects["left_arm"].eq("D")
            & effects["right_arm"].eq("B")
            & effects["horizon"].eq(500)
            & effects["stratum"].eq("all")
        ].iloc[0]
        self.assertEqual(float(db500["point_difference"]), 1.0)
        self.assertEqual(float(db500["crossed_ci_low"]), 1.0)
        pooled = flips[
            flips["left_arm"].eq("D")
            & flips["right_arm"].eq("B")
            & flips["horizon"].eq(500)
            & flips["stratum"].eq("all")
            & flips["seed"].eq("pooled")
        ].iloc[0]
        self.assertEqual(int(pooled["left_success_right_failure"]), 12)
        self.assertEqual(int(pooled["left_failure_right_success"]), 0)

    def test_equivalence_requires_the_full_crossed_interval_within_pm0p5pp(self) -> None:
        effects = self._effect_rows((("D", "A"), ("D", "B")))
        effects["crossed_equivalent_pm0p5pp"] = False
        safety = self._safety_rows(PROMOTION_ARMS)
        gates, decision = _decision_gates(
            effects, safety, reset_pairing_valid=True, stage="promotion"
        )
        equivalence = gates[
            gates["gate"].eq("d_b_h500_crossed_equivalence_pm0p5pp")
        ].iloc[0]
        self.assertTrue(bool(equivalence["passed"]))
        self.assertEqual(
            decision["cvar_event_package_interpretation"],
            "no_material_cvar_event_package_effect_within_pm0p5pp",
        )
        self.assertFalse(bool(decision["cvar_event_package_supported"]))
        self.assertTrue(bool(decision["promotion_evaluated"]))

    def test_causal_stage_uses_only_b_d_and_safety_covers_both_arms(self) -> None:
        effects = self._effect_rows(CAUSAL_COMPARISONS)
        safety = self._safety_rows(CAUSAL_ARMS)
        safety.loc[
            safety["arm"].eq("B") & safety["seed"].eq(SEEDS[0]),
            "rejected_updates",
        ] = 1
        gates, decision = _decision_gates(
            effects, safety, reset_pairing_valid=True, stage="causal"
        )
        self.assertEqual(decision["stage"], "causal")
        self.assertFalse(bool(decision["training_safety_passed"]))
        self.assertFalse(bool(decision["promotion_evaluated"]))
        self.assertIsNone(decision["promotion_passed"])
        self.assertEqual(
            decision["cvar_event_package_interpretation"],
            "hard_safety_or_provenance_gate_failed",
        )
        self.assertFalse(gates["category"].eq("promotion").any())
        rejected = gates[
            gates["gate"].eq("all_stage_arms_zero_rejected_updates")
        ].iloc[0]
        self.assertEqual(int(rejected["value"]), 1)
        self.assertFalse(bool(rejected["passed"]))

    def test_promotion_safety_fails_when_arm_a_is_nonfinite(self) -> None:
        effects = self._effect_rows((("D", "A"), ("D", "B")))
        safety = self._safety_rows(PROMOTION_ARMS)
        safety.loc[
            safety["arm"].eq("A") & safety["seed"].eq(SEEDS[-1]),
            "invalid_rollouts",
        ] = 1
        gates, decision = _decision_gates(
            effects, safety, reset_pairing_valid=True, stage="promotion"
        )
        self.assertFalse(bool(decision["training_safety_passed"]))
        nonfinite = gates[
            gates["gate"].eq("all_stage_arms_zero_nonfinite_rollouts")
        ].iloc[0]
        self.assertEqual(int(nonfinite["value"]), 1)
        self.assertFalse(bool(nonfinite["passed"]))

    def test_positive_performance_does_not_support_claim_when_safety_fails(self) -> None:
        effects = self._effect_rows(CAUSAL_COMPARISONS)
        h500 = effects["horizon"].eq(500)
        effects.loc[h500, ["point_difference", "crossed_ci_low", "crossed_ci_high"]] = (
            0.01,
            0.001,
            0.02,
        )
        safety = self._safety_rows(CAUSAL_ARMS)
        safety.loc[safety["arm"].eq("D"), "gradient_skips"] = 1
        _, decision = _decision_gates(
            effects, safety, reset_pairing_valid=True, stage="causal"
        )
        self.assertTrue(
            bool(decision["cvar_event_package_performance_gates_passed"])
        )
        self.assertFalse(bool(decision["cvar_event_package_supported"]))
        self.assertEqual(
            decision["cvar_event_package_interpretation"],
            "hard_safety_or_provenance_gate_failed",
        )

    def test_causal_effect_tables_emit_only_d_b(self) -> None:
        scenario_uids = [f"scenario-{index}" for index in range(4)]
        rows: list[dict[str, object]] = []
        for seed in SEEDS:
            for arm in CAUSAL_ARMS:
                for horizon in HORIZONS:
                    for scenario_uid in scenario_uids:
                        rows.append(
                            {
                                "seed": seed,
                                "arm": arm,
                                "horizon": horizon,
                                "scenario_uid": scenario_uid,
                                "success": int(arm == "D"),
                            }
                        )
        effects, seed_effects, flips = _effect_tables(
            pd.DataFrame(rows),
            scenario_uids=scenario_uids,
            feasible_mask=np.asarray([True, True, False, False]),
            arms=CAUSAL_ARMS,
            comparisons=CAUSAL_COMPARISONS,
            n_bootstrap=50,
            bootstrap_seed=20260806,
        )
        for frame in (effects, seed_effects, flips):
            self.assertEqual(set(zip(frame["left_arm"], frame["right_arm"])), {("D", "B")})


class ArmDTrainingSafetyTest(unittest.TestCase):
    @staticmethod
    def _rows(arm: str) -> pd.DataFrame:
        episode_horizons = [500] * 75 if arm == "A" else [500] * 59 + [1000] * 8
        rows: list[dict[str, object]] = []
        update = 0
        for episode_horizon in episode_horizons:
            segment_count = episode_horizon // 250
            for segment in range(segment_count):
                reset_boundary = segment == segment_count - 1
                optimizer_boundary = reset_boundary
                tail_boundary = (segment + 1) % 2 == 0
                if arm != "D":
                    tail_boundary = optimizer_boundary
                update += int(optimizer_boundary)
                rows.append(
                    {
                        "physical_steps": 64_000 * (len(rows) + 1),
                        "optimizer_update": update,
                        "episode_target_steps": episode_horizon,
                        "reset_episode_boundary": int(reset_boundary),
                        "optimization_block_boundary": int(optimizer_boundary),
                        "update_applied": int(optimizer_boundary),
                        "first_tail_supervision_segment": int(
                            segment % 2 == 0 if arm == "D" else segment == 0
                        ),
                        "tail_supervision_block_boundary": int(tail_boundary),
                        "tail_supervision_block_horizon": (
                            500 if arm == "D" else episode_horizon
                        ),
                        "rollout_valid": 1,
                        "skip_reason": "",
                    }
                )
        return pd.DataFrame(rows)

    def test_full_d_schedule_and_b_d_reset_pairing_are_safety_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = {arm: root for arm in ARMS}
            for seed in SEEDS:
                for arm in ARMS:
                    run = root / f"seed_{seed}" / f"arm_{arm}"
                    _write_run_manifest(
                        run,
                        seed=seed,
                        arm=arm,
                        command=_command(seed, arm, run),
                    )
                    manifest_path = run / "RUN_MANIFEST.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest.update(
                        {
                            "cuda_available": True,
                            "cuda_device_count": 1,
                            "cuda_device_name": "fixture CUDA device",
                            "torch_cuda_build": str(
                                manifest.get("torch_cuda_build") or "fixture CUDA build"
                            ),
                        }
                    )
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    self._rows(arm).to_csv(run / "train.csv", index=False)
                    reset_bytes = b"paired-reset\n" if arm in {"B", "D"} else b"a-reset\n"
                    (run / "reset_samples.csv").write_bytes(reset_bytes)
                    (run / "checkpoints/model.pt").write_bytes(b"checkpoint")

            safety, _, reset_pairing = _load_training_safety(roots)
            self.assertTrue(reset_pairing)
            d_rows = safety[safety["arm"].eq("D")]
            self.assertTrue(bool(d_rows["schedule_valid"].all()))
            self.assertEqual(int(d_rows["mid_h1000_optimizer_boundaries"].sum()), 0)
            self.assertEqual(set(d_rows["tail_supervision_boundaries"]), {75})

    def test_empty_manifest_is_a_hard_provenance_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "seed_7" / "arm_D"
            run.mkdir(parents=True)
            manifest = run / "RUN_MANIFEST.json"
            manifest.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "training provenance failed"):
                _validate_training_manifest(
                    manifest, run_dir=run, seed=7, arm="D"
                )


class ArmDEvaluationLinkageTest(unittest.TestCase):
    def test_checkpoint_index_links_matlab_outputs_to_training_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_rows: list[dict[str, object]] = []
            eval_rows: list[dict[str, object]] = []
            safety_rows: list[dict[str, object]] = []
            for seed in SEEDS:
                for arm in CAUSAL_ARMS:
                    run = root / f"seed_{seed}" / f"arm_{arm}"
                    run.mkdir(parents=True)
                    sample = run / "samples.csv"
                    model_mat = run / "model.mat"
                    sample.write_text("sample_id\n1\n", encoding="utf-8")
                    model_mat.write_bytes(f"model-{seed}-{arm}".encode())
                    checkpoint_bytes = 1000 + seed
                    checkpoint_sha = hashlib.sha256(
                        f"checkpoint-{seed}-{arm}".encode()
                    ).hexdigest()
                    label = f"seed_{seed}_arm_{arm}_physical_steps_9600000"
                    checkpoint_path = (
                        root / f"seed_{seed}" / f"arm_{arm}" / "checkpoints/model.pt"
                    )
                    checkpoint_rows.append(
                        {
                            "label": label,
                            "seed": seed,
                            "arm": arm,
                            "physical_steps": 9_600_000,
                            "checkpoint_path": checkpoint_path,
                            "checkpoint_sha256": checkpoint_sha,
                            "checkpoint_bytes": checkpoint_bytes,
                            "model_mat_path": model_mat,
                            "model_mat_sha256": hashlib.sha256(
                                model_mat.read_bytes()
                            ).hexdigest(),
                            "sample_output_path": sample,
                            "eval_seed": 1007,
                            "horizon": 10_000,
                        }
                    )
                    eval_rows.append(
                        {
                            "label": label,
                            "checkpoint_path": checkpoint_path,
                            "weights_path": model_mat,
                            "output_path": run / "summary.csv",
                            "sample_output_path": sample,
                            "mat_output_path": run / "metrics.mat",
                            "batch_size": 1024,
                            "eval_seed": 1007,
                            "horizon": 10_000,
                        }
                    )
                    safety_rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "checkpoint_sha256": checkpoint_sha,
                            "checkpoint_bytes": checkpoint_bytes,
                        }
                    )
            checkpoints_path = root / "CHECKPOINTS.csv"
            pd.DataFrame(checkpoint_rows).to_csv(checkpoints_path, index=False)
            pd.DataFrame(eval_rows).to_csv(root / "eval_manifest.csv", index=False)

            linkage, _ = _validate_matlab_linkage(
                {"B": root, "D": root},
                pd.DataFrame(safety_rows),
                arms=CAUSAL_ARMS,
            )
            self.assertEqual(len(linkage), len(SEEDS) * len(CAUSAL_ARMS))
            self.assertTrue(bool(linkage["linkage_valid"].all()))

            checkpoint_rows[0]["checkpoint_sha256"] = "0" * 64
            pd.DataFrame(checkpoint_rows).to_csv(checkpoints_path, index=False)
            with self.assertRaisesRegex(RuntimeError, "checkpoint_sha256"):
                _validate_matlab_linkage(
                    {"B": root, "D": root},
                    pd.DataFrame(safety_rows),
                    arms=CAUSAL_ARMS,
                )


if __name__ == "__main__":
    unittest.main()
