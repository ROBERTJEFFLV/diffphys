#!/usr/bin/env python3
"""Common held-out evaluation of the two final, from-zero hover Actors."""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from diffphys.compiled_physics import make_simulator
from diffphys.env_raptor import RaptorParams
from diffphys.hover_time_experiment import (HoverTimeConfig, experiment, _quality_samples,
                                           settling_proxy)
from diffphys.response_noise import DisturbanceConfig
from diffphys.response_task import TaskLossConfig, rollout, scenario_costs
from diffphys import response_training as training


def initial_checkpoint(work):
    paths = list((work / "checkpoints").glob("eval_00000000_*.pt"))
    if len(paths) != 1:
        raise ValueError("expected one saved random-initialization evaluation")
    return paths[0]


def verify_pair(folder, updates):
    saved = [torch.load(folder / arm / "latest.pt", map_location="cpu", weights_only=True)
             for arm in ("baseline", "time")]
    starts = [torch.load(initial_checkpoint(folder / arm), map_location="cpu", weights_only=True)
              for arm in ("baseline", "time")]
    if starts[0]["model_sha256"] != starts[1]["model_sha256"]:
        raise ValueError("arms did not start from identical random weights")
    for value in starts:
        if value["optimizer"]["state"] or "initialization" in value["progress"]:
            raise ValueError("comparison must start from zero with empty Adam")
    normalized = []
    for index, value in enumerate(saved):
        if value["progress"]["updates"] != updates or value["next_update"] != updates:
            raise ValueError("both arms must finish the requested accepted update count")
        if any(state["step"].item() != updates for state in value["optimizer"]["state"].values()):
            raise ValueError("Adam step count does not match accepted updates")
        binding = copy.deepcopy(value["binding"])
        cfg = binding["protocol"]["hover_time_experiment"]
        if cfg["weight"] != float(index):
            raise ValueError("wrong ablation weight")
        cfg["weight"] = 0.
        normalized.append(binding)
    if normalized[0] != normalized[1]:
        raise ValueError("an additional source/configuration difference confounds the comparison")
    with (folder / "baseline/history.jsonl").open() as a, (folder / "time/history.jsonl").open() as b:
        count = 0
        from itertools import zip_longest
        for first, second in zip_longest(a, b):
            if first is None or second is None:
                raise ValueError("different training history lengths")
            x, y = json.loads(first), json.loads(second)
            count += 1
            if (x["update"] != count or y["update"] != count
                    or x["train_seeds"] != y["train_seeds"]
                    or x["group_balance"] != y["group_balance"]):
                raise ValueError("unmatched TRAIN seeds or physical scene groups")
        if count != updates:
            raise ValueError("training history does not contain every accepted update")
    return saved, starts[0]["model_sha256"]


def state_hash(state):
    digest = hashlib.sha256()
    for field in fields(state):
        value = getattr(state, field.name).detach().cpu().contiguous()
        digest.update(field.name.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def evaluate_model(path, weight, initial, horizon, device):
    config = HoverTimeConfig(weight=weight)
    with experiment(config):
        policy, saved = training.load_policy_checkpoint(path, device, torch.float32)
        protocol = saved["binding"]["protocol"]
        loss = TaskLossConfig(**protocol["loss"])
        simulator = make_simulator(RaptorParams(**protocol["environment_params"]), device)
        report = training.evaluate(policy, simulator, initial, horizon, loss)
        # One additional read-only trace retains paired per-scene measurements.
        trace = rollout(policy, simulator, initial, horizon)
        first, first_alive = _quality_samples(trace, config, birth=True)
        rest, rest_alive = _quality_samples(trace, config)
        ratio, alive = torch.cat((first, rest)), torch.cat((first_alive, rest_alive))
        proxy = settling_proxy(ratio, alive, dt=policy.config.dt)
        inside = (ratio <= 1) & alive
        index = torch.arange(horizon + 1, device=device)[:, None]
        last = torch.where(~inside, index, -1).amax(0)
        hard = (last + 1).clamp_max(horizon) * policy.config.dt
        values = {
            "hover_objective": scenario_costs(trace, loss),
            "proxy_seconds": proxy,
            "settling_seconds_censored": hard,
            "success": inside[-1],
            "boundary_failure": ~alive[-1],
        }
        per_scene = {key: value.cpu().tolist() for key, value in values.items()}
        return report, per_scene


def plots(folder, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (("hover_objective", "Original hover objective"),
               ("hover_time_proxy_seconds", "Settling proxy [s]"),
               ("hover_settling_time_seconds", "Censored settling time [s]"),
               ("hover_success_fraction", "Finite-horizon hover success"),
               ("raptor_share_terminated", "Boundary-exit fraction"),
               ("position_rms", "Position RMS [m]"))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for arm, color in (("baseline", "#2878b5"), ("time", "#e58224")):
        rows = []
        with (folder / arm / "evaluation.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                rows.append({key: row[key] for key in ("update", *(key for key, _ in metrics))})
        for axis, (key, label) in zip(axes.flat, metrics):
            axis.plot([a["update"] for a in rows], [a[key] for a in rows],
                      color=color, label=arm, linewidth=1.6)
            axis.set(xlabel="Accepted updates", ylabel=label)
            axis.grid(alpha=.25)
    axes[0, 0].legend()
    fig.suptitle("Same random initialization, scenes and noise; one training seed")
    fig.savefig(folder / "learning_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, (key, label) in zip(axes.flat, metrics):
        axis.bar(("baseline", "+time"), [reports[a][key] for a in ("baseline", "time")],
                 color=("#2878b5", "#e58224"))
        axis.set_title(label)
        axis.grid(axis="y", alpha=.25)
    fig.suptitle("Final checkpoint comparison on the same 512 held-out scenes")
    fig.savefig(folder / "heldout_comparison.png", dpi=160)
    plt.close(fig)


def analyze(folder, device="cuda"):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    updates = manifest["updates_per_arm"]
    saved, initial_hash = verify_pair(folder, updates)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device)
    cfg = saved[0]["binding"]["protocol"]
    initial = training.sample_pool(manifest["heldout_scenes_per_bank"], manifest["heldout_seeds"],
        horizon=saved[0]["binding"]["horizon"], dtype=torch.float32, device=device,
        disturbances=DisturbanceConfig(**cfg["disturbances"]))
    digest = state_hash(initial)
    reports, scenarios = {}, {}
    for arm, weight in (("baseline", 0.), ("time", 1.)):
        reports[arm], scenarios[arm] = evaluate_model(folder / arm / "latest.pt", weight,
            initial, saved[0]["binding"]["horizon"], device)
        if state_hash(initial) != digest:
            raise ValueError("evaluation mutated the common initial state or tapes")
    names = list(scenarios["baseline"])
    with (folder / "paired_heldout.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scene_id", *(a + "_" + n
                               for a in ("baseline", "time") for n in names)])
        writer.writeheader()
        for index in range(len(initial.mass)):
            writer.writerow({"scene_id": index, **{a + "_" + n: scenarios[a][n][index]
                              for a in ("baseline", "time") for n in names}})
    keys = ("hover_objective", "hover_time_proxy_seconds", "hover_settling_time_seconds",
            "hover_success_fraction", "position_rms", "velocity_rms", "omega_rms",
            "raptor_share_terminated")
    result = {"complete": True, "accepted_updates_per_arm": updates,
              "initialization": "from zero; identical random Actor weights and empty Adam",
              "initial_model_sha256": initial_hash, "matched_train_seeds_and_groups": True,
              "only_configuration_difference": "hover time weight 0 versus 1",
              "heldout_scene_count": len(initial.mass), "heldout_state_sha256": digest,
              "reports": reports,
              "time_minus_baseline": {key: reports["time"][key]-reports["baseline"][key] for key in keys},
              "limitations": manifest["limitations"]}
    training.atomic_json(folder / "comparison.json", result)
    plots(folder, reports)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(analyze(args.experiment_dir, args.device), indent=2, allow_nan=False))
