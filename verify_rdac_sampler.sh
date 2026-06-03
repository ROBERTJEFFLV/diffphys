#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_TOOLS_DIR="${ROOT_DIR}/rl-tools"
BUILD_DIR="${BUILD_DIR:-/tmp/raptor_diff_rdac_build}"
BIN="${BUILD_DIR}/src/foundation_policy/foundation_policy_diff_pre_training"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/verify_rdac_sampler_$(date +%Y%m%d_%H%M%S)}"
SAMPLES="${SAMPLES:-4096}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TRAIN_STEPS_LIST="${TRAIN_STEPS_LIST:-5000 20000}"
CURVE_STRIDE="${CURVE_STRIDE:-100}"
JOBS="${JOBS:-2}"

mkdir -p "${OUT_DIR}"

if (( SAMPLES < 1000 || SAMPLES > 10000 )); then
    echo "SAMPLES must be between 1000 and 10000; got ${SAMPLES}" >&2
    exit 1
fi

if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    cmake -S "${RL_TOOLS_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
fi

echo "[build] foundation_policy_diff_pre_training"
cmake --build "${BUILD_DIR}" --target foundation_policy_diff_pre_training -j "${JOBS}"

echo "[sampler] dumping ${SAMPLES} broad samples"
DUMP_CSV="${OUT_DIR}/sampler_dump_${SAMPLES}.csv"
DUMP_STDOUT="${OUT_DIR}/sampler_dump_${SAMPLES}.stdout"
"${BIN}" \
    --diff-model euler \
    --sample-dynamics \
    --sampled-dynamics-level broad \
    --balanced-dynamics-sampling \
    --horizon 1 \
    --batch-size "${BATCH_SIZE}" \
    --sampler-dump-samples "${SAMPLES}" \
    --sampler-dump-path "${DUMP_CSV}" \
    > "${DUMP_STDOUT}"

python3 - "${DUMP_CSV}" "${SAMPLES}" "${OUT_DIR}/sampler_summary.txt" <<'PY'
import csv
import math
import sys
from collections import Counter

path, expected_s, summary_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(path, newline="") as f:
    rows = list(csv.DictReader(f))
if len(rows) != expected_s:
    raise SystemExit(f"sampler row count mismatch: got {len(rows)}, expected {expected_s}")

ranges = {
    "mass": (0.02, 0.25),
    "thrust_to_weight_ratio": (1.5, 5.0),
    "torque_to_inertia_ratio": (40.0, 1200.0),
    "sampled_tau_rise": (0.03, 0.10),
    "sampled_tau_fall": (0.03, 0.30),
    "sampled_curve_shape": (0.05, 0.95),
    "rotor_torque_constant": (0.005, 0.05),
}

lines = []
lines.append(f"samples={len(rows)}")
for key, (lo, hi) in ranges.items():
    values = [float(r[key]) for r in rows]
    if not all(math.isfinite(v) for v in values):
        raise SystemExit(f"{key} has non-finite values")
    mn, mx = min(values), max(values)
    lines.append(f"{key}: min={mn:.9g} max={mx:.9g} allowed=[{lo}, {hi}]")
    eps = 1e-6
    if mn < lo - eps or mx > hi + eps:
        raise SystemExit(f"{key} out of range: min={mn}, max={mx}, allowed=[{lo}, {hi}]")

inside = Counter(r["sampled_parameters_inside_allowed_ranges"] for r in rows)
lines.append(f"sampled_parameters_inside_allowed_ranges={dict(inside)}")
if inside.get("true", 0) != len(rows):
    raise SystemExit(f"not all sampled parameters are inside allowed ranges: {inside}")

bin_columns = [
    "dynamics_size_mass_bin",
    "dynamics_thrust_to_weight_bin",
    "dynamics_torque_to_inertia_bin",
    "dynamics_motor_delay_bin",
    "dynamics_curve_shape_bin",
]
for key in bin_columns:
    counts = Counter(int(float(r[key])) for r in rows)
    missing = [i for i in range(4) if counts.get(i, 0) == 0]
    lines.append(f"{key}: {dict(sorted(counts.items()))}")
    if missing:
        raise SystemExit(f"{key} missing bins: {missing}")
    if max(counts.values()) - min(counts.values()) > 1:
        raise SystemExit(f"{key} is not balanced: {counts}")

group_counts = Counter(int(float(r["dynamics_group_key"])) for r in rows)
lines.append(f"dynamics_group_count={len(group_counts)}")
lines.append(f"dynamics_group_count_min={min(group_counts.values())} max={max(group_counts.values())}")
if len(rows) >= 4**5 and len(group_counts) != 4**5:
    raise SystemExit(f"expected all {4**5} 5D bin groups, saw {len(group_counts)}")
if max(group_counts.values()) - min(group_counts.values()) > 1:
    raise SystemExit("5D dynamics group counts are not balanced")

with open(summary_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
PY

run_smoke() {
    local horizon="$1"
    local log_csv="${OUT_DIR}/smoke_h${horizon}.csv"
    local stdout="${OUT_DIR}/smoke_h${horizon}.stdout"
    echo "[smoke] H=${horizon}"
    "${BIN}" \
        --diff-model euler \
        --steps 1 \
        --sample-dynamics \
        --sampled-dynamics-level broad \
        --balanced-dynamics-sampling \
        --horizon "${horizon}" \
        --batch-size "${BATCH_SIZE}" \
        --log-path "${log_csv}" \
        > "${stdout}"
    python3 - "${horizon}" "${stdout}" "${log_csv}" <<'PY'
import csv
import math
import sys

horizon, stdout_path, csv_path = sys.argv[1], sys.argv[2], sys.argv[3]
stdout = open(stdout_path).read()
if "actor_output_dim=4" not in stdout or "action_dim=4" not in stdout:
    raise SystemExit(f"H={horizon}: actor/action dimensions are not 4D")
with open(csv_path, newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit(f"H={horizon}: no smoke CSV rows")
row = rows[-1]
required_true = [
    "single_step_state_finite",
    "single_step_action_finite",
    "single_step_reward_finite",
    "single_step_done_finite",
    "dynamics_batch_balanced",
    "sampled_parameters_inside_allowed_ranges",
    "hidden_dynamics_separable",
]
for key in required_true:
    if row[key] != "true":
        raise SystemExit(f"H={horizon}: {key}={row[key]}")
grad_keys = [
    "rdac_encoder_grad_norm_before_clip",
    "rdac_gru_grad_norm_before_clip",
    "rdac_actor_head_grad_norm_before_clip",
]
for key in grad_keys:
    value = float(row[key])
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f"H={horizon}: {key} invalid: {row[key]}")
print(
    f"H={horizon} ok: actor_output_dim=4 "
    f"encoder_grad={row['rdac_encoder_grad_norm_before_clip']} "
    f"gru_grad={row['rdac_gru_grad_norm_before_clip']} "
    f"actor_grad={row['rdac_actor_head_grad_norm_before_clip']} "
    f"hidden_sep={row['hidden_dynamics_separation_ratio']}"
)
PY
}

run_smoke 1
run_smoke 16

read -r -a TRAIN_STEPS_ARRAY <<< "${TRAIN_STEPS_LIST}"
for steps in "${TRAIN_STEPS_ARRAY[@]}"; do
    [[ -z "${steps}" ]] && continue
    train_csv="${OUT_DIR}/train_h16_${steps}.csv"
    train_stdout="${OUT_DIR}/train_h16_${steps}.stdout"
    curve_csv="${OUT_DIR}/train_h16_${steps}_curves.csv"
    curve_png="${OUT_DIR}/train_h16_${steps}_curves.png"
    echo "[train] H=16 steps=${steps}"
    "${BIN}" \
        --diff-model euler \
        --steps "${steps}" \
        --sample-dynamics \
        --sampled-dynamics-level broad \
        --balanced-dynamics-sampling \
        --horizon 16 \
        --batch-size "${BATCH_SIZE}" \
        --log-path "${train_csv}" \
        > "${train_stdout}"
    python3 - "${train_csv}" "${curve_csv}" "${curve_png}" "${CURVE_STRIDE}" <<'PY'
import csv
import math
import statistics
import sys

train_csv, curve_csv, curve_png, stride_s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
stride = max(1, int(stride_s))
with open(train_csv, newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit(f"no training rows in {train_csv}")

curve_fields = [
    "step",
    "total_loss_mean",
    "training_success_rate",
    "action_saturation_ratio",
    "hidden_dynamics_separation_ratio",
    "rdac_actor_head_grad_norm_before_clip",
    "rdac_actor_head_grad_norm_after_clip",
]
for i, row in enumerate(rows):
    for key in curve_fields[1:]:
        value = float(row[key])
        if not math.isfinite(value):
            raise SystemExit(f"{train_csv}: non-finite {key} at row {i}: {row[key]}")
    if row["single_step_state_finite"] != "true" or row["single_step_action_finite"] != "true":
        raise SystemExit(f"{train_csv}: non-finite single-step smoke flag at row {i}")
    if row["sampled_parameters_inside_allowed_ranges"] != "true":
        raise SystemExit(f"{train_csv}: sampled parameters out of range at row {i}")

selected = []
for i, row in enumerate(rows):
    if i == 0 or i == len(rows) - 1 or i % stride == 0:
        selected.append(row)

with open(curve_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=curve_fields)
    writer.writeheader()
    for row in selected:
        writer.writerow({key: row[key] for key in curve_fields})

def tail_mean(key, n=100):
    vals = [float(r[key]) for r in rows[-min(n, len(rows)):]]
    return statistics.fmean(vals)

last = rows[-1]
print(
    f"steps={len(rows)} final_loss={float(last['total_loss_mean']):.6g} "
    f"tail100_loss={tail_mean('total_loss_mean'):.6g} "
    f"tail100_success={tail_mean('training_success_rate'):.6g} "
    f"tail100_action_saturation={tail_mean('action_saturation_ratio'):.6g} "
    f"tail100_hidden_separation={tail_mean('hidden_dynamics_separation_ratio'):.6g} "
    f"curve_csv={curve_csv}"
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [int(r["step"]) for r in selected]
    series = [
        ("total_loss_mean", "loss"),
        ("training_success_rate", "success"),
        ("action_saturation_ratio", "action_saturation"),
        ("hidden_dynamics_separation_ratio", "hidden_separation"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title) in zip(axes.ravel(), series):
        ys = [float(r[key]) for r in selected]
        ax.plot(xs, ys)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(curve_png, dpi=150)
    print(f"curve_png={curve_png}")
except Exception as exc:
    print(f"curve_png_unavailable={exc}")
PY
done

echo "[done] artifacts: ${OUT_DIR}"
