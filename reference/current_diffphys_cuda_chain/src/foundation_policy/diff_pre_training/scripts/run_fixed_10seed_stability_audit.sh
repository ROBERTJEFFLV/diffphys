#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_TOOLS_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${RL_TOOLS_DIR}"

BUILD_DIR="${BUILD_DIR:-build_diff}"
RUN_NAME="${RUN_NAME:-fixed_vstrong_10seed_audit}"
STEPS="${STEPS:-6000}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
EVAL_SEEDS="${EVAL_SEEDS:-10000 10001 10002}"
EVAL_MODELS="${EVAL_MODELS:-euler l2f}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
EVAL_HORIZON="${EVAL_HORIZON:-128}"
HORIZON_STAGE_STEPS="${HORIZON_STAGE_STEPS:-1000}"
STATE_CURRICULUM_STAGE_STEPS="${STATE_CURRICULUM_STAGE_STEPS:-1000}"
ACTOR_GRAD_CLIP="${ACTOR_GRAD_CLIP:-50}"
TERMINAL_RAMP_AFTER_HORIZON_CHANGE="${TERMINAL_RAMP_AFTER_HORIZON_CHANGE:-0}"
TERMINAL_RAMP_MIN="${TERMINAL_RAMP_MIN:-0.25}"
TERMINAL_RAMP_STEPS="${TERMINAL_RAMP_STEPS:-1000}"
TERMINAL_RAMP_TERMINAL_ONLY="${TERMINAL_RAMP_TERMINAL_ONLY:-0}"
RESET_OPTIMIZER_ON_CURRICULUM_TRANSITION="${RESET_OPTIMIZER_ON_CURRICULUM_TRANSITION:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_PHYSICS_GATE="${SKIP_PHYSICS_GATE:-0}"
PYTHON="${PYTHON:-python}"

if [[ -z "${CMAKE:-}" ]]; then
    if command -v cmake >/dev/null 2>&1; then
        CMAKE="$(command -v cmake)"
    elif [[ -x "/c/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe" ]]; then
        CMAKE="/c/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
    else
        echo "cmake was not found. Set CMAKE=/path/to/cmake and rerun." >&2
        exit 1
    fi
fi

RUN_ROOT="runs/diff_pre_training/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/logs"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
REPORT_DIR="${RUN_ROOT}/reports"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${REPORT_DIR}"

TRAIN_EXTRA_ARGS=()
if [[ "${TERMINAL_RAMP_AFTER_HORIZON_CHANGE:-0}" == "1" ]]; then
    TRAIN_EXTRA_ARGS+=(
        --terminal-ramp-after-horizon-change
        --terminal-ramp-min "${TERMINAL_RAMP_MIN:-0.25}"
        --terminal-ramp-steps "${TERMINAL_RAMP_STEPS:-1000}"
    )
fi
if [[ "${TERMINAL_RAMP_TERMINAL_ONLY:-0}" == "1" ]]; then
    TRAIN_EXTRA_ARGS+=(--terminal-ramp-terminal-only)
fi
if [[ "${RESET_OPTIMIZER_ON_CURRICULUM_TRANSITION:-0}" == "1" ]]; then
    TRAIN_EXTRA_ARGS+=(--reset-optimizer-on-curriculum-transition)
fi
if [[ "${DECOUPLED_CURRICULUM:-0}" == "1" ]]; then
    TRAIN_EXTRA_ARGS+=(--decoupled-curriculum)
    TRAIN_EXTRA_ARGS+=(--state-curriculum-lag-steps "${STATE_CURRICULUM_LAG_STEPS:-0}")
fi
if [[ "${ONE_CURRICULUM_AXIS_AT_A_TIME:-0}" == "1" ]]; then
    TRAIN_EXTRA_ARGS+=(--one-curriculum-axis-at-a-time)
fi
if [[ "${HOLD_H64_EXTRA_STEPS:-0}" -gt 0 ]]; then
    TRAIN_EXTRA_ARGS+=(--hold-h64-extra-steps "${HOLD_H64_EXTRA_STEPS}")
fi
if [[ "${STABILITY_GATED_CURRICULUM:-0}" == "1" ]]; then
    TRAIN_EXTRA_ARGS+=(--stability-gated-curriculum)
    TRAIN_EXTRA_ARGS+=(--curriculum-stability-window "${CURRICULUM_STABILITY_WINDOW:-100}")
    TRAIN_EXTRA_ARGS+=(--curriculum-min-stage-steps "${CURRICULUM_MIN_STAGE_STEPS:-1000}")
    TRAIN_EXTRA_ARGS+=(--curriculum-max-stage-steps "${CURRICULUM_MAX_STAGE_STEPS:-3000}")
    TRAIN_EXTRA_ARGS+=(--curriculum-loss-spike-ratio "${CURRICULUM_LOSS_SPIKE_RATIO:-2.0}")
fi
if [[ "${SUCCESS_GATED_CURRICULUM:-0}" == "1" ]]; then
    TRAIN_EXTRA_ARGS+=(--success-gated-curriculum)
    TRAIN_EXTRA_ARGS+=(--curriculum-success-window "${CURRICULUM_SUCCESS_WINDOW:-100}")
    TRAIN_EXTRA_ARGS+=(--curriculum-success-threshold "${CURRICULUM_SUCCESS_THRESHOLD:-0.25}")
    TRAIN_EXTRA_ARGS+=(--curriculum-min-stage-steps "${CURRICULUM_MIN_STAGE_STEPS:-1000}")
    TRAIN_EXTRA_ARGS+=(--curriculum-gate-check-interval "${CURRICULUM_GATE_CHECK_INTERVAL:-50}")
    TRAIN_EXTRA_ARGS+=(--curriculum-max-stage-steps "${CURRICULUM_MAX_STAGE_STEPS:-2000}")
    TRAIN_EXTRA_ARGS+=(--curriculum-stage-plan "${CURRICULUM_STAGE_PLAN:-alternating}")
    if [[ "${CURRICULUM_NO_FORCED_ADVANCE:-1}" == "1" ]]; then
        TRAIN_EXTRA_ARGS+=(--curriculum-no-forced-advance)
    fi
fi
if [[ "${H128_PRIORITIZED_CURRICULUM:-0}" == "1" ]]; then
    TRAIN_EXTRA_ARGS+=(--h128-prioritized-curriculum)
    TRAIN_EXTRA_ARGS+=(--h128-schedule "${H128_SCHEDULE:-short_warmup_12000}")
fi
if [[ -n "${ACTION_GRAD_CLIP:-}" ]]; then
    TRAIN_EXTRA_ARGS+=(--action-grad-clip "${ACTION_GRAD_CLIP}")
fi
if [[ "${DIAGNOSTIC_LOG_DETAIL:-0}" == "1" ]]; then
    TRAIN_EXTRA_ARGS+=(--diagnostic-log-detail)
    TRAIN_EXTRA_ARGS+=(--diagnostic-log-every "${DIAGNOSTIC_LOG_EVERY:-20}")
    TRAIN_EXTRA_ARGS+=(--diagnostic-log-first-steps "${DIAGNOSTIC_LOG_FIRST_STEPS:-200}")
fi

print_cmd(){
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

run_cmd(){
    print_cmd "$@"
    if [[ "${DRY_RUN}" != "1" ]]; then
        "$@"
    fi
}

resolve_exe(){
    local target="$1"
    local candidates=(
        "${BUILD_DIR}/src/foundation_policy/${target}"
        "${BUILD_DIR}/src/foundation_policy/${target}.exe"
        "${BUILD_DIR}/bin/${target}"
        "${BUILD_DIR}/bin/${target}.exe"
        "${BUILD_DIR}/bin/Release/${target}"
        "${BUILD_DIR}/bin/Release/${target}.exe"
        "${BUILD_DIR}/Release/${target}"
        "${BUILD_DIR}/Release/${target}.exe"
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -x "${candidate}" || -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

build_targets(){
    if [[ "${SKIP_BUILD}" == "1" ]]; then
        echo "Skipping build because SKIP_BUILD=1"
        return 0
    fi
    run_cmd "${CMAKE}" -S . -B "${BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DRL_TOOLS_ENABLE_TARGETS=ON \
        -DRL_TOOLS_DISABLE_CPU_SPECIFIC_OPTIMIZATIONS=ON
    run_cmd "${CMAKE}" --build "${BUILD_DIR}" --config Release \
        --target foundation_policy_diff_physics_check foundation_policy_diff_pre_training -j2
}

physics_gate(){
    local physics_exe="$1"
    local gate_log="${LOG_DIR}/physics_gate_euler.log"
    if [[ "${SKIP_PHYSICS_GATE}" == "1" ]]; then
        echo "Skipping physics gate because SKIP_PHYSICS_GATE=1"
        return 0
    fi
    print_cmd "${physics_exe}" --diff-model euler
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    "${physics_exe}" --diff-model euler > "${gate_log}" 2>&1
    cat "${gate_log}"
    if ! grep -q "overall=STRICT_PASS" "${gate_log}"; then
        echo "Euler physics gate did not report overall=STRICT_PASS; aborting." >&2
        exit 1
    fi
}

train_one_seed(){
    local train_seed="$1"
    local checkpoint="${CHECKPOINT_DIR}/fixed_vstrong_seed${train_seed}_policy.bin"
    local train_csv="${LOG_DIR}/fixed_vstrong_seed${train_seed}_train.csv"
    local train_log="${LOG_DIR}/fixed_vstrong_seed${train_seed}_train.log"
    if [[ "${FORCE}" != "1" && -f "${checkpoint}" ]]; then
        echo "Skipping training seed ${train_seed}; checkpoint exists: ${checkpoint}"
        return 0
    fi
    print_cmd "${TRAIN_EXE}" \
        --diff-model euler \
        --steps "${STEPS}" \
        --fixed-dynamics \
        --horizon 128 \
        --horizon-curriculum \
        --state-curriculum \
        --horizon-stage-steps "${HORIZON_STAGE_STEPS}" \
        --state-curriculum-stage-steps "${STATE_CURRICULUM_STAGE_STEPS}" \
        --actor-grad-clip "${ACTOR_GRAD_CLIP}" \
        --w-v 1.5 \
        --w-w 1.0 \
        --w-terminal-v 10 \
        --w-terminal-w 6 \
        "${TRAIN_EXTRA_ARGS[@]}" \
        --seed "${train_seed}" \
        --log-path "${train_csv}" \
        --save-path "${checkpoint}"
    if [[ "${DRY_RUN}" != "1" ]]; then
        "${TRAIN_EXE}" \
            --diff-model euler \
            --steps "${STEPS}" \
            --fixed-dynamics \
            --horizon 128 \
            --horizon-curriculum \
            --state-curriculum \
            --horizon-stage-steps "${HORIZON_STAGE_STEPS}" \
            --state-curriculum-stage-steps "${STATE_CURRICULUM_STAGE_STEPS}" \
            --actor-grad-clip "${ACTOR_GRAD_CLIP}" \
            --w-v 1.5 \
            --w-w 1.0 \
            --w-terminal-v 10 \
            --w-terminal-w 6 \
            "${TRAIN_EXTRA_ARGS[@]}" \
            --seed "${train_seed}" \
            --log-path "${train_csv}" \
            --save-path "${checkpoint}" \
            > "${train_log}" 2>&1
    fi
}

eval_one(){
    local train_seed="$1"
    local eval_seed="$2"
    local eval_model="$3"
    local checkpoint="${CHECKPOINT_DIR}/fixed_vstrong_seed${train_seed}_policy.bin"
    local eval_csv="${LOG_DIR}/fixed_vstrong_seed${train_seed}_evalseed${eval_seed}_${eval_model}.csv"
    local eval_log="${LOG_DIR}/fixed_vstrong_seed${train_seed}_evalseed${eval_seed}_${eval_model}.log"
    if [[ "${FORCE}" != "1" && -f "${eval_csv}" ]]; then
        echo "Skipping eval train_seed=${train_seed} eval_seed=${eval_seed} model=${eval_model}; CSV exists: ${eval_csv}"
        return 0
    fi
    print_cmd "${TRAIN_EXE}" \
        --eval-only \
        --load-path "${checkpoint}" \
        --eval-model "${eval_model}" \
        --fixed-dynamics \
        --seed "${eval_seed}" \
        --eval-episodes "${EVAL_EPISODES}" \
        --eval-horizon "${EVAL_HORIZON}" \
        --log-path "${eval_csv}"
    if [[ "${DRY_RUN}" != "1" ]]; then
        "${TRAIN_EXE}" \
            --eval-only \
            --load-path "${checkpoint}" \
            --eval-model "${eval_model}" \
            --fixed-dynamics \
            --seed "${eval_seed}" \
            --eval-episodes "${EVAL_EPISODES}" \
            --eval-horizon "${EVAL_HORIZON}" \
            --log-path "${eval_csv}" \
            > "${eval_log}" 2>&1
    fi
}

build_targets

PHYSICS_EXE="$(resolve_exe foundation_policy_diff_physics_check || true)"
TRAIN_EXE="$(resolve_exe foundation_policy_diff_pre_training || true)"
if [[ -z "${PHYSICS_EXE}" || -z "${TRAIN_EXE}" ]]; then
    echo "Required executables were not found under ${BUILD_DIR}." >&2
    exit 1
fi

physics_gate "${PHYSICS_EXE}"

for train_seed in ${TRAIN_SEEDS}; do
    train_one_seed "${train_seed}"
    for eval_seed in ${EVAL_SEEDS}; do
        for eval_model in ${EVAL_MODELS}; do
            eval_one "${train_seed}" "${eval_seed}" "${eval_model}"
        done
    done
done

run_cmd "${PYTHON}" "${SCRIPT_DIR}/summarize_fixed_10seed_audit.py"
