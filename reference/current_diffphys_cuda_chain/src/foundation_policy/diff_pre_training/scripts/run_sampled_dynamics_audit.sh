#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_TOOLS_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${RL_TOOLS_DIR}"

BUILD_DIR="${BUILD_DIR:-build_diff}"
RUN_NAME="${RUN_NAME:-sampled_dynamics_audit}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2 3 4}"
STEPS="${STEPS:-8000}"
EVAL_SEEDS="${EVAL_SEEDS:-10000 10001 10002}"
EVAL_MODELS="${EVAL_MODELS:-euler l2f}"
EVAL_DYNAMICS_MODES="${EVAL_DYNAMICS_MODES:-sampled fixed}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
EVAL_HORIZON="${EVAL_HORIZON:-128}"
SAMPLED_DYNAMICS_LEVEL="${SAMPLED_DYNAMICS_LEVEL:-narrow}"
INIT_ACTOR_PATH="${INIT_ACTOR_PATH:-}"
ZERO_SHOT="${ZERO_SHOT:-0}"
H128_PRIORITIZED_CURRICULUM="${H128_PRIORITIZED_CURRICULUM:-1}"
H128_SCHEDULE="${H128_SCHEDULE:-balanced_16000}"
HORIZON_STAGE_STEPS="${HORIZON_STAGE_STEPS:-1000}"
STATE_CURRICULUM_STAGE_STEPS="${STATE_CURRICULUM_STAGE_STEPS:-1000}"
ACTOR_GRAD_CLIP="${ACTOR_GRAD_CLIP:-50}"
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
if [[ "${H128_PRIORITIZED_CURRICULUM}" == "1" ]]; then
    TRAIN_EXTRA_ARGS+=(--h128-prioritized-curriculum --h128-schedule "${H128_SCHEDULE}")
fi
if [[ "${DYNAMICS_CURRICULUM:-0}" == "1" ]]; then
    TRAIN_EXTRA_ARGS+=(--dynamics-curriculum)
fi
if [[ "${SAMPLED_DYNAMICS_CURRICULUM_LEVELS:-0}" == "1" ]]; then
    TRAIN_EXTRA_ARGS+=(--sampled-dynamics-curriculum-levels)
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

resolve_init_checkpoint(){
    local train_seed="$1"
    if [[ -z "${INIT_ACTOR_PATH}" ]]; then
        printf '\n'
        return 0
    fi
    if [[ -d "${INIT_ACTOR_PATH}" ]]; then
        printf '%s\n' "${INIT_ACTOR_PATH}/fixed_vstrong_seed${train_seed}_policy.bin"
        return 0
    fi
    local resolved="${INIT_ACTOR_PATH/\{seed\}/${train_seed}}"
    printf '%s\n' "${resolved}"
}

trained_checkpoint(){
    local train_seed="$1"
    printf '%s\n' "${CHECKPOINT_DIR}/sampled_${SAMPLED_DYNAMICS_LEVEL}_seed${train_seed}_policy.bin"
}

write_metadata(){
    local metadata="${REPORT_DIR}/run_metadata.csv"
    {
        echo "key,value"
        echo "run_name,${RUN_NAME}"
        echo "train_seeds,\"${TRAIN_SEEDS}\""
        echo "steps,${STEPS}"
        echo "eval_seeds,\"${EVAL_SEEDS}\""
        echo "eval_models,\"${EVAL_MODELS}\""
        echo "eval_dynamics_modes,\"${EVAL_DYNAMICS_MODES}\""
        echo "eval_episodes,${EVAL_EPISODES}"
        echo "eval_horizon,${EVAL_HORIZON}"
        echo "sampled_dynamics_level,${SAMPLED_DYNAMICS_LEVEL}"
        echo "init_actor_path,${INIT_ACTOR_PATH:-none}"
        echo "zero_shot,${ZERO_SHOT}"
        echo "h128_prioritized_curriculum,${H128_PRIORITIZED_CURRICULUM}"
        echo "h128_schedule,${H128_SCHEDULE}"
    } > "${metadata}"
}

train_one_seed(){
    local train_seed="$1"
    local checkpoint
    checkpoint="$(trained_checkpoint "${train_seed}")"
    local train_csv="${LOG_DIR}/seed${train_seed}_train.csv"
    local train_log="${LOG_DIR}/seed${train_seed}_train.log"
    local init_checkpoint
    init_checkpoint="$(resolve_init_checkpoint "${train_seed}")"
    if [[ "${ZERO_SHOT}" == "1" ]]; then
        return 0
    fi
    if [[ "${FORCE}" != "1" && -f "${checkpoint}" ]]; then
        echo "Skipping training seed ${train_seed}; checkpoint exists: ${checkpoint}"
        return 0
    fi
    INIT_ARGS=()
    if [[ -n "${init_checkpoint}" ]]; then
        if [[ ! -f "${init_checkpoint}" ]]; then
            echo "Missing INIT_ACTOR_PATH for seed ${train_seed}: ${init_checkpoint}" >&2
            exit 1
        fi
        INIT_ARGS+=(--init-actor-path "${init_checkpoint}")
    fi
    print_cmd "${TRAIN_EXE}" \
        --diff-model euler \
        --steps "${STEPS}" \
        --sample-dynamics \
        --sampled-dynamics-level "${SAMPLED_DYNAMICS_LEVEL}" \
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
        "${INIT_ARGS[@]}" \
        --seed "${train_seed}" \
        --log-path "${train_csv}" \
        --save-path "${checkpoint}"
    if [[ "${DRY_RUN}" != "1" ]]; then
        "${TRAIN_EXE}" \
            --diff-model euler \
            --steps "${STEPS}" \
            --sample-dynamics \
            --sampled-dynamics-level "${SAMPLED_DYNAMICS_LEVEL}" \
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
            "${INIT_ARGS[@]}" \
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
    local eval_dynamics_mode="$4"
    local checkpoint
    if [[ "${ZERO_SHOT}" == "1" ]]; then
        checkpoint="$(resolve_init_checkpoint "${train_seed}")"
    else
        checkpoint="$(trained_checkpoint "${train_seed}")"
    fi
    if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
        echo "Missing checkpoint for eval seed ${train_seed}: ${checkpoint}" >&2
        exit 1
    fi
    local eval_csv="${LOG_DIR}/seed${train_seed}_evalseed${eval_seed}_${eval_dynamics_mode}_${eval_model}.csv"
    local eval_log="${LOG_DIR}/seed${train_seed}_evalseed${eval_seed}_${eval_dynamics_mode}_${eval_model}.log"
    if [[ "${FORCE}" != "1" && -f "${eval_csv}" ]]; then
        echo "Skipping eval train_seed=${train_seed} eval_seed=${eval_seed} mode=${eval_dynamics_mode} model=${eval_model}; CSV exists: ${eval_csv}"
        return 0
    fi
    DYNAMICS_ARGS=()
    if [[ "${eval_dynamics_mode}" == "sampled" ]]; then
        DYNAMICS_ARGS+=(--sample-dynamics --sampled-dynamics-level "${SAMPLED_DYNAMICS_LEVEL}")
    else
        DYNAMICS_ARGS+=(--fixed-dynamics)
    fi
    print_cmd "${TRAIN_EXE}" \
        --eval-only \
        --load-path "${checkpoint}" \
        --eval-model "${eval_model}" \
        "${DYNAMICS_ARGS[@]}" \
        --seed "${eval_seed}" \
        --eval-episodes "${EVAL_EPISODES}" \
        --eval-horizon "${EVAL_HORIZON}" \
        --log-path "${eval_csv}"
    if [[ "${DRY_RUN}" != "1" ]]; then
        "${TRAIN_EXE}" \
            --eval-only \
            --load-path "${checkpoint}" \
            --eval-model "${eval_model}" \
            "${DYNAMICS_ARGS[@]}" \
            --seed "${eval_seed}" \
            --eval-episodes "${EVAL_EPISODES}" \
            --eval-horizon "${EVAL_HORIZON}" \
            --log-path "${eval_csv}" \
            > "${eval_log}" 2>&1
    fi
}

write_metadata
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
            for eval_dynamics_mode in ${EVAL_DYNAMICS_MODES}; do
                eval_one "${train_seed}" "${eval_seed}" "${eval_model}" "${eval_dynamics_mode}"
            done
        done
    done
done

run_cmd "${PYTHON}" "${SCRIPT_DIR}/summarize_sampled_dynamics_audit.py" --run-root "${RUN_ROOT}"
