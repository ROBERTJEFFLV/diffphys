#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RL_TOOLS_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${RL_TOOLS_DIR}"

BUILD_DIR="${BUILD_DIR:-build_diff}"
STEPS="${STEPS:-6000}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2}"
EVAL_SEEDS="${EVAL_SEEDS:-10000 10001 10002}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
EVAL_HORIZON="${EVAL_HORIZON:-128}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
CMAKE="${CMAKE:-cmake}"
PYTHON="${PYTHON:-python}"

RUN_ROOT="runs/diff_pre_training/fixed_vstrong_multiseed"
LOG_DIR="${RUN_ROOT}/logs"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
REPORT_DIR="${RUN_ROOT}/reports"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${REPORT_DIR}"

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

configure_if_needed(){
    if [[ ! -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
        run_cmd "${CMAKE}" -S . -B "${BUILD_DIR}" \
            -DCMAKE_BUILD_TYPE=Release \
            -DRL_TOOLS_ENABLE_TARGETS=ON \
            -DRL_TOOLS_DISABLE_CPU_SPECIFIC_OPTIMIZATIONS=ON
    fi
}

build_if_missing(){
    local physics_exe=""
    local train_exe=""
    physics_exe="$(resolve_exe foundation_policy_diff_physics_check || true)"
    train_exe="$(resolve_exe foundation_policy_diff_pre_training || true)"
    if [[ -z "${physics_exe}" || -z "${train_exe}" ]]; then
        run_cmd "${CMAKE}" --build "${BUILD_DIR}" --config Release \
            --target foundation_policy_diff_physics_check foundation_policy_diff_pre_training -j
    fi
}

physics_gate(){
    local physics_exe="$1"
    local gate_log="${LOG_DIR}/physics_gate_euler.log"
    print_cmd "${physics_exe}" --diff-model euler
    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi
    "${physics_exe}" --diff-model euler 2>&1 | tee "${gate_log}"
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
        --actor-grad-clip 50 \
        --w-v 1.5 \
        --w-w 1.0 \
        --w-terminal-v 10 \
        --w-terminal-w 6 \
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
            --actor-grad-clip 50 \
            --w-v 1.5 \
            --w-w 1.0 \
            --w-terminal-v 10 \
            --w-terminal-w 6 \
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

configure_if_needed
build_if_missing

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
        eval_one "${train_seed}" "${eval_seed}" euler
        eval_one "${train_seed}" "${eval_seed}" l2f
    done
done

run_cmd "${PYTHON}" "${SCRIPT_DIR}/summarize_fixed_multiseed.py"
