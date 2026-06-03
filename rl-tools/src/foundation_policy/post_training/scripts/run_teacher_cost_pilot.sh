#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-/tmp/diffphys_teacher_cost_build_hdf5}"
POST_EXE="${POST_EXE:-${BUILD_DIR}/src/foundation_policy/foundation_policy_post_training}"
EVAL_EXE="${EVAL_EXE:-${BUILD_DIR}/src/foundation_policy/foundation_policy_post_training_evaluate}"
ACTOR_EQ_EXE="${ACTOR_EQ_EXE:-${BUILD_DIR}/src/foundation_policy/foundation_policy_actor_equivalence_check}"

DEFAULT_DIFF_INIT="${REPO_ROOT}/runs/diff_pre_training/sampled_medium_finetune_from_fixed_12000/checkpoints/actor_checkpoint.txt"
DIFF_INIT_ACTOR_PATH="${DIFF_INIT_ACTOR_PATH:-${DEFAULT_DIFF_INIT}}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/teacher_cost_pilot/minimal_$(date +%Y%m%d_%H%M%S)}"

TEACHER_EXPERIMENT="${TEACHER_EXPERIMENT:-2025-04-16_20-10-58}"
TEACHER_SEARCH_ROOT="${TEACHER_SEARCH_ROOT:-${REPO_ROOT}/1k-experiments}"
TEACHER_INDEX_PATH="${TEACHER_INDEX_PATH:-${REPO_ROOT}/src/foundation_policy/checkpoints_${TEACHER_EXPERIMENT}.txt}"
DYNAMICS_PARAMETERS_PATH="${DYNAMICS_PARAMETERS_PATH:-${REPO_ROOT}/src/foundation_policy/dynamics_parameters_${TEACHER_EXPERIMENT}}"

TEACHER_COUNTS="${TEACHER_COUNTS:-8 16 32}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2}"
POST_TRAIN_EPOCHS="${POST_TRAIN_EPOCHS:-1000}"
ZERO_TEACHER_EPOCHS="${ZERO_TEACHER_EPOCHS:-0}"
TEACHER_FORCING_EPOCHS="${TEACHER_FORCING_EPOCHS:-10}"
POST_TRAIN_MAX_BATCHES_PER_EPOCH="${POST_TRAIN_MAX_BATCHES_PER_EPOCH:-1}"
MIN_TEACHER_RETURN="${MIN_TEACHER_RETURN:--1e20}"
DRY_RUN="${DRY_RUN:-0}"
RUN_EQUIVALENCE="${RUN_EQUIVALENCE:-1}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
EVAL_SEEDS="${EVAL_SEEDS:-10000 10001 10002}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
EVAL_HORIZON="${EVAL_HORIZON:-128}"
EVAL_DOMAINS="${EVAL_DOMAINS:-fixed narrow medium}"

require_file(){
    local path="$1"
    local label="$2"
    if [[ ! -f "${path}" ]]; then
        echo "Missing ${label}: ${path}" >&2
        exit 2
    fi
}

require_dir(){
    local path="$1"
    local label="$2"
    if [[ ! -d "${path}" ]]; then
        echo "Missing ${label}: ${path}" >&2
        exit 2
    fi
}

require_file "${POST_EXE}" "post-training executable"
require_file "${DIFF_INIT_ACTOR_PATH}" "diff-init actor text checkpoint"
if [[ "${RUN_EVALUATION}" == "1" ]]; then
    require_file "${EVAL_EXE}" "post-training evaluator executable"
fi

if [[ "${RUN_EQUIVALENCE}" == "1" ]]; then
    require_file "${ACTOR_EQ_EXE}" "actor equivalence executable"
    "${ACTOR_EQ_EXE}" --checkpoint "${DIFF_INIT_ACTOR_PATH}"
fi

mkdir -p "${RUN_ROOT}/reports"
SUMMARY_CSV="${RUN_ROOT}/reports/teacher_cost_pilot_summary.csv"
echo "teacher_count,condition,train_seed,eval_domain,eval_seed,success_rate,mean_final_position_norm,mean_final_velocity_norm,mean_final_angular_velocity_norm,invalid_or_nan_rate,mean_action_norm,skipped_updates,training_nan_inf,checkpoint" > "${SUMMARY_CSV}"

append_eval_row(){
    local csv_path="$1"
    local teacher_count="$2"
    local init="$3"
    local train_seed="$4"
    local eval_domain="$5"
    local eval_seed="$6"
    local post_log="$7"
    python3 - "${SUMMARY_CSV}" "${csv_path}" "${teacher_count}" "${init}" "${train_seed}" "${eval_domain}" "${eval_seed}" "${post_log}" <<'PYAPPEND'
import csv
import re
import sys
from pathlib import Path
summary_path, eval_csv, teacher_count, condition, train_seed, domain, eval_seed, post_log = sys.argv[1:]
with open(eval_csv, newline='') as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit(f"empty eval csv: {eval_csv}")
row = rows[0]
log_text = Path(post_log).read_text(errors='ignore').lower() if Path(post_log).exists() else ''
skipped = len(re.findall(r'skipped', log_text))
training_nan_inf = 1 if re.search(r'(^|[^a-z])(nan|inf)([^a-z]|$)', log_text) else 0
out = {
    'teacher_count': teacher_count,
    'condition': condition,
    'train_seed': train_seed,
    'eval_domain': domain,
    'eval_seed': eval_seed,
    'success_rate': row['success_rate'],
    'mean_final_position_norm': row['mean_final_position_norm'],
    'mean_final_velocity_norm': row['mean_final_velocity_norm'],
    'mean_final_angular_velocity_norm': row['mean_final_angular_velocity_norm'],
    'invalid_or_nan_rate': row['invalid_or_nan_rate'],
    'mean_action_norm': row['mean_action_norm'],
    'skipped_updates': skipped,
    'training_nan_inf': training_nan_inf,
    'checkpoint': row['checkpoint'],
}
with open(summary_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(out.keys()))
    writer.writerow(out)
PYAPPEND
}

for teacher_count in ${TEACHER_COUNTS}; do
    require_file "${TEACHER_INDEX_PATH}" "teacher index"
    require_dir "${DYNAMICS_PARAMETERS_PATH}" "dynamics parameter directory"
    require_dir "${TEACHER_SEARCH_ROOT}" "teacher checkpoint search root"
    for seed in ${TRAIN_SEEDS}; do
        for init in scratch diff_init; do
            run_epochs="${POST_TRAIN_EPOCHS}"
            if [[ "${teacher_count}" == "0" ]]; then
                run_epochs="${ZERO_TEACHER_EPOCHS}"
            fi
            run_path="${RUN_ROOT}/${init}/teachers_${teacher_count}/seed_${seed}"
            cmd=("${POST_EXE}"
                --seed "${seed}"
                --num-teachers "${teacher_count}"
                --epochs "${run_epochs}"
                --teacher-forcing-epochs "${TEACHER_FORCING_EPOCHS}"
                --max-batches-per-epoch "${POST_TRAIN_MAX_BATCHES_PER_EPOCH}"
                --min-teacher-return "${MIN_TEACHER_RETURN}"
                --teacher-experiment "${TEACHER_EXPERIMENT}"
                --teacher-search-root "${TEACHER_SEARCH_ROOT}"
                --teacher-index-path "${TEACHER_INDEX_PATH}"
                --dynamics-parameters-path "${DYNAMICS_PARAMETERS_PATH}"
                --run-path "${run_path}")
            if [[ "${init}" == "diff_init" ]]; then
                cmd+=(--init-actor-path "${DIFF_INIT_ACTOR_PATH}")
            fi
            echo "Running ${init} teachers=${teacher_count} seed=${seed} epochs=${run_epochs}"
            echo "  ${cmd[*]}"
            if [[ "${DRY_RUN}" != "1" ]]; then
                mkdir -p "${run_path}"
                "${cmd[@]}" > "${run_path}/post_training.log" 2>&1
                if [[ "${RUN_EVALUATION}" == "1" ]]; then
                    checkpoint="${run_path}/checkpoint.h5"
                    require_file "${checkpoint}" "post-training checkpoint"
                    for eval_domain in ${EVAL_DOMAINS}; do
                        for eval_seed in ${EVAL_SEEDS}; do
                            eval_dir="${run_path}/eval"
                            mkdir -p "${eval_dir}"
                            eval_csv="${eval_dir}/${eval_domain}_seed_${eval_seed}.csv"
                            eval_log="${eval_dir}/${eval_domain}_seed_${eval_seed}.log"
                            "${EVAL_EXE}" \
                                --checkpoint "${checkpoint}" \
                                --domain "${eval_domain}" \
                                --seed "${eval_seed}" \
                                --episodes "${EVAL_EPISODES}" \
                                --horizon "${EVAL_HORIZON}" \
                                --csv "${eval_csv}" > "${eval_log}" 2>&1
                            append_eval_row "${eval_csv}" "${teacher_count}" "${init}" "${seed}" "${eval_domain}" "${eval_seed}" "${run_path}/post_training.log"
                        done
                    done
                fi
            fi
        done
    done
done

python3 - "${SUMMARY_CSV}" "${RUN_ROOT}/reports/teacher_cost_pilot_report.md" <<'PYREPORT'
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path
summary_path, report_path = sys.argv[1:]
rows = []
with open(summary_path, newline='') as f:
    for row in csv.DictReader(f):
        row['teacher_count'] = int(row['teacher_count'])
        row['train_seed'] = int(row['train_seed'])
        row['eval_seed'] = int(row['eval_seed'])
        for k in ['success_rate','mean_final_position_norm','mean_final_velocity_norm','mean_final_angular_velocity_norm','invalid_or_nan_rate','mean_action_norm']:
            row[k] = float(row[k])
        row['skipped_updates'] = int(row['skipped_updates'])
        row['training_nan_inf'] = int(row['training_nan_inf'])
        rows.append(row)

def mean(values):
    return statistics.mean(values) if values else float('nan')

by_group = defaultdict(list)
for row in rows:
    by_group[(row['teacher_count'], row['condition'], row['eval_domain'])].append(row)

counts = sorted({r['teacher_count'] for r in rows})
conditions = ['scratch', 'diff_init']
domains = ['fixed', 'narrow', 'medium']

same_count_medium_improvement = []
for c in counts:
    s = mean([r['success_rate'] for r in by_group[(c, 'scratch', 'medium')]])
    d = mean([r['success_rate'] for r in by_group[(c, 'diff_init', 'medium')]])
    same_count_medium_improvement.append(d > s)

def group_success(count, condition, domain='medium'):
    return mean([r['success_rate'] for r in by_group[(count, condition, domain)]])

lower_matches = []
if 8 in counts and 16 in counts:
    lower_matches.append(('diff_init_8_ge_scratch_16', group_success(8, 'diff_init') >= group_success(16, 'scratch')))
if 8 in counts and 32 in counts:
    lower_matches.append(('diff_init_8_ge_scratch_32', group_success(8, 'diff_init') >= group_success(32, 'scratch')))
if 16 in counts and 32 in counts:
    lower_matches.append(('diff_init_16_ge_scratch_32', group_success(16, 'diff_init') >= group_success(32, 'scratch')))

unstable = any(r['invalid_or_nan_rate'] > 0 or r['skipped_updates'] > 0 or r['training_nan_inf'] for r in rows)
if rows and all(same_count_medium_improvement) and any(v for _, v in lower_matches) and not unstable:
    decision = 'MINI_TEACHER_SIGNAL_FOUND'
elif rows:
    decision = 'MINI_TEACHER_SIGNAL_NOT_CONFIRMED'
else:
    decision = 'INCOMPLETE'

def auc(condition, domain='medium'):
    pts = [(c, group_success(c, condition, domain)) for c in counts]
    pts = [(x,y) for x,y in pts if y == y]
    if len(pts) < 2:
        return float('nan')
    total = 0.0
    for (x0,y0),(x1,y1) in zip(pts, pts[1:]):
        total += (x1-x0)*(y0+y1)/2.0
    return total

lines = []
lines.append('# Mini Teacher-Cost Pilot Report')
lines.append('')
lines.append('Scope: local mini-teacher pilot only. This does not prove replacement of RAPTOR 1000 teachers, official RAPTOR matching, Sim2Real, or broad sampled-dynamics success.')
lines.append('')
lines.append(f'Decision: `{decision}`')
lines.append('')
lines.append('## Mean L2F Success by Teacher Count')
lines.append('| teacher_count | condition | fixed | narrow | medium | invalid_or_nan_max | skipped_updates_sum | training_nan_inf_sum |')
lines.append('| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |')
for c in counts:
    for cond in conditions:
        vals = {dom: group_success(c, cond, dom) for dom in domains}
        group_rows = [r for r in rows if r['teacher_count'] == c and r['condition'] == cond]
        invalid_max = max([r['invalid_or_nan_rate'] for r in group_rows], default=float('nan'))
        skipped_sum = sum(r['skipped_updates'] for r in group_rows)
        nan_sum = sum(r['training_nan_inf'] for r in group_rows)
        lines.append(f"| {c} | {cond} | {vals['fixed']:.6g} | {vals['narrow']:.6g} | {vals['medium']:.6g} | {invalid_max:.6g} | {skipped_sum} | {nan_sum} |")
lines.append('')
lines.append('## Medium L2F Deltas')
lines.append('| teacher_count | diff_init_minus_scratch |')
lines.append('| ---: | ---: |')
for c in counts:
    delta = group_success(c, 'diff_init') - group_success(c, 'scratch')
    lines.append(f'| {c} | {delta:.6g} |')
lines.append('')
lines.append('## Teacher-Cost Signal Checks')
for name, value in lower_matches:
    lines.append(f'- `{name}`: {str(value).lower()}')
lines.append(f'- `medium_auc_scratch`: {auc("scratch"):.6g}')
lines.append(f'- `medium_auc_diff_init`: {auc("diff_init"):.6g}')
lines.append('')
lines.append('## Per-Seed Rows')
lines.append('| teacher_count | condition | train_seed | domain | mean_success | mean_final_p | mean_final_v | mean_final_w | invalid |')
lines.append('| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |')
per_seed = defaultdict(list)
for row in rows:
    per_seed[(row['teacher_count'], row['condition'], row['train_seed'], row['eval_domain'])].append(row)
for key in sorted(per_seed):
    rs = per_seed[key]
    lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {mean([r['success_rate'] for r in rs]):.6g} | {mean([r['mean_final_position_norm'] for r in rs]):.6g} | {mean([r['mean_final_velocity_norm'] for r in rs]):.6g} | {mean([r['mean_final_angular_velocity_norm'] for r in rs]):.6g} | {mean([r['invalid_or_nan_rate'] for r in rs]):.6g} |")
Path(report_path).write_text('\n'.join(lines) + '\n')
print(f'teacher_cost_pilot_report={report_path}')
print(f'decision={decision}')
PYREPORT

echo "Teacher-cost pilot runs written under: ${RUN_ROOT}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Report: ${RUN_ROOT}/reports/teacher_cost_pilot_report.md"
