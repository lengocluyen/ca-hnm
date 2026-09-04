#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-cahnm/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/paper_experiments}"
JOB_MATRIX="${JOB_MATRIX:-${RUN_ROOT}/server_jobs/core_dev.tsv}"
DATASETS="${DATASETS:-mooccubex course_skill_atlas}"
MODELS="${MODELS:-bge-base e5-base}"
LOSSES="${LOSSES:-triplet cached-mnrl}"
SEEDS="${SEEDS:-11 22 33 44 55 66 77 88 99 111}"
PROFILE="${PROFILE:-core}"
EVAL_SPLIT="${EVAL_SPLIT:-dev}"

read -r -a DATASET_ARGS <<< "${DATASETS}"
read -r -a MODEL_ARGS <<< "${MODELS}"
read -r -a LOSS_ARGS <<< "${LOSSES}"
read -r -a SEED_ARGS <<< "${SEEDS}"

# Slurm opens output files before the job script body runs.
mkdir -p runs/slurm_logs "$(dirname "${JOB_MATRIX}")"

"${PYTHON_BIN}" scripts/make_experiment_job_matrix.py \
  --datasets "${DATASET_ARGS[@]}" \
  --models "${MODEL_ARGS[@]}" \
  --losses "${LOSS_ARGS[@]}" \
  --seeds "${SEED_ARGS[@]}" \
  --profile "${PROFILE}" \
  --eval-split "${EVAL_SPLIT}" \
  --out "${JOB_MATRIX}"

JOB_COUNT=$(($(wc -l < "${JOB_MATRIX}") - 1))
if (( JOB_COUNT <= 0 )); then
  echo "Empty job matrix" >&2
  exit 2
fi

export PYTHON_BIN RUN_ROOT JOB_MATRIX DATASETS MODELS PROFILE EVAL_SPLIT
MINE_JOB=$(sbatch --parsable scripts/server/slurm_mine.sh)
TRAIN_JOB=$(sbatch --parsable --dependency="afterok:${MINE_JOB}" --array="0-$((JOB_COUNT - 1))" scripts/server/slurm_train_array.sh)
ANALYSIS_JOB=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" scripts/server/slurm_aggregate.sh)

echo "Mining job:  ${MINE_JOB}"
echo "Training job: ${TRAIN_JOB} (${JOB_COUNT} array tasks)"
echo "Analysis job: ${ANALYSIS_JOB}"
