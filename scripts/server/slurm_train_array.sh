#!/usr/bin/env bash
#SBATCH --job-name=cahnm-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm_logs/%x-%A_%a.out
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-cahnm/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/paper_experiments}"
JOB_MATRIX="${JOB_MATRIX:?Set JOB_MATRIX to the TSV emitted by make_experiment_job_matrix.py}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array}"
mkdir -p runs/slurm_logs

LINE_NUMBER=$((TASK_ID + 2))
ROW=$(awk -F '\t' -v line="${LINE_NUMBER}" 'NR==line {print $0}' "${JOB_MATRIX}")
if [[ -z "${ROW}" ]]; then
  echo "No job row for array index ${TASK_ID}" >&2
  exit 2
fi
IFS=$'\t' read -r JOB_INDEX DATASET MODEL LOSS SEED PROFILE EVAL_SPLIT MODEL_FITS <<< "${ROW}"

"${PYTHON_BIN}" scripts/run_experiment_matrix.py \
  --stage train \
  --datasets "${DATASET}" \
  --models "${MODEL}" \
  --losses "${LOSS}" \
  --seeds "${SEED}" \
  --profile "${PROFILE}" \
  --eval-split "${EVAL_SPLIT}" \
  --device cuda \
  --epochs "${EPOCHS:-3}" \
  --train-batch-size "${TRAIN_BATCH_SIZE:-32}" \
  --cached-mini-batch-size "${CACHED_MINI_BATCH_SIZE:-4}" \
  --out-root "${RUN_ROOT}" \
  --limit-runs 1 \
  --resume
