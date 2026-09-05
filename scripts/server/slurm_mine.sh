#!/usr/bin/env bash
#SBATCH --job-name=cahnm-mine
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=runs/slurm_logs/%x-%j.out
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-cahnm/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/paper_experiments}"
read -r -a DATASET_ARGS <<< "${DATASETS:-mooccubex course_skill_atlas}"
read -r -a MODEL_ARGS <<< "${MODELS:-bge-base e5-base}"
mkdir -p runs/slurm_logs

"${PYTHON_BIN}" scripts/run_experiment_matrix.py \
  --stage mine \
  --datasets "${DATASET_ARGS[@]}" \
  --models "${MODEL_ARGS[@]}" \
  --profile "${PROFILE:-core}" \
  --device cuda \
  --out-root "${RUN_ROOT}" \
  --resume
