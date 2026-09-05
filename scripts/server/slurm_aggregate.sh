#!/usr/bin/env bash
#SBATCH --job-name=cahnm-analysis
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm_logs/%x-%j.out
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-cahnm/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/paper_experiments}"
mkdir -p runs/slurm_logs

"${PYTHON_BIN}" scripts/aggregate_experiment_runs.py \
  --runs-root "${RUN_ROOT}/training" \
  --baseline CA-HNM-rank-matched \
  --out-dir "${RUN_ROOT}/analysis"
"${PYTHON_BIN}" scripts/experiment_status.py --root "${RUN_ROOT}" --out "${RUN_ROOT}/status.json"
