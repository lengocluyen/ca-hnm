#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-all-dev}"
PYTHON_BIN="${PYTHON_BIN:-.venv-cahnm/bin/python}"
RUN_ROOT="${RUN_ROOT:-runs/paper_experiments}"
DATASETS="${DATASETS:-mooccubex course_skill_atlas}"
MODELS="${MODELS:-bge-base e5-base}"
LOSSES="${LOSSES:-triplet cached-mnrl}"
SEEDS="${SEEDS:-11 22 33 44 55 66 77 88 99 111}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
CACHED_MINI_BATCH_SIZE="${CACHED_MINI_BATCH_SIZE:-4}"

read -r -a DATASET_ARGS <<< "${DATASETS}"
read -r -a MODEL_ARGS <<< "${MODELS}"
read -r -a LOSS_ARGS <<< "${LOSSES}"
read -r -a SEED_ARGS <<< "${SEEDS}"

mkdir -p "${RUN_ROOT}/server_logs"

run_smoke() {
  "${PYTHON_BIN}" scripts/check_server_environment.py --require-cuda --out "${RUN_ROOT}/environment.json"
  "${PYTHON_BIN}" -m compileall -q src scripts tests
  "${PYTHON_BIN}" -m pytest -q
}

run_mining() {
  "${PYTHON_BIN}" scripts/run_experiment_matrix.py \
    --stage mine \
    --datasets "${DATASET_ARGS[@]}" \
    --models "${MODEL_ARGS[@]}" \
    --profile core \
    --device "${DEVICE}" \
    --out-root "${RUN_ROOT}" \
    --resume
}

run_dev_core() {
  "${PYTHON_BIN}" scripts/run_experiment_matrix.py \
    --stage train \
    --datasets "${DATASET_ARGS[@]}" \
    --models "${MODEL_ARGS[@]}" \
    --losses "${LOSS_ARGS[@]}" \
    --seeds "${SEED_ARGS[@]}" \
    --profile core \
    --eval-split dev \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --cached-mini-batch-size "${CACHED_MINI_BATCH_SIZE}" \
    --out-root "${RUN_ROOT}" \
    --resume
}

run_dev_ablation() {
  "${PYTHON_BIN}" scripts/run_experiment_matrix.py \
    --stage all \
    --datasets "${DATASET_ARGS[@]}" \
    --models "${MODEL_ARGS[0]}" \
    --losses cached-mnrl \
    --seeds "${SEED_ARGS[@]}" \
    --profile ablation \
    --eval-split dev \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --cached-mini-batch-size "${CACHED_MINI_BATCH_SIZE}" \
    --out-root "${RUN_ROOT}_expanded" \
    --resume
}

aggregate_dev() {
  "${PYTHON_BIN}" scripts/aggregate_experiment_runs.py \
    --runs-root "${RUN_ROOT}/training" \
    --baseline CA-HNM-rank-matched \
    --out-dir "${RUN_ROOT}/analysis"
  "${PYTHON_BIN}" scripts/aggregate_experiment_runs.py \
    --runs-root "${RUN_ROOT}_expanded/training" \
    --baseline CA-HNM-rank-matched \
    --out-dir "${RUN_ROOT}_expanded/analysis"
}

run_locked_test() {
  if [[ "${LOCKED_TEST_ACK:-}" != "YES" ]]; then
    echo "Set LOCKED_TEST_ACK=YES only after the dev configuration is frozen." >&2
    exit 3
  fi
  "${PYTHON_BIN}" scripts/run_experiment_matrix.py \
    --stage all \
    --datasets "${DATASET_ARGS[@]}" \
    --models "${MODEL_ARGS[@]}" \
    --losses "${LOSS_ARGS[@]}" \
    --seeds "${SEED_ARGS[@]}" \
    --profile core \
    --eval-split test \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --cached-mini-batch-size "${CACHED_MINI_BATCH_SIZE}" \
    --out-root "${RUN_ROOT}_locked_test" \
    --resume
  "${PYTHON_BIN}" scripts/aggregate_experiment_runs.py \
    --runs-root "${RUN_ROOT}_locked_test/training" \
    --baseline CA-HNM-rank-matched \
    --out-dir "${RUN_ROOT}_locked_test/analysis"
}

case "${PHASE}" in
  smoke) run_smoke ;;
  mine) run_mining ;;
  dev-core) run_dev_core ;;
  dev-ablation) run_dev_ablation ;;
  aggregate-dev) aggregate_dev ;;
  locked-test) run_locked_test ;;
  all-dev)
    run_smoke
    run_mining
    run_dev_core
    run_dev_ablation
    aggregate_dev
    ;;
  *)
    echo "Unknown phase: ${PHASE}" >&2
    echo "Use: smoke | mine | dev-core | dev-ablation | aggregate-dev | locked-test | all-dev" >&2
    exit 2
    ;;
esac
