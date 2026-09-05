# Reproducibility guide

This guide matches the protocol reported in the current CA-HNM paper.

## Canonical configuration

| Item | Value |
| --- | --- |
| Collections | MOOCCubeX, Course-Skill Atlas |
| Encoders | `BAAI/bge-base-en-v1.5`, `intfloat/e5-base-v2` |
| Losses | triplet, cached MNRL |
| Training seeds | 11, 22, 33, 44, 55, 66, 77, 88, 99, 111 |
| Mining seed | 13 |
| Candidate depth | 100 per lexical/dense retriever |
| Explicit negatives | At most 4 per query |
| Epochs | 3 |
| Batch size | 32 |
| Cached mini-batch | 4 |
| Optimizer | AdamW (`betas=(0.9, 0.999)`, `eps=1e-8`, weight decay `0.01`) |
| Learning rate | 2e-5 |
| Warmup | 10% |
| Maximum sequence length | 128 |
| Evaluation partition | development |

Prepare both datasets first by following [DATASETS.md](DATASETS.md).

## 1. Factorial RQ1 experiment

This matrix evaluates DPR-Random, DenseNeg, rank-matched, and CA-HNM-Pure for
two datasets, two encoders, two losses, and ten seeds:

```bash
python scripts/run_experiment_matrix.py \
  --stage all \
  --datasets mooccubex course_skill_atlas \
  --models bge-base e5-base \
  --losses triplet cached-mnrl \
  --seeds 11 22 33 44 55 66 77 88 99 111 \
  --profile core \
  --eval-split dev \
  --device cuda \
  --dense-batch-size 8 \
  --train-batch-size 32 \
  --cached-mini-batch-size 4 \
  --max-seq-length 128 \
  --epochs 3 \
  --learning-rate 2e-5 \
  --warmup-ratio 0.1 \
  --top-k 100 \
  --negatives-per-query 4 \
  --mining-seed 13 \
  --out-root runs/paper_experiments \
  --resume
```

Negative pools are mined once per dataset/encoder and reused unchanged across
training seeds.

Each execution writes `training_data_summary.csv` and
`training_data_summary.json`. The summaries report realized retained-query,
negative, and triplet counts instead of assuming that every input query reaches
the four-negative cap. Training logs explicitly record `optimizer: AdamW` and
its fixed defaults. Counts from the reported runs are included in
[`results/training_data_counts.csv`](results/training_data_counts.csv).

## 2. Expanded RQ2/RQ3 experiment

The mixed-pool and structural analyses use BGE with cached MNRL:

```bash
python scripts/run_experiment_matrix.py \
  --stage all \
  --datasets mooccubex course_skill_atlas \
  --models bge-base \
  --losses cached-mnrl \
  --seeds 11 22 33 44 55 66 77 88 99 111 \
  --profile ablation \
  --eval-split dev \
  --device cuda \
  --dense-batch-size 8 \
  --train-batch-size 32 \
  --cached-mini-batch-size 4 \
  --max-seq-length 128 \
  --epochs 3 \
  --learning-rate 2e-5 \
  --warmup-ratio 0.1 \
  --top-k 100 \
  --negatives-per-query 4 \
  --mining-seed 13 \
  --out-root runs/paper_experiments_expanded \
  --resume
```

## 3. Aggregate paired results

```bash
python scripts/aggregate_experiment_runs.py \
  --runs-root runs/paper_experiments/training \
  --baseline CA-HNM-rank-matched \
  --bootstrap-samples 5000 \
  --seed 20260825 \
  --out-dir runs/paper_experiments/analysis

python scripts/aggregate_experiment_runs.py \
  --runs-root runs/paper_experiments_expanded/training \
  --baseline CA-HNM-rank-matched \
  --bootstrap-samples 5000 \
  --seed 20260825 \
  --out-dir runs/paper_experiments_expanded/analysis
```

The hierarchical bootstrap resamples training seeds and, within sampled seeds,
development queries. The paper interprets the paired mean differences and 95%
bootstrap intervals.

## 4. Canonical source rule for repeated cells

The expanded execution is canonical for every BGE/cached-MNRL cell because it
contains the mixed and structural controls. The factorial execution supplies the
other six encoder-loss configurations. `make_results_tables.py` and
`make_results_figures.py` implement this rule explicitly, preventing the same
method/configuration from appearing with values from different executions.

## 5. Regenerate paper outputs from included summaries

The repository includes compact copies of the aggregate CSVs under `results/`.
Regenerate the LaTeX tables and Figure 3 without model checkpoints:

```bash
python scripts/make_results_tables.py
python scripts/make_results_figures.py
```

Outputs are written to:

```text
artifacts/tables/core_results.tex
artifacts/tables/hybrid_results.tex
artifacts/tables/ablation_results.tex
artifacts/figures/effect_forest.pdf
artifacts/figures/effect_forest.png
```

The figure script also regenerates `results/effect_forest_source.csv` and checks
the paper's numerical statement: six of eight mean differences are positive and
four 95% intervals are entirely above zero.

## 6. Linux/Slurm helpers

For a direct Linux run:

```bash
export RUN_ROOT=runs/paper_experiments
bash scripts/server/run_pipeline.sh all-dev
```

For Slurm, adjust the `#SBATCH` resource lines and run:

```bash
bash scripts/server/submit_slurm.sh
```

The launchers default to the exact datasets, encoders, losses, and seeds listed
above. Check `runs/.../matrix_manifest.json`, `commands.jsonl`, and each
`experiment_run.json` before interpreting results.

## 7. Hardware note

Install a PyTorch build compatible with the GPU's compute capability. Tesla V100
(`sm_70`) systems require a wheel that still contains Volta kernels and a cuDNN
version compatible with `sm_70`; newer wheels may omit this support.
