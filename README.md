# CA-HNM: Constraint-Aware Hard Negative Mining

This public artifact implements **CA-HNM**, an offline negative-selection method
for dense retrieval with structured domain models. It is aligned with the current
conference manuscript and its three empirical questions:

1. CA-HNM-Pure versus a retrieval-rank-matched control;
2. CA-HNM-Mixed versus matched-mixed negative pools; and
3. sensitivity to label-only, shuffled-graph, and no-structure variants.

The structured model is used only while constructing training negatives. The
retriever remains a standard dual encoder during training and inference.

## Paper terminology and implementation names

| Paper name | Command-line strategy |
| --- | --- |
| DPR-Random | `DPR-Random` |
| DenseNeg | `DenseNeg` |
| Rank-matched | `CA-HNM-rank-matched` |
| CA-HNM-Pure | `CA-HNM-full` |
| CA-HNM-Mixed | `CA-HNM-mixed` |
| Matched-mixed | `CA-HNM-matched-mixed` |
| Label-only | `CA-HNM-label-only` |
| Shuffled graph | `CA-HNM-shuffled-graph` |
| No structure | `CA-HNM-no-ontology` |

## Repository layout

```text
src/cahnm/                 Mining, retrieval, evaluation, and training code
scripts/                   Dataset, experiment, aggregation, and figure scripts
scripts/server/            Optional Linux/Slurm launchers
examples/sample_benchmark/ Small synthetic smoke-test collection
results/factorial/         Canonical RQ1 aggregate summaries
results/expanded/          Canonical RQ2/RQ3 aggregate summaries
results/training_data_counts.csv  Realized query/negative/triplet counts
tests/                     Deterministic unit and integration tests
DATASETS.md                Dataset preparation and split construction
REPRODUCIBILITY.md         Commands matching the paper protocol
```

Raw datasets, model checkpoints, per-query training outputs, and downloaded model
weights are intentionally excluded. The compact CSV files under `results/` are
included so that the manuscript tables and Figure 3 can be checked without the
large checkpoints.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dense,plots,dev]"
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.

## Quick smoke test

The smoke test uses a deterministic hash retriever and does not require a GPU:

```bash
python scripts/compare_baselines.py \
  --corpus examples/sample_benchmark/corpus.jsonl \
  --queries examples/sample_benchmark/queries.jsonl \
  --qrels examples/sample_benchmark/qrels.tsv \
  --ontology examples/sample_benchmark/ontology.json \
  --strategies DPR-Random DenseNeg CA-HNM-rank-matched CA-HNM-full \
  --dense-backend hash \
  --top-k 5 \
  --negatives-per-query 2 \
  --out runs/sample_comparison

python -m pytest -q
```

Every comparison run writes `training_data_summary.csv` and
`training_data_summary.json`. These files report, per strategy, the number of
input queries, queries that retain negatives, selected negatives, queries that
produce training triplets, and triplets.

## Reproducing the study

1. Follow [DATASETS.md](DATASETS.md) to create the two processed collections and
   target-disjoint train/development/test splits.
2. Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) to run the factorial and
   expanded experiment matrices.
3. Aggregate the completed runs and regenerate the paper tables and forest plot.

The reported paper results use the development partitions. The held-out test
qrels were not used for model or configuration selection.

## Included result summaries

- `results/factorial/`: two datasets, two encoders, two losses, four strategies,
  and ten training seeds.
- `results/expanded/`: BGE with cached MNRL for the mixed-pool and structural
  controls, also with ten seeds.
- `results/effect_forest_source.csv`: the eight paired NDCG@10 effects and
  hierarchical 95% bootstrap intervals plotted in Figure 3.
- `results/training_data_counts.csv`: realized retained-query, negative, and
  triplet counts for the factorial and expanded executions.

These files contain aggregate and seed-level summaries, not model checkpoints.

## Scope

The public artifact contains the methods and experiments reported in the current
paper. Historical one-epoch runs, LLM-assisted validation, human-annotation
utilities, biomedical/NFCorpus experiments, proxy related-work baselines, and
figures from the rejected submission have been removed from this release.

## Citation

Please use [CITATION.cff](CITATION.cff). Add the final proceedings DOI and
repository URL after they are assigned.
