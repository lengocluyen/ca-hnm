# CA-HNM: Constraint-Aware Hard Negative Mining

This repository contains the implementation for **Constraint-Aware Hard Negative Mining (CA-HNM)**, a framework for mining hard negatives for dense retrieval using structured domain constraints.

CA-HNM selects negatives that are:

1. close to the query under lexical or dense retrieval,
2. not labeled as relevant,
3. invalid under ontology-derived constraints.

The ontology is used only during offline negative mining. It is not used as retriever input and is not required at inference time.

## Repository Layout

```text
src/cahnm/                 Core package
scripts/                   CLI wrappers and experiment utilities
examples/sample_benchmark/ Small runnable benchmark
configs/                   Example configuration
docs/                      Additional implementation notes
tests/                     Unit tests
DATASETS.md                Dataset download and preprocessing guide
REPRODUCIBILITY.md         Paper experiment commands
```

Generated files are intentionally excluded from the public artifact:

- `data/`
- `runs/`
- trained model checkpoints
- LLM cache files
- downloaded third-party datasets

## Installation

Create a Python environment with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the package:

```bash
python -m pip install -e .
```

For dense retrieval and training with sentence-transformers:

```bash
python -m pip install -e ".[dense]"
```

For LLM-assisted validation through an OpenAI-compatible API:

```bash
python -m pip install -e ".[llm]"
```

For tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Quick Smoke Test

The repository includes a tiny sample benchmark.

```bash
python scripts/compare_baselines.py \
  --corpus examples/sample_benchmark/corpus.jsonl \
  --queries examples/sample_benchmark/queries.jsonl \
  --qrels examples/sample_benchmark/qrels.tsv \
  --ontology examples/sample_benchmark/ontology.json \
  --strategies DPR-Random ANCE CA-HNM-full CA-HNM-mixed \
  --dense-backend hash \
  --out runs/sample_comparison
```

Expected outputs:

```text
runs/sample_comparison/*.negatives.jsonl
runs/sample_comparison/*.triplets.jsonl
runs/sample_comparison/retrieval_metrics.json
runs/sample_comparison/negative_quality.csv
```

## Reproducing the Paper Experiments

See:

- [DATASETS.md](DATASETS.md) for obtaining and preprocessing MoocCubeX and Course-Skill Atlas.
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the main training, validation, statistics, and figure commands.

The main paper configuration uses:

- retriever: `BAAI/bge-base-en-v1.5`
- top-k candidate pool: `100`
- negatives per query: `4`
- loss: Multiple Negatives Ranking Loss (`mnrl`)
- sequence length: `128`
- training epochs: `1`
- mining judge: deterministic heuristic classifier

## Data Format

`corpus.jsonl`

```json
{"_id": "d1", "title": "Introduction to SQL Joins", "text": "Beginner SQL joins...", "metadata": {"level": "beginner"}}
```

`queries.jsonl`

```json
{"_id": "q1", "text": "beginner SQL joins", "target_concept": "sql_joins", "metadata": {}}
```

`qrels.tsv`

```text
query_id	doc_id	relevance
q1	d1	1
```

`ontology.json`

```json
{
  "nodes": [
    {"id": "sql_joins", "label": "SQL joins", "aliases": ["INNER JOIN"], "level": "beginner"}
  ],
  "relations": [
    {"source": "sql_joins", "target": "sql_querying", "type": "is_a"}
  ]
}
```

## Citation

If you use this code, please cite the accompanying paper. A citation template is provided in [CITATION.cff](CITATION.cff).

