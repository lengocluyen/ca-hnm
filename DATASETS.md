# Dataset Acquisition and Preprocessing

This repository does not include third-party datasets. Downloaded datasets, processed benchmarks, trained models, and run outputs should remain outside version control.

Recommended local layout:

```text
data/raw/                  Downloaded source datasets
data/processed/            Processed retrieval benchmarks and ontologies
runs/                      Experiment outputs
```

## 1. MoocCubeX

MoocCubeX files are downloaded from the public AMiner-hosted release used by the project downloader. The downloader writes a manifest and skips optional files that are unavailable from the upstream host unless `--strict-downloads` is used.

Download the core files used by CA-HNM:

```bash
python scripts/download_datasets.py mooccubex \
  --out data/raw/mooccubex_full \
  --no-metadata-only \
  --mooccubex-preset core
```

Build the ontology:

```bash
python scripts/build_ontology.py mooccubex \
  --input data/raw/mooccubex_full \
  --out data/processed/mooccubex_ontology_full.json
```

Prepare the retrieval benchmark:

```bash
python scripts/prepare_mooccubex.py \
  --input data/raw/mooccubex_full \
  --ontology data/processed/mooccubex_ontology_full.json \
  --out data/processed/mooccubex_full \
  --max-queries 5000
```

Expected processed files:

```text
data/processed/mooccubex_full/corpus.jsonl
data/processed/mooccubex_full/queries.jsonl
data/processed/mooccubex_full/qrels.tsv
data/processed/mooccubex_ontology_full.json
```

For a larger server-side download, use:

```bash
python scripts/download_datasets.py mooccubex \
  --out data/raw/mooccubex_full \
  --no-metadata-only \
  --mooccubex-preset all
```

The full preset is much larger and is not required for the main CA-HNM retrieval benchmark.

## 2. Course-Skill Atlas

Course-Skill Atlas is downloaded through the Figshare article metadata endpoint used by the project downloader.

Download the dataset:

```bash
python scripts/download_datasets.py course-skill-atlas \
  --out data/raw/course_skill_atlas \
  --no-metadata-only \
  --timeout 900 \
  --retries 10
```

Build the ontology:

```bash
python scripts/build_ontology.py course-skill-atlas \
  --input data/raw/course_skill_atlas \
  --out data/processed/course_skill_atlas_ontology.json
```

Prepare the retrieval benchmark:

```bash
python scripts/prepare_course_skill_atlas.py \
  --input data/raw/course_skill_atlas \
  --ontology data/processed/course_skill_atlas_ontology.json \
  --out data/processed/course_skill_atlas \
  --max-queries 1000 \
  --max-positives-per-query 100
```

Expected processed files:

```text
data/processed/course_skill_atlas/corpus.jsonl
data/processed/course_skill_atlas/queries.jsonl
data/processed/course_skill_atlas/qrels.tsv
data/processed/course_skill_atlas_ontology.json
```

## 3. Verify Processed Files

After preprocessing, check that the expected benchmark files exist:

```bash
python - <<'PY'
from pathlib import Path

paths = [
    "data/processed/mooccubex_full/corpus.jsonl",
    "data/processed/mooccubex_full/queries.jsonl",
    "data/processed/mooccubex_full/qrels.tsv",
    "data/processed/mooccubex_ontology_full.json",
    "data/processed/course_skill_atlas/corpus.jsonl",
    "data/processed/course_skill_atlas/queries.jsonl",
    "data/processed/course_skill_atlas/qrels.tsv",
    "data/processed/course_skill_atlas_ontology.json",
]

for path in paths:
    p = Path(path)
    print(f"{path}: {'OK' if p.exists() else 'MISSING'}")
PY
```

## 4. Notes on Dataset Licenses

Before redistributing any dataset files, check the original dataset licenses and terms. This repository is intended to redistribute code and small synthetic examples only, not third-party dataset contents.

