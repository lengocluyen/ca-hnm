# Dataset acquisition and preprocessing

The paper evaluates MOOCCubeX and Course-Skill Atlas. Third-party dataset files
are not redistributed in this repository. Check the original licenses before
downloading or sharing them.

The commands below create the paths expected by `scripts/run_experiment_matrix.py`.

## 1. MOOCCubeX

Download the core files used by the retrieval benchmark:

```bash
python scripts/download_datasets.py mooccubex \
  --out data/raw/mooccubex_full \
  --no-metadata-only \
  --mooccubex-preset core
```

Build the concept/prerequisite structure:

```bash
python scripts/build_ontology.py mooccubex \
  --input data/raw/mooccubex_full \
  --out data/processed/mooccubex_ontology_full.json
```

Prepare concepts as queries, courses as documents, and source concept-course
links as binary qrels:

```bash
python scripts/prepare_mooccubex.py \
  --input data/raw/mooccubex_full \
  --ontology data/processed/mooccubex_ontology_full.json \
  --out data/paper/mooccubex \
  --max-queries 5000
```

Create the target-disjoint 70/10/20 split used in the paper:

```bash
python scripts/split_benchmark.py \
  --queries data/paper/mooccubex/queries.jsonl \
  --qrels data/paper/mooccubex/qrels.tsv \
  --out-dir data/splits/paper/mooccubex \
  --seed 20260825
```

The paper instance contains 3,781 documents and 3,500/500/1,000
train/development/test queries.

## 2. Course-Skill Atlas

Download the source files through the Figshare metadata endpoint used by the
project downloader:

```bash
python scripts/download_datasets.py course-skill-atlas \
  --out data/raw/course_skill_atlas \
  --no-metadata-only \
  --timeout 900 \
  --retries 10
```

Build the hierarchy and related-concept structure:

```bash
python scripts/build_ontology.py course-skill-atlas \
  --input data/raw/course_skill_atlas \
  --out data/processed/course_skill_atlas_ontology.json
```

Prepare institution-field-year profiles as documents and Detailed Work
Activities (DWAs) as queries:

```bash
python scripts/prepare_course_skill_atlas.py \
  --input data/raw/course_skill_atlas \
  --ontology data/processed/course_skill_atlas_ontology.json \
  --out data/paper/course_skill_atlas \
  --max-queries 1000 \
  --max-positives-per-query 100
```

A DWA query is linked to fields whose source top-10 DWA list contains that
activity. Retained qrels are binary. When more than 100 profiles are associated
with a query, the preparation code retains the first 100 after its deterministic
source ordering.

Create the paper split:

```bash
python scripts/split_benchmark.py \
  --queries data/paper/course_skill_atlas/queries.jsonl \
  --qrels data/paper/course_skill_atlas/qrels.tsv \
  --out-dir data/splits/paper/course_skill_atlas \
  --seed 20260825
```

The paper instance contains 281,153 documents and 227/32/65
train/development/test queries.

## 3. Split verification

Each split directory contains:

```text
train.queries.jsonl   train.qrels.tsv
dev.queries.jsonl     dev.qrels.tsv
test.queries.jsonl    test.qrels.tsv
split_manifest.json
```

`split_manifest.json` records input/output hashes, counts, the split seed, and an
overlap audit. `run_experiment_matrix.py` refuses a split whose overlap audit is
non-zero.

The paper reports development performance only. Do not inspect the test metrics
until a configuration has been frozen.
