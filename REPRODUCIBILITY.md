# Reproducibility Guide

This file gives the main commands used for the CA-HNM paper experiments. The commands assume the datasets have already been prepared as described in [DATASETS.md](DATASETS.md).

## Main Paper Runs

### MoocCubeX

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_full_suite.py \
  --datasets mooccubex \
  --judges heuristic \
  --gpu 0 \
  --dense-model BAAI/bge-base-en-v1.5 \
  --train-model BAAI/bge-base-en-v1.5 \
  --dense-batch-size 4 \
  --train-batch-size 4 \
  --max-seq-length 128 \
  --top-k 100 \
  --negatives-per-query 4 \
  --epochs 1 \
  --loss mnrl \
  --candidate-fusion rrf \
  --selection-policy retrieval_aware \
  --eval-top-k 100 \
  --llm-model gpt-oss-20b \
  --llm-base-url http://localhost:1234/v1 \
  --llm-validation-sample-size 1000 \
  --allow-validation-failure \
  --out-root runs/e1_mooccubex_v2_heuristic_bge_base_top100_neg4_mnrl
```

### Course-Skill Atlas

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/run_full_suite.py \
  --datasets course_skill_atlas \
  --judges heuristic \
  --gpu 0 \
  --dense-model BAAI/bge-base-en-v1.5 \
  --train-model BAAI/bge-base-en-v1.5 \
  --dense-batch-size 4 \
  --train-batch-size 4 \
  --max-seq-length 128 \
  --top-k 100 \
  --negatives-per-query 4 \
  --epochs 1 \
  --loss mnrl \
  --candidate-fusion rrf \
  --selection-policy retrieval_aware \
  --eval-top-k 100 \
  --llm-model gpt-oss-20b \
  --llm-base-url http://localhost:1234/v1 \
  --llm-validation-sample-size 1000 \
  --allow-validation-failure \
  --out-root runs/e1_course_skill_atlas_v2_heuristic_bge_base_top100_neg4_mnrl
```

## Key Output Files

For each dataset run, the main output directory contains:

```text
retrieval_metrics.json
trained_retrieval_metrics.json
negative_quality.csv
training_summary.csv
training_summary.json
*.negatives.jsonl
*.triplets.jsonl
*.gpt-oss-20b_validation_summary.json
*.gpt-oss-20b_validation_decisions.jsonl
models/
analysis/
```

## Zero-Shot Baselines

```bash
python scripts/evaluate_zero_shot_baselines.py \
  --corpus data/processed/mooccubex_full/corpus.jsonl \
  --queries data/processed/mooccubex_full/queries.jsonl \
  --qrels data/processed/mooccubex_full/qrels.tsv \
  --ontology data/processed/mooccubex_ontology_full.json \
  --dense-model BAAI/bge-base-en-v1.5 \
  --dense-device cuda \
  --out runs/old_main_stats/mooccubex_zero_shot_baselines.json
```

```bash
python scripts/evaluate_zero_shot_baselines.py \
  --corpus data/processed/course_skill_atlas/corpus.jsonl \
  --queries data/processed/course_skill_atlas/queries.jsonl \
  --qrels data/processed/course_skill_atlas/qrels.tsv \
  --ontology data/processed/course_skill_atlas_ontology.json \
  --dense-model BAAI/bge-base-en-v1.5 \
  --dense-device cuda \
  --out runs/old_main_stats/course_skill_atlas_zero_shot_baselines.json
```

## Statistical Tests

Run paired tests on trained model outputs:

```bash
python scripts/statistical_tests.py \
  --run runs/e1_mooccubex_v2_heuristic_bge_base_top100_neg4_mnrl/mooccubex_heuristic_baai_bge_base_en_v1_5 \
  --baseline DPR-Random \
  --methods CA-HNM-mixed CA-HNM-v2-mixed \
  --out runs/old_main_stats/mooccubex
```

```bash
python scripts/statistical_tests.py \
  --run runs/e1_course_skill_atlas_v2_heuristic_bge_base_top100_neg4_mnrl/course_skill_atlas_heuristic_baai_bge_base_en_v1_5 \
  --baseline DPR-Random \
  --methods CA-HNM-mixed CA-HNM-v2-mixed \
  --out runs/old_main_stats/course_skill_atlas
```

## LLM-Assisted Validation From Annotation Sheets

If using an OpenAI-compatible local or remote API:

```bash
python scripts/llm_validate_sheet.py \
  --input runs/human_validation_old/mooccubex_validation_sheet.csv \
  --output runs/human_validation_old/gpt_family/mooccubex_judgments.jsonl \
  --llm-base-url http://localhost:1234/v1 \
  --llm-model gpt-oss-20b
```

```bash
python scripts/llm_validate_sheet.py \
  --input runs/human_validation_old/course_skill_atlas_validation_sheet.csv \
  --output runs/human_validation_old/gpt_family/course_skill_atlas_judgments.jsonl \
  --llm-base-url http://localhost:1234/v1 \
  --llm-model gpt-oss-20b
```

For commercial LLMs that are evaluated manually or through a separate UI, keep the same JSON schema:

```json
{"item_id": "...", "label": "HardNeg", "violation_types": ["target_concept_mismatch"], "evidence": "...", "confidence": 0.82}
```

Allowed labels:

```text
HardNeg, EasyNeg, Positive, Ambiguous
```

## Figure Generation

```bash
python scripts/make_paper_figures.py \
  --mooccubex-run runs/e1_mooccubex_v2_heuristic_bge_base_top100_neg4_mnrl/mooccubex_heuristic_baai_bge_base_en_v1_5 \
  --csa-run runs/e1_course_skill_atlas_v2_heuristic_bge_base_top100_neg4_mnrl/course_skill_atlas_heuristic_baai_bge_base_en_v1_5 \
  --out runs/paper_figures_base_grouped
```

```bash
python scripts/make_review_figures.py \
  --stats-root runs/old_main_stats \
  --out runs/paper_figures_review_oldcfg
```

## Notes

- Use `--resume` when restarting interrupted experiments.
- Use `--allow-validation-failure` when LLM validation is optional.
- The deterministic heuristic judge is the scalable mining method used in the main experiments.
- LLM-assisted validation is a complementary diagnostic and should not be interpreted as ground-truth annotation.

