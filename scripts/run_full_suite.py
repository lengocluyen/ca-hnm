#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


DATASETS = {
    "mooccubex": {
        "corpus": "data/processed/mooccubex_full/corpus.jsonl",
        "queries": "data/processed/mooccubex_full/queries.jsonl",
        "qrels": "data/processed/mooccubex_full/qrels.tsv",
        "ontology": "data/processed/mooccubex_ontology_full.json",
    },
    "course_skill_atlas": {
        "corpus": "data/processed/course_skill_atlas/corpus.jsonl",
        "queries": "data/processed/course_skill_atlas/queries.jsonl",
        "qrels": "data/processed/course_skill_atlas/qrels.tsv",
        "ontology": "data/processed/course_skill_atlas_ontology.json",
    },
}

ALL_STRATEGIES = [
    "DPR-Random",
    "ANCE",
    "ADORE",
    "RocketQA-Denoised",
    "TAS-Balanced",
    "GPL-Pseudo",
    "SyNeg",
    "CA-HNM-v2",
    "CA-HNM-v2-mixed",
    "CA-HNM-full",
    "CA-HNM-mixed",
    "CA-HNM-no-ontology",
    "CA-HNM-no-reasoning",
    "CA-HNM-no-fn-filter",
    "CA-HNM-prereq-only",
    "CA-HNM-sibling-only",
]

VALIDATION_NAME_ALIASES = {
    "CA-HNM-v2": "CA-HNM-RA",
    "CA-HNM-v2-mixed": "CA-HNM-RA-Mix",
    "CA-HNM-full": "CA-HNM-C",
    "CA-HNM-mixed": "CA-HNM-Mix",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full CA-HNM experiments across datasets and judges.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["mooccubex", "course_skill_atlas"])
    parser.add_argument("--judges", nargs="+", choices=["heuristic", "lmstudio"], default=["heuristic", "lmstudio"])
    parser.add_argument("--out-root", default="runs/full_suite")
    parser.add_argument("--gpu", default="2", help="Physical GPU id exposed to Python with CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--dense-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--train-model", default=None, help="Defaults to --dense-model.")
    parser.add_argument("--dense-batch-size", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--negatives-per-query", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--loss", choices=["triplet", "mnrl", "multiple-negatives", "hybrid"], default="triplet")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--candidate-fusion", choices=["max_score", "rrf"], default="max_score")
    parser.add_argument("--selection-policy", choices=["balanced", "retrieval_aware", "top_ranked"], default="balanced")
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--retrieval-weight", type=float, default=0.55)
    parser.add_argument("--constraint-weight", type=float, default=0.30)
    parser.add_argument("--ontology-weight", type=float, default=0.15)
    parser.add_argument("--diversity-penalty", type=float, default=0.06)
    parser.add_argument("--eval-top-k", type=int, default=100)
    parser.add_argument("--llm-model", default="gpt-oss-20b")
    parser.add_argument("--llm-base-url", default="http://localhost:1234/v1")
    parser.add_argument("--llm-candidate-chars", type=int, default=1200)
    parser.add_argument("--llm-context-items", type=int, default=10)
    parser.add_argument("--llm-label-chars", type=int, default=80)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--llm-retry-sleep", type=float, default=10.0)
    parser.add_argument("--llm-validation-sample-size", type=int, default=1000)
    parser.add_argument(
        "--llm-validation-strategies",
        nargs="+",
        default=["CA-HNM-mixed", "CA-HNM-v2-mixed"],
        help="Negative files to validate after mining/training. Defaults to CA-HNM-mixed and CA-HNM-v2-mixed.",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-trained-eval", action="store_true")
    parser.add_argument("--skip-llm-validation", action="store_true")
    parser.add_argument("--allow-validation-failure", action="store_true", help="Continue to analysis if post-hoc LLM validation fails.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed strategy artifacts when rerunning after interruption.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_model = args.train_model or args.dense_model
    repo = Path.cwd()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "suite_commands.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    for dataset_name in args.datasets:
        dataset = DATASETS[dataset_name]
        _check_dataset_files(dataset_name, dataset)
        for judge in args.judges:
            run_name = f"{dataset_name}_{judge}_{_model_slug(args.dense_model)}"
            run_dir = out_root / run_name
            llm_cache = run_dir / f"llm_cache.{args.llm_model}.jsonl"
            compare_cmd = [
                sys.executable,
                "scripts/compare_baselines.py",
                "--corpus",
                dataset["corpus"],
                "--queries",
                dataset["queries"],
                "--qrels",
                dataset["qrels"],
                "--ontology",
                dataset["ontology"],
                "--strategies",
                *ALL_STRATEGIES,
                "--dense-backend",
                "sentence-transformers",
                "--dense-model",
                args.dense_model,
                "--dense-device",
                "cuda",
                "--dense-batch-size",
                str(args.dense_batch_size),
                "--dense-max-seq-length",
                str(args.max_seq_length),
                "--judge",
                judge,
                "--top-k",
                str(args.top_k),
                "--negatives-per-query",
                str(args.negatives_per_query),
                "--random-seed",
                str(args.random_seed),
                "--candidate-fusion",
                args.candidate_fusion,
                "--selection-policy",
                args.selection_policy,
                "--confidence",
                str(args.confidence),
                "--retrieval-weight",
                str(args.retrieval_weight),
                "--constraint-weight",
                str(args.constraint_weight),
                "--ontology-weight",
                str(args.ontology_weight),
                "--diversity-penalty",
                str(args.diversity_penalty),
                "--out",
                str(run_dir),
            ]
            if judge == "lmstudio":
                compare_cmd.extend(
                    [
                        "--llm-base-url",
                        args.llm_base_url,
                        "--llm-model",
                        args.llm_model,
                        "--llm-cache",
                        str(llm_cache),
                        "--llm-candidate-chars",
                        str(args.llm_candidate_chars),
                        "--llm-context-items",
                        str(args.llm_context_items),
                        "--llm-label-chars",
                        str(args.llm_label_chars),
                        "--llm-retries",
                        str(args.llm_retries),
                        "--llm-retry-sleep",
                        str(args.llm_retry_sleep),
                    ]
                )
            if not args.skip_train:
                compare_cmd.extend(
                    [
                        "--train",
                        "--train-model",
                        train_model,
                        "--train-device",
                        "cuda",
                        "--train-max-seq-length",
                        str(args.max_seq_length),
                        "--train-use-amp",
                        "--epochs",
                        str(args.epochs),
                        "--batch-size",
                        str(args.train_batch_size),
                        "--loss",
                        args.loss,
                        "--warmup-ratio",
                        str(args.warmup_ratio),
                        "--no-eval-trained-models",
                    ]
                )
                if args.learning_rate is not None:
                    compare_cmd.extend(["--learning-rate", str(args.learning_rate)])
            if args.resume:
                compare_cmd.append("--resume")
            _run(compare_cmd, env, repo, log_path, args.dry_run)

            if not args.skip_train and not args.skip_trained_eval:
                eval_cmd = [
                    sys.executable,
                    "scripts/evaluate_trained_models.py",
                    "--corpus",
                    dataset["corpus"],
                    "--queries",
                    dataset["queries"],
                    "--qrels",
                    dataset["qrels"],
                    "--ontology",
                    dataset["ontology"],
                    "--models-dir",
                    str(run_dir / "models"),
                    "--dense-device",
                    "cuda",
                    "--dense-batch-size",
                    str(args.dense_batch_size),
                    "--dense-max-seq-length",
                    str(args.max_seq_length),
                    "--eval-top-k",
                    str(args.eval_top_k),
                    "--out",
                    str(run_dir / "trained_retrieval_metrics.json"),
                ]
                if judge == "lmstudio":
                    eval_cmd.extend(
                        [
                            "--judge",
                            "lmstudio",
                            "--llm-base-url",
                            args.llm_base_url,
                            "--llm-model",
                            args.llm_model,
                            "--llm-cache",
                            str(llm_cache),
                            "--llm-candidate-chars",
                            str(args.llm_candidate_chars),
                            "--llm-context-items",
                            str(args.llm_context_items),
                            "--llm-label-chars",
                            str(args.llm_label_chars),
                            "--llm-retries",
                            str(args.llm_retries),
                            "--llm-retry-sleep",
                            str(args.llm_retry_sleep),
                        ]
                    )
                _run(eval_cmd, env, repo, log_path, args.dry_run)

            if not args.skip_llm_validation:
                for strategy in args.llm_validation_strategies:
                    negatives_path = run_dir / f"{strategy}.negatives.jsonl"
                    output_stem = VALIDATION_NAME_ALIASES.get(strategy, strategy)
                    validation_cmd = [
                        sys.executable,
                        "scripts/validate_negatives.py",
                        "--corpus",
                        dataset["corpus"],
                        "--queries",
                        dataset["queries"],
                        "--qrels",
                        dataset["qrels"],
                        "--ontology",
                        dataset["ontology"],
                        "--negatives",
                        str(negatives_path),
                        "--sample-size",
                        str(args.llm_validation_sample_size),
                        "--judge",
                        "lmstudio",
                        "--llm-base-url",
                        args.llm_base_url,
                        "--llm-model",
                        args.llm_model,
                        "--llm-cache",
                        str(llm_cache),
                        "--llm-candidate-chars",
                        str(args.llm_candidate_chars),
                        "--llm-context-items",
                        str(args.llm_context_items),
                        "--llm-label-chars",
                        str(args.llm_label_chars),
                        "--llm-retries",
                        str(args.llm_retries),
                        "--llm-retry-sleep",
                        str(args.llm_retry_sleep),
                        "--out",
                        str(run_dir / f"{output_stem}.{args.llm_model}_validation_summary.json"),
                        "--decisions-out",
                        str(run_dir / f"{output_stem}.{args.llm_model}_validation_decisions.jsonl"),
                    ]
                    _run(
                        validation_cmd,
                        env,
                        repo,
                        log_path,
                        args.dry_run,
                        allow_failure=args.allow_validation_failure,
                    )

            analyze_cmd = [
                sys.executable,
                "scripts/analyze_results.py",
                "--run",
                str(run_dir),
            ]
            _run(analyze_cmd, env, repo, log_path, args.dry_run)


def _check_dataset_files(dataset_name: str, dataset: dict[str, str]) -> None:
    missing = [path for path in dataset.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files for {dataset_name}: {missing}")


def _run(
    cmd: list[str],
    env: dict[str, str],
    cwd: Path,
    log_path: Path,
    dry_run: bool,
    allow_failure: bool = False,
) -> None:
    rendered = " ".join(shlex.quote(part) for part in cmd)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
    print(f"\n$ {rendered}", flush=True)
    if dry_run:
        return
    completed = subprocess.run(cmd, cwd=cwd, env=env, check=not allow_failure)
    if allow_failure and completed.returncode:
        print(f"warning: command failed with exit code {completed.returncode}; continuing", flush=True)


def _model_slug(model: str) -> str:
    return (
        model.replace("/", "_")
        .replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )


if __name__ == "__main__":
    main()
