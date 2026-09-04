#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path


DATASETS = {
    "mooccubex": {
        "corpus": "data/paper/mooccubex/corpus.jsonl",
        "ontology": "data/processed/mooccubex_ontology_full.json",
        "splits": "data/splits/paper/mooccubex",
    },
    "course_skill_atlas": {
        "corpus": "data/paper/course_skill_atlas/corpus.jsonl",
        "ontology": "data/processed/course_skill_atlas_ontology.json",
        "splits": "data/splits/paper/course_skill_atlas",
    },
}

MODELS = {
    "bge-base": "BAAI/bge-base-en-v1.5",
    "e5-base": "intfloat/e5-base-v2",
}

PROFILES = {
    "core": ["DPR-Random", "DenseNeg", "CA-HNM-rank-matched", "CA-HNM-full"],
    "ablation": [
        "DPR-Random",
        "DenseNeg",
        "CA-HNM-rank-matched",
        "CA-HNM-full",
        "CA-HNM-matched-mixed",
        "CA-HNM-mixed",
        "CA-HNM-label-only",
        "CA-HNM-shuffled-graph",
        "CA-HNM-no-ontology",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine immutable negative pools and run the disjoint multi-seed CA-HNM experiment matrix."
    )
    parser.add_argument("--stage", choices=["mine", "train", "all"], default="all")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=["mooccubex", "course_skill_atlas"],
    )
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=["bge-base", "e5-base"])
    parser.add_argument(
        "--losses",
        nargs="+",
        choices=["triplet", "cached-mnrl"],
        default=["triplet", "cached-mnrl"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55, 66, 77, 88, 99, 111])
    parser.add_argument("--profile", choices=sorted(PROFILES), default="core")
    parser.add_argument("--eval-split", choices=["dev", "test"], default="dev")
    parser.add_argument("--out-root", default="runs/experiments")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--dense-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--cached-mini-batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--mnrl-scale", type=float, default=20.0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--mining-seed", type=int, default=13)
    parser.add_argument("--triplet-seed", type=int, default=17)
    parser.add_argument("--timeout-hours", type=float, default=24.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-runs", type=int, help="Run at most N training configurations after mining.")
    args = parser.parse_args()

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    strategies = PROFILES[args.profile]
    matrix = [
        (dataset, model_key, loss, seed)
        for dataset in args.datasets
        for model_key in args.models
        for loss in args.losses
        for seed in args.seeds
    ]
    if args.limit_runs is not None:
        matrix = matrix[: args.limit_runs]

    manifest = {
        "schema_version": 1,
        "created_at": time.time(),
        "stage": args.stage,
        "datasets": args.datasets,
        "models": {key: MODELS[key] for key in args.models},
        "losses": args.losses,
        "seeds": args.seeds,
        "profile": args.profile,
        "strategies": strategies,
        "eval_split": args.eval_split,
        "planned_training_configurations": len(matrix),
        "planned_model_fits": len(matrix) * len(strategies),
        "device": device,
        "python": sys.version,
        "platform": platform.platform(),
        "dry_run": args.dry_run,
    }
    (root / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"planned {manifest['planned_training_configurations']} configurations / "
        f"{manifest['planned_model_fits']} model fits on {device}",
        flush=True,
    )

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    command_log = root / "commands.jsonl"

    if args.stage in {"mine", "all"}:
        for dataset_name in args.datasets:
            dataset = _validated_dataset(dataset_name)
            for model_key in args.models:
                mining_dir = root / "mining" / dataset_name / model_key
                command = _compare_base_command(
                    dataset,
                    model=MODELS[model_key],
                    device=device,
                    dense_batch_size=args.dense_batch_size,
                    max_seq_length=args.max_seq_length,
                    top_k=args.top_k,
                    negatives_per_query=args.negatives_per_query,
                    mining_seed=args.mining_seed,
                    strategies=strategies,
                    out_dir=mining_dir,
                    eval_split="dev",
                )
                if args.resume and _mining_complete(mining_dir, strategies):
                    print(f"resume: immutable mining artifacts already complete at {mining_dir}", flush=True)
                else:
                    _run(command, env, command_log, args.dry_run, args.timeout_hours)

    if args.stage in {"train", "all"}:
        for dataset_name, model_key, loss, seed in matrix:
            dataset = _validated_dataset(dataset_name)
            mining_dir = root / "mining" / dataset_name / model_key
            if not args.dry_run and not _mining_complete(mining_dir, strategies):
                raise FileNotFoundError(
                    f"immutable mining artifacts are incomplete at {mining_dir}; run --stage mine first"
                )
            run_dir = (
                root
                / "training"
                / args.eval_split
                / dataset_name
                / model_key
                / loss
                / f"seed_{seed}"
            )
            if args.resume and (run_dir / "trained_per_query_metrics.csv").exists():
                print(f"resume: completed training run at {run_dir}", flush=True)
                continue
            command = _compare_base_command(
                dataset,
                model=MODELS[model_key],
                device=device,
                dense_batch_size=args.dense_batch_size,
                max_seq_length=args.max_seq_length,
                top_k=args.top_k,
                negatives_per_query=args.negatives_per_query,
                mining_seed=args.mining_seed,
                strategies=strategies,
                out_dir=run_dir,
                eval_split=args.eval_split,
            )
            command.extend(
                [
                    "--artifacts-from",
                    str(mining_dir),
                    "--train",
                    "--train-model",
                    MODELS[model_key],
                    "--train-device",
                    device,
                    "--train-max-seq-length",
                    str(args.max_seq_length),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.train_batch_size),
                    "--triplet-seed",
                    str(args.triplet_seed),
                    "--train-seed",
                    str(seed),
                    "--loss",
                    loss,
                    "--cached-mini-batch-size",
                    str(args.cached_mini_batch_size),
                    "--mnrl-scale",
                    str(args.mnrl_scale),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--warmup-ratio",
                    str(args.warmup_ratio),
                ]
            )
            if device.startswith("cuda"):
                command.append("--train-use-amp")

            run_metadata = {
                "schema_version": 1,
                "dataset": dataset_name,
                "model_key": model_key,
                "model": MODELS[model_key],
                "loss": loss,
                "training_seed": seed,
                "mining_seed": args.mining_seed,
                "triplet_seed": args.triplet_seed,
                "eval_split": args.eval_split,
                "strategies": strategies,
                "epochs": args.epochs,
                "batch_size": args.train_batch_size,
                "cached_mini_batch_size": args.cached_mini_batch_size,
                "device": device,
                "status": "planned" if args.dry_run else "running",
                "command": command,
                "started_at": time.time(),
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = run_dir / "experiment_run.json"
            metadata_path.write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")
            try:
                _run(command, env, command_log, args.dry_run, args.timeout_hours)
            except Exception:
                run_metadata["status"] = "failed"
                run_metadata["finished_at"] = time.time()
                metadata_path.write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")
                raise
            run_metadata["status"] = "planned" if args.dry_run else "complete"
            run_metadata["finished_at"] = time.time()
            metadata_path.write_text(json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8")


def _compare_base_command(
    dataset: dict[str, Path],
    model: str,
    device: str,
    dense_batch_size: int,
    max_seq_length: int,
    top_k: int,
    negatives_per_query: int,
    mining_seed: int,
    strategies: list[str],
    out_dir: Path,
    eval_split: str,
) -> list[str]:
    return [
        sys.executable,
        "scripts/compare_baselines.py",
        "--corpus",
        str(dataset["corpus"]),
        "--queries",
        str(dataset["train_queries"]),
        "--qrels",
        str(dataset["train_qrels"]),
        "--eval-queries",
        str(dataset[f"{eval_split}_queries"]),
        "--eval-qrels",
        str(dataset[f"{eval_split}_qrels"]),
        "--ontology",
        str(dataset["ontology"]),
        "--strategies",
        *strategies,
        "--dense-backend",
        "sentence-transformers",
        "--dense-model",
        model,
        "--dense-device",
        device,
        "--dense-batch-size",
        str(dense_batch_size),
        "--dense-max-seq-length",
        str(max_seq_length),
        "--top-k",
        str(top_k),
        "--negatives-per-query",
        str(negatives_per_query),
        "--random-seed",
        str(mining_seed),
        "--eval-top-k",
        "100",
        "--out",
        str(out_dir),
    ]


def _validated_dataset(name: str) -> dict[str, Path]:
    raw = DATASETS[name]
    split_dir = Path(raw["splits"])
    dataset = {
        "corpus": Path(raw["corpus"]),
        "ontology": Path(raw["ontology"]),
        "train_queries": split_dir / "train.queries.jsonl",
        "train_qrels": split_dir / "train.qrels.tsv",
        "dev_queries": split_dir / "dev.queries.jsonl",
        "dev_qrels": split_dir / "dev.qrels.tsv",
        "test_queries": split_dir / "test.queries.jsonl",
        "test_qrels": split_dir / "test.qrels.tsv",
        "split_manifest": split_dir / "split_manifest.json",
    }
    missing = [str(path) for path in dataset.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing experiment dataset inputs for {name}: {missing}")
    manifest = json.loads(dataset["split_manifest"].read_text(encoding="utf-8"))
    if any(manifest.get("overlap_audit", {}).values()):
        raise ValueError(f"split manifest for {name} reports query leakage")
    return dataset


def _mining_complete(path: Path, strategies: list[str]) -> bool:
    return all(
        (path / f"{strategy}.negatives.jsonl").exists()
        and (path / f"{strategy}.triplets.jsonl").exists()
        for strategy in strategies
    )


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _run(
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    dry_run: bool,
    timeout_hours: float,
) -> None:
    rendered = " ".join(shlex.quote(part) for part in command)
    print(f"\n$ {rendered}", flush=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"time": time.time(), "command": command, "dry_run": dry_run}) + "\n")
    if dry_run:
        return
    subprocess.run(
        command,
        cwd=Path.cwd(),
        env=env,
        check=True,
        timeout=max(1.0, timeout_hours) * 3600,
    )


if __name__ == "__main__":
    main()
