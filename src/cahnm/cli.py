from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from .baselines import DEFAULT_STRATEGIES, BaselineConfig, NegativeStrategyRunner, summarize_negative_quality
from .data.download import (
    build_course_skill_atlas_ontology,
    build_mooccubex_ontology,
    download_course_skill_atlas,
    download_mooccubex,
)
from .data.prepare import prepare_course_skill_atlas_benchmark, prepare_mooccubex_benchmark
from .evaluation import constraint_violation_at_k, evaluate_run, evaluate_run_per_query
from .io_utils import load_corpus, load_negatives, load_qrels, load_queries, read_jsonl, write_negatives
from .judges import HeuristicConstraintJudge
from .ontology import Ontology
from .retrievers import BM25Retriever, DenseRetriever
from .schemas import RetrievalRun
from .training import build_triplets, train_sentence_transformer_triplets, write_triplets


def cmd_download(args: argparse.Namespace) -> None:
    if args.insecure_ssl:
        os.environ["CAHNM_INSECURE_SSL"] = "1"
    if args.dataset == "mooccubex":
        path = download_mooccubex(
            args.out,
            metadata_only=args.metadata_only,
            file_pattern=args.file_pattern,
            max_files=args.max_files,
            preset=args.mooccubex_preset,
            force=args.force,
            resume=not args.no_resume,
            strict_downloads=args.strict_downloads,
            timeout=args.timeout,
            retries=args.retries,
        )
    elif args.dataset == "course-skill-atlas":
        path = download_course_skill_atlas(
            args.out,
            metadata_only=args.metadata_only,
            file_pattern=args.file_pattern,
            max_files=args.max_files,
            force=args.force,
            resume=not args.no_resume,
            timeout=args.timeout,
            retries=args.retries,
        )
    else:  # pragma: no cover - argparse enforces the choices
        raise ValueError(args.dataset)
    print(path)


def cmd_build_ontology(args: argparse.Namespace) -> None:
    if args.source == "mooccubex":
        path = build_mooccubex_ontology(
            args.input,
            args.out,
            max_concepts=args.max_concepts,
            max_prerequisites=args.max_prerequisites,
        )
    elif args.source == "course-skill-atlas":
        path = build_course_skill_atlas_ontology(
            args.input,
            args.out,
            include_abilities=not args.no_abilities,
            include_tasks=not args.no_tasks,
        )
    else:  # pragma: no cover - argparse enforces the choices
        raise ValueError(args.source)
    print(path)


def cmd_prepare_mooccubex(args: argparse.Namespace) -> None:
    print(
        prepare_mooccubex_benchmark(
            input_dir=args.input,
            output_dir=args.out,
            ontology_path=args.ontology,
            max_queries=args.max_queries,
            max_courses=args.max_courses,
        )
    )


def cmd_prepare_course_skill_atlas(args: argparse.Namespace) -> None:
    max_positives = None if args.max_positives_per_query == 0 else args.max_positives_per_query
    print(
        prepare_course_skill_atlas_benchmark(
            input_dir=args.input,
            output_dir=args.out,
            ontology_path=args.ontology,
            max_queries=args.max_queries,
            max_docs=args.max_docs,
            min_syllabi=args.min_syllabi,
            max_positives_per_query=max_positives,
        )
    )


def _metric_ks(eval_top_k: int) -> tuple[int, ...]:
    values = [k for k in (10, 50, 100) if k <= eval_top_k]
    if eval_top_k not in values:
        values.append(eval_top_k)
    return tuple(sorted(set(values)))


def cmd_compare(args: argparse.Namespace) -> None:
    docs = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    ontology = Ontology.load(args.ontology)
    eval_queries = load_queries(args.eval_queries) if args.eval_queries else queries
    eval_qrels = load_qrels(args.eval_qrels) if args.eval_qrels else qrels
    if bool(args.eval_queries) != bool(args.eval_qrels):
        raise ValueError("--eval-queries and --eval-qrels must be provided together")
    _validate_qrels_match_queries(queries, qrels, split_name="train")
    _validate_qrels_match_queries(eval_queries, eval_qrels, split_name="eval")
    overlap = sorted({query.id for query in queries} & {query.id for query in eval_queries})
    if args.eval_queries and overlap:
        raise ValueError(
            f"train/eval query leakage: {len(overlap)} overlapping query IDs; first={overlap[0]!r}"
        )

    judge = HeuristicConstraintJudge()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_audit = {
        "train_queries": len(queries),
        "train_qrels": len(qrels),
        "eval_queries": len(eval_queries),
        "eval_qrels": len(eval_qrels),
        "query_id_overlap": len(overlap),
        "disjoint_eval_requested": bool(args.eval_queries),
        "train_queries_path": str(args.queries),
        "train_qrels_path": str(args.qrels),
        "eval_queries_path": str(args.eval_queries or args.queries),
        "eval_qrels_path": str(args.eval_qrels or args.qrels),
    }
    (out_dir / "data_audit.json").write_text(
        json.dumps(data_audit, indent=2, sort_keys=True), encoding="utf-8"
    )

    runner = NegativeStrategyRunner(
        docs,
        queries,
        qrels,
        ontology,
        judge=judge,
        config=BaselineConfig(
            max_negatives_per_query=args.negatives_per_query,
            top_k=args.top_k,
            dense_model=args.dense_model,
            dense_backend=args.dense_backend,
            dense_batch_size=args.dense_batch_size,
            dense_device=args.dense_device,
            dense_max_seq_length=args.dense_max_seq_length,
            random_seed=args.random_seed,
            confidence_threshold=args.confidence,
            candidate_fusion="max_score",
        ),
    )
    strategies = args.strategies or DEFAULT_STRATEGIES
    negatives_by_strategy = {}
    trained_retrieval_rows = []
    trained_per_query: dict[str, dict[str, dict[str, float]]] = {}
    training_summaries = []
    artifacts_from = Path(args.artifacts_from) if args.artifacts_from else None

    for strategy in strategies:
        negatives_path = out_dir / f"{strategy}.negatives.jsonl"
        triplets_path = out_dir / f"{strategy}.triplets.jsonl"
        model_dir = out_dir / "models" / strategy
        shared_negatives_path = artifacts_from / f"{strategy}.negatives.jsonl" if artifacts_from else None
        shared_triplets_path = artifacts_from / f"{strategy}.triplets.jsonl" if artifacts_from else None

        if shared_negatives_path is not None:
            if not shared_negatives_path.exists():
                raise FileNotFoundError(f"missing shared negatives for {strategy}: {shared_negatives_path}")
            print(f"loading immutable negatives for {strategy} from {shared_negatives_path}")
            negatives = load_negatives(shared_negatives_path)
        elif args.resume and negatives_path.exists():
            print(f"resume: loading existing negatives for {strategy} from {negatives_path}")
            negatives = load_negatives(negatives_path)
        else:
            negatives = runner.run(strategy)
            write_negatives(negatives_path, negatives)
        negatives_by_strategy[strategy] = negatives

        if shared_triplets_path is not None:
            if not shared_triplets_path.exists():
                raise FileNotFoundError(f"missing shared triplets for {strategy}: {shared_triplets_path}")
            print(f"loading immutable triplets for {strategy} from {shared_triplets_path}")
            triplets = list(read_jsonl(shared_triplets_path))
        elif args.resume and triplets_path.exists():
            print(f"resume: loading existing triplets for {strategy} from {triplets_path}")
            triplets = list(read_jsonl(triplets_path))
        else:
            triplets = build_triplets(queries, docs, qrels, negatives, random_seed=args.triplet_seed)
            write_triplets(triplets_path, triplets)

        if args.train and triplets:
            if args.resume and _looks_like_sentence_transformer_model(model_dir):
                print(f"resume: found trained model for {strategy} at {model_dir}; skipping training")
            else:
                training_summaries.append(
                    train_sentence_transformer_triplets(
                        triplets,
                        model_dir,
                        model_name=args.train_model,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        strategy=strategy,
                        logs_dir=out_dir / "training_logs" / strategy,
                        device=args.train_device,
                        max_seq_length=args.train_max_seq_length,
                        use_amp=args.train_use_amp,
                        loss_name=args.loss,
                        learning_rate=args.learning_rate,
                        warmup_ratio=args.warmup_ratio,
                        random_seed=args.train_seed,
                        deterministic=args.train_deterministic,
                        cached_mini_batch_size=args.cached_mini_batch_size,
                        mnrl_scale=args.mnrl_scale,
                    )
                )
            if not args.no_eval_trained_models:
                trained_run = DenseRetriever(
                    docs,
                    model_name=str(model_dir),
                    backend="sentence-transformers",
                    batch_size=args.dense_batch_size,
                    device=args.dense_device,
                    max_seq_length=args.dense_max_seq_length,
                ).run(eval_queries, top_k=args.eval_top_k)
                trained_run = RetrievalRun(f"trained:{strategy}", trained_run.rankings)
                row = {"run": trained_run.name, "strategy": strategy, "model_path": str(model_dir)}
                row.update(evaluate_run(trained_run, eval_qrels, ks=_metric_ks(args.eval_top_k)))
                row.update(
                    constraint_violation_at_k(
                        trained_run, eval_queries, docs, eval_qrels, ontology, k=10, judge=judge
                    )
                )
                trained_retrieval_rows.append(row)
                trained_per_query[strategy] = evaluate_run_per_query(
                    trained_run, eval_qrels, ks=_metric_ks(args.eval_top_k)
                )

    _write_negative_quality(
        out_dir / "negative_quality.csv",
        summarize_negative_quality(negatives_by_strategy, docs, queries, qrels, ontology, judge),
    )
    retrieval_rows = []
    for run in (
        BM25Retriever(docs).run(eval_queries, top_k=args.eval_top_k),
        DenseRetriever(
            docs,
            model_name=args.dense_model,
            backend=args.dense_backend,
            batch_size=args.dense_batch_size,
            device=args.dense_device,
            max_seq_length=args.dense_max_seq_length,
        ).run(eval_queries, top_k=args.eval_top_k),
    ):
        row = {"run": run.name}
        row.update(evaluate_run(run, eval_qrels, ks=_metric_ks(args.eval_top_k)))
        row.update(constraint_violation_at_k(run, eval_queries, docs, eval_qrels, ontology, k=10, judge=judge))
        retrieval_rows.append(row)
    (out_dir / "retrieval_metrics.json").write_text(
        json.dumps(retrieval_rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    if trained_retrieval_rows:
        (out_dir / "trained_retrieval_metrics.json").write_text(
            json.dumps(trained_retrieval_rows, indent=2, sort_keys=True), encoding="utf-8"
        )
        _write_per_query_metrics(out_dir / "trained_per_query_metrics.csv", trained_per_query)
    if training_summaries:
        _write_training_summaries(out_dir, training_summaries)
    print(f"wrote comparison artifacts to {out_dir}")


def cmd_eval_trained_models(args: argparse.Namespace) -> None:
    docs = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    ontology = Ontology.load(args.ontology)
    judge = HeuristicConstraintJudge()
    models_dir = Path(args.models_dir)
    model_dirs = (
        [models_dir / strategy for strategy in args.strategies]
        if args.strategies
        else sorted(path for path in models_dir.iterdir() if path.is_dir())
    )
    rows = []
    for model_dir in model_dirs:
        if not model_dir.exists():
            raise FileNotFoundError(f"could not find trained model directory: {model_dir}")
        run = DenseRetriever(
            docs,
            model_name=str(model_dir),
            backend="sentence-transformers",
            batch_size=args.dense_batch_size,
            device=args.dense_device,
            max_seq_length=args.dense_max_seq_length,
        ).run(queries, top_k=args.eval_top_k)
        run = RetrievalRun(f"trained:{model_dir.name}", run.rankings)
        row = {"run": run.name, "strategy": model_dir.name, "model_path": str(model_dir)}
        row.update(evaluate_run(run, qrels, ks=_metric_ks(args.eval_top_k)))
        row.update(
            constraint_violation_at_k(
                run, queries, docs, qrels, ontology, k=args.constraint_k, judge=judge
            )
        )
        rows.append(row)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote trained model retrieval metrics to {out_path}")


def _validate_qrels_match_queries(queries, qrels, split_name: str) -> None:
    query_ids = {query.id for query in queries}
    qrel_query_ids = {qrel.query_id for qrel in qrels if qrel.relevance > 0}
    unknown = sorted(qrel_query_ids - query_ids)
    missing = sorted(query_ids - qrel_query_ids)
    if unknown:
        raise ValueError(f"{split_name} qrels contain unknown query IDs; first={unknown[0]!r}")
    if missing:
        raise ValueError(f"{split_name} contains queries without positive qrels; first={missing[0]!r}")


def _write_negative_quality(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "strategy",
        "negatives",
        "valid_hard_rate",
        "false_negative_rate",
        "target_leakage_rate",
        "avg_query_doc_overlap",
        "avg_confidence",
        "violation_distribution",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["violation_distribution"] = json.dumps(
                output["violation_distribution"], sort_keys=True
            )
            writer.writerow(output)


def _write_per_query_metrics(path: Path, per_query: dict[str, dict[str, dict[str, float]]]) -> None:
    metric_names = sorted(
        {metric for strategy_rows in per_query.values() for row in strategy_rows.values() for metric in row}
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["strategy", "query_id", *metric_names])
        writer.writeheader()
        for strategy, strategy_rows in sorted(per_query.items()):
            for query_id, metrics in sorted(strategy_rows.items()):
                writer.writerow({"strategy": strategy, "query_id": query_id, **metrics})


def _write_training_summaries(out_dir: Path, summaries: list[dict]) -> None:
    (out_dir / "training_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )
    fieldnames = [
        "strategy", "model_name", "triplets", "epochs", "batch_size", "loss",
        "learning_rate", "warmup_ratio", "random_seed", "deterministic",
        "cached_mini_batch_size", "mnrl_scale", "steps_per_epoch", "total_steps",
        "duration_sec", "logged_events", "output_dir",
    ]
    with (out_dir / "training_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _looks_like_sentence_transformer_model(path: Path) -> bool:
    return path.exists() and (path / "modules.json").exists()


def _add_download_parser(subparsers) -> None:
    parser = subparsers.add_parser("download", help="Download a paper dataset.")
    parser.add_argument("dataset", choices=["mooccubex", "course-skill-atlas"])
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--metadata-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--file-pattern")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--mooccubex-preset", choices=["core", "course", "concept", "all"], default="core")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--strict-downloads", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--insecure-ssl", action="store_true")
    parser.set_defaults(func=cmd_download)


def _add_data_parsers(subparsers) -> None:
    parser = subparsers.add_parser("build-ontology", help="Build a paper dataset ontology.")
    parser.add_argument("source", choices=["mooccubex", "course-skill-atlas"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-concepts", type=int)
    parser.add_argument("--max-prerequisites", type=int)
    parser.add_argument("--no-abilities", action="store_true")
    parser.add_argument("--no-tasks", action="store_true")
    parser.set_defaults(func=cmd_build_ontology)

    parser = subparsers.add_parser("prepare-mooccubex")
    parser.add_argument("--input", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--out", default="data/processed/mooccubex")
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--max-courses", type=int)
    parser.set_defaults(func=cmd_prepare_mooccubex)

    parser = subparsers.add_parser("prepare-course-skill-atlas")
    parser.add_argument("--input", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--out", default="data/processed/course_skill_atlas")
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--max-docs", type=int)
    parser.add_argument("--min-syllabi", type=int, default=1)
    parser.add_argument("--max-positives-per-query", type=int, default=100)
    parser.set_defaults(func=cmd_prepare_course_skill_atlas)


def _add_compare_parser(subparsers) -> None:
    parser = subparsers.add_parser("compare", help="Mine, train, and evaluate the paper strategies.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--dense-model")
    parser.add_argument("--dense-backend", choices=["auto", "hash", "tfidf", "sentence-transformers"], default="auto")
    parser.add_argument("--dense-batch-size", type=int, default=8)
    parser.add_argument("--dense-device")
    parser.add_argument("--dense-max-seq-length", type=int, default=128)
    parser.add_argument("--out", default="runs/comparison")
    parser.add_argument("--strategies", nargs="*")
    parser.add_argument("--eval-queries")
    parser.add_argument("--eval-qrels")
    parser.add_argument("--artifacts-from")
    parser.add_argument("--eval-top-k", type=int, default=100)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--triplet-seed", type=int, default=17)
    parser.add_argument("--train-seed", type=int, default=13)
    parser.add_argument("--train-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-device")
    parser.add_argument("--train-max-seq-length", type=int, default=128)
    parser.add_argument("--train-use-amp", action="store_true")
    parser.add_argument("--loss", choices=["triplet", "cached-mnrl"], default="triplet")
    parser.add_argument("--cached-mini-batch-size", type=int, default=4)
    parser.add_argument("--mnrl-scale", type=float, default=20.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--no-eval-trained-models", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(func=cmd_compare)


def _add_evaluation_parser(subparsers) -> None:
    parser = subparsers.add_parser("eval-trained-models")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--strategies", nargs="*")
    parser.add_argument("--eval-top-k", type=int, default=100)
    parser.add_argument("--constraint-k", type=int, default=10)
    parser.add_argument("--dense-batch-size", type=int, default=8)
    parser.add_argument("--dense-device")
    parser.add_argument("--dense-max-seq-length", type=int, default=128)
    parser.add_argument("--out", default="runs/trained_retrieval_metrics.json")
    parser.set_defaults(func=cmd_eval_trained_models)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cahnm",
        description="Reproduce the CA-HNM experiments reported in the corrected paper.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_download_parser(subparsers)
    _add_data_parsers(subparsers)
    _add_compare_parser(subparsers)
    _add_evaluation_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
