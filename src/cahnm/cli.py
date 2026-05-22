from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
import random
from pathlib import Path

from .baselines import DEFAULT_STRATEGIES, BaselineConfig, NegativeStrategyRunner, summarize_negative_quality
from .data.download import (
    build_course_skill_atlas_ontology,
    build_esco_ontology,
    build_assistments_ontology,
    build_mooccubex_ontology,
    build_onet_ontology,
    download_assistments_2009,
    download_course_skill_atlas,
    download_mooccubex,
    download_onet,
    write_sample_dataset,
)
from .data.prepare import prepare_course_skill_atlas_benchmark, prepare_keyword_benchmark, prepare_mooccubex_benchmark
from .evaluation import constraint_violation_at_k, evaluate_run
from .external import load_external_negatives, load_run
from .io_utils import load_corpus, load_negatives, load_qrels, load_queries, read_jsonl, write_jsonl, write_negatives
from .judges import make_judge
from .miner import CAHNMiner, MiningConfig, positives_by_query
from .ontology import Ontology
from .retrievers import BM25Retriever, DenseRetriever
from .schemas import RetrievalRun
from .training import build_triplets, train_sentence_transformer_triplets, write_triplets


def _baseline_config(**kwargs) -> BaselineConfig:
    allowed = getattr(BaselineConfig, "__dataclass_fields__", {})
    return BaselineConfig(**{key: value for key, value in kwargs.items() if key in allowed})


def _mining_config(**kwargs) -> MiningConfig:
    allowed = getattr(MiningConfig, "__dataclass_fields__", {})
    return MiningConfig(**{key: value for key, value in kwargs.items() if key in allowed})


def _make_judge_from_args(args: argparse.Namespace):
    return make_judge(
        args.judge,
        args.llm_model,
        base_url=args.llm_base_url,
        cache_path=args.llm_cache,
        candidate_chars=getattr(args, "llm_candidate_chars", 1200),
        context_items=getattr(args, "llm_context_items", 10),
        label_chars=getattr(args, "llm_label_chars", 80),
        max_retries=getattr(args, "llm_retries", 2),
        retry_sleep=getattr(args, "llm_retry_sleep", 10.0),
    )


def cmd_download(args: argparse.Namespace) -> None:
    if args.insecure_ssl:
        os.environ["CAHNM_INSECURE_SSL"] = "1"
    if args.dataset == "sample":
        path = write_sample_dataset(args.out)
    elif args.dataset == "mooccubex":
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
    elif args.dataset == "onet":
        path = download_onet(args.out, version=args.version)
    elif args.dataset == "assistments2009":
        path = download_assistments_2009(args.out)
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
    else:
        raise ValueError(args.dataset)
    print(path)


def cmd_build_ontology(args: argparse.Namespace) -> None:
    if args.source == "sample":
        path = write_sample_dataset(Path(args.out).parent)
        print(Path(path) / "ontology.json")
    elif args.source == "onet":
        print(build_onet_ontology(args.input, args.out))
    elif args.source == "mooccubex":
        print(
            build_mooccubex_ontology(
                args.input,
                args.out,
                max_concepts=args.max_concepts,
                max_prerequisites=args.max_prerequisites,
            )
        )
    elif args.source == "course-skill-atlas":
        print(
            build_course_skill_atlas_ontology(
                args.input,
                args.out,
                include_abilities=not args.no_abilities,
                include_tasks=not args.no_tasks,
            )
        )
    elif args.source == "assistments":
        print(build_assistments_ontology(args.input, args.out))
    elif args.source == "esco":
        print(build_esco_ontology(args.input, args.out))
    else:
        raise ValueError(args.source)


def cmd_prepare(args: argparse.Namespace) -> None:
    print(
        prepare_keyword_benchmark(
            corpus_path=args.corpus,
            ontology_path=args.ontology,
            output_dir=args.out,
            max_queries=args.max_queries,
        )
    )


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


def cmd_mine(args: argparse.Namespace) -> None:
    docs = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    ontology = Ontology.load(args.ontology)
    judge = _make_judge_from_args(args)
    miner = CAHNMiner(
        docs,
        ontology,
        judge=judge,
        config=_mining_config(
            top_k_bm25=args.top_k,
            top_k_dense=args.top_k,
            max_negatives_per_query=args.negatives_per_query,
            confidence_threshold=args.confidence,
            dense_model=args.dense_model,
            dense_backend=args.dense_backend,
            dense_batch_size=args.dense_batch_size,
            dense_device=args.dense_device,
            dense_max_seq_length=args.dense_max_seq_length,
            random_seed=args.random_seed,
            candidate_fusion=args.candidate_fusion,
            selection_policy=args.selection_policy,
            retrieval_weight=args.retrieval_weight,
            constraint_weight=args.constraint_weight,
            ontology_weight=args.ontology_weight,
            diversity_penalty=args.diversity_penalty,
        ),
    )
    negatives = miner.mine(queries, qrels)
    write_negatives(args.out, negatives)
    print(f"wrote {len(negatives)} negatives to {args.out}")


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
    judge = _make_judge_from_args(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = NegativeStrategyRunner(
        docs,
        queries,
        qrels,
        ontology,
        judge=judge,
        config=_baseline_config(
            max_negatives_per_query=args.negatives_per_query,
            top_k=args.top_k,
            dense_model=args.dense_model,
            dense_backend=args.dense_backend,
            dense_batch_size=args.dense_batch_size,
            dense_device=args.dense_device,
            dense_max_seq_length=args.dense_max_seq_length,
            random_seed=args.random_seed,
            confidence_threshold=args.confidence,
            candidate_fusion=args.candidate_fusion,
            selection_policy=args.selection_policy,
            retrieval_weight=args.retrieval_weight,
            constraint_weight=args.constraint_weight,
            ontology_weight=args.ontology_weight,
            diversity_penalty=args.diversity_penalty,
        ),
    )
    strategies = args.strategies or DEFAULT_STRATEGIES
    negatives_by_strategy = {}
    trained_retrieval_rows = []
    training_summaries = []
    for strategy in strategies:
        negatives_path = out_dir / f"{strategy}.negatives.jsonl"
        triplets_path = out_dir / f"{strategy}.triplets.jsonl"
        model_dir = out_dir / "models" / strategy

        if args.resume and negatives_path.exists():
            print(f"resume: loading existing negatives for {strategy} from {negatives_path}")
            negatives = load_negatives(negatives_path)
        else:
            negatives = runner.run(strategy)
            write_negatives(negatives_path, negatives)
        negatives_by_strategy[strategy] = negatives

        if args.resume and triplets_path.exists():
            print(f"resume: loading existing triplets for {strategy} from {triplets_path}")
            triplets = list(read_jsonl(triplets_path))
        else:
            triplets = build_triplets(queries, docs, qrels, negatives)
            write_triplets(triplets_path, triplets)

        if args.train and triplets:
            if args.resume and _looks_like_sentence_transformer_model(model_dir):
                print(f"resume: found trained model for {strategy} at {model_dir}; skipping training")
            else:
                train_summary = train_sentence_transformer_triplets(
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
                )
                training_summaries.append(train_summary)
            if not args.no_eval_trained_models:
                trained_run = DenseRetriever(
                    docs,
                    model_name=str(model_dir),
                    backend="sentence-transformers",
                    batch_size=args.dense_batch_size,
                    device=args.dense_device,
                    max_seq_length=args.dense_max_seq_length,
                ).run(queries, top_k=args.eval_top_k)
                trained_run = RetrievalRun(f"trained:{strategy}", trained_run.rankings)
                row = {"run": trained_run.name, "strategy": strategy, "model_path": str(model_dir)}
                row.update(evaluate_run(trained_run, qrels, ks=_metric_ks(args.eval_top_k)))
                row.update(constraint_violation_at_k(trained_run, queries, docs, qrels, ontology, k=10, judge=judge))
                trained_retrieval_rows.append(row)

    quality_rows = summarize_negative_quality(negatives_by_strategy, docs, queries, qrels, ontology, judge)
    quality_path = out_dir / "negative_quality.csv"
    with quality_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy",
                "negatives",
                "valid_hard_rate",
                "false_negative_rate",
                "target_leakage_rate",
                "avg_query_doc_overlap",
                "avg_confidence",
                "violation_distribution",
            ],
        )
        writer.writeheader()
        for row in quality_rows:
            row = dict(row)
            row["violation_distribution"] = json.dumps(row["violation_distribution"], sort_keys=True)
            writer.writerow(row)

    bm25 = BM25Retriever(docs).run(queries, top_k=args.eval_top_k)
    dense = DenseRetriever(
        docs,
        model_name=args.dense_model,
        backend=args.dense_backend,
        batch_size=args.dense_batch_size,
        device=args.dense_device,
        max_seq_length=args.dense_max_seq_length,
    ).run(queries, top_k=args.eval_top_k)
    retrieval_rows = []
    for run in [bm25, dense]:
        row = {"run": run.name}
        row.update(evaluate_run(run, qrels, ks=_metric_ks(args.eval_top_k)))
        row.update(constraint_violation_at_k(run, queries, docs, qrels, ontology, k=10, judge=judge))
        retrieval_rows.append(row)
    retrieval_path = out_dir / "retrieval_metrics.json"
    retrieval_path.write_text(json.dumps(retrieval_rows, indent=2, sort_keys=True), encoding="utf-8")
    if trained_retrieval_rows:
        trained_path = out_dir / "trained_retrieval_metrics.json"
        trained_path.write_text(json.dumps(trained_retrieval_rows, indent=2, sort_keys=True), encoding="utf-8")
    if training_summaries:
        _write_training_summaries(out_dir, training_summaries)
    print(f"wrote comparison artifacts to {out_dir}")


def cmd_eval_trained_models(args: argparse.Namespace) -> None:
    docs = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    ontology = Ontology.load(args.ontology)
    judge = _make_judge_from_args(args)
    models_dir = Path(args.models_dir)
    if args.strategies:
        model_dirs = [models_dir / strategy for strategy in args.strategies]
    else:
        model_dirs = sorted(path for path in models_dir.iterdir() if path.is_dir())

    rows = []
    for model_dir in model_dirs:
        if not model_dir.exists():
            raise FileNotFoundError(f"Could not find trained model directory: {model_dir}")
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
        row.update(constraint_violation_at_k(run, queries, docs, qrels, ontology, k=args.constraint_k, judge=judge))
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote trained model retrieval metrics to {out_path}")


def _write_training_summaries(out_dir: Path, summaries: list[dict]) -> None:
    summary_path = out_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = out_dir / "training_summary.csv"
    fieldnames = [
        "strategy",
        "model_name",
        "triplets",
        "epochs",
        "batch_size",
        "loss",
        "learning_rate",
        "warmup_ratio",
        "steps_per_epoch",
        "total_steps",
        "duration_sec",
        "logged_events",
        "output_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _looks_like_sentence_transformer_model(path: Path) -> bool:
    return path.exists() and (path / "modules.json").exists()


def cmd_eval_run(args: argparse.Namespace) -> None:
    docs = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    ontology = Ontology.load(args.ontology)
    judge = _make_judge_from_args(args)
    run = load_run(args.run, run_format=args.run_format, name=args.name, queries=queries)

    row = {"run": run.name}
    row.update(evaluate_run(run, qrels, ks=tuple(args.k)))
    row.update(constraint_violation_at_k(run, queries, docs, qrels, ontology, k=args.constraint_k, judge=judge))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote external run metrics to {out_path}")


def cmd_normalize_negatives(args: argparse.Namespace) -> None:
    negatives = load_external_negatives(
        args.input,
        source=args.source,
        input_format=args.input_format,
        label=args.label,
    )
    write_negatives(args.out, negatives)
    print(f"wrote {len(negatives)} normalized negatives to {args.out}")


def cmd_validate_negatives(args: argparse.Namespace) -> None:
    docs = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    ontology = Ontology.load(args.ontology)
    negatives = load_negatives(args.negatives)
    if args.source:
        negatives = [negative for negative in negatives if negative.source == args.source]
    sampled_from = len(negatives)
    if args.sample_size and len(negatives) > args.sample_size:
        negatives = random.Random(args.seed).sample(negatives, args.sample_size)

    judge = _make_judge_from_args(args)
    doc_by_id = {doc.id: doc for doc in docs}
    query_by_id = {query.id: query for query in queries}
    positives = positives_by_query(qrels)
    label_counts: Counter[str] = Counter()
    violation_counts: Counter[str] = Counter()
    false_negative = 0
    valid_hard = 0
    missing_query = 0
    missing_doc = 0
    decisions = []

    for negative in negatives:
        query = query_by_id.get(negative.query_id)
        doc = doc_by_id.get(negative.doc_id)
        if query is None:
            missing_query += 1
            continue
        if doc is None:
            missing_doc += 1
            continue
        concept = query.target_concept
        if concept is None:
            inferred = ontology.find_in_text(query.text)
            concept = inferred.id if inferred else None
        decision = judge.classify(query, doc, ontology.context(concept))
        label_counts[decision.label] += 1
        violation_counts.update(decision.violation_types)
        is_false_negative = negative.doc_id in positives.get(negative.query_id, set())
        false_negative += int(is_false_negative)
        valid_hard += int(decision.is_hard_negative)
        decisions.append(
            {
                "query_id": negative.query_id,
                "doc_id": negative.doc_id,
                "source": negative.source,
                "original_label": negative.label,
                "original_violation_types": list(negative.violation_types),
                "judge_label": decision.label,
                "judge_violation_types": list(decision.violation_types),
                "judge_evidence": decision.evidence,
                "judge_confidence": decision.confidence,
                "is_false_negative": is_false_negative,
            }
        )

    evaluated = len(decisions)
    summary = {
        "negative_file": str(args.negatives),
        "source_filter": args.source,
        "sampled_from": sampled_from,
        "evaluated": evaluated,
        "missing_query": missing_query,
        "missing_doc": missing_doc,
        "judge": args.judge,
        "llm_model": args.llm_model,
        "llm_base_url": args.llm_base_url,
        "valid_hard_rate": valid_hard / evaluated if evaluated else 0.0,
        "false_negative_rate": false_negative / evaluated if evaluated else 0.0,
        "label_distribution": dict(label_counts),
        "violation_distribution": dict(violation_counts),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.decisions_out:
        write_jsonl(args.decisions_out, decisions)
    print(f"wrote LLM/constraint validation summary to {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cahnm")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Download or create raw datasets.")
    p.add_argument("dataset", choices=["sample", "mooccubex", "onet", "assistments2009", "course-skill-atlas"])
    p.add_argument("--out", default="data/raw")
    p.add_argument("--version", default="30_2", help="O*NET version token, e.g. 30_2.")
    p.add_argument("--metadata-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--file-pattern", help="Dataset file name substring to download.")
    p.add_argument("--max-files", type=int)
    p.add_argument("--mooccubex-preset", choices=["core", "course", "concept", "all"], default="core")
    p.add_argument("--force", action="store_true", help="Re-download files even if complete files already exist.")
    p.add_argument("--no-resume", action="store_true", help="Do not resume from .part files.")
    p.add_argument("--strict-downloads", action="store_true", help="Fail instead of skipping unavailable optional files.")
    p.add_argument("--timeout", type=int, default=300, help="Per-request download timeout in seconds.")
    p.add_argument("--retries", type=int, default=5, help="Retry count for transient download failures.")
    p.add_argument(
        "--insecure-ssl",
        action="store_true",
        help="Disable TLS certificate verification for dataset download requests.",
    )
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("build-ontology", help="Build ontology.json from source files.")
    p.add_argument("source", choices=["sample", "onet", "mooccubex", "course-skill-atlas", "assistments", "esco"])
    p.add_argument("--input", help="Input O*NET extract directory or ASSISTments CSV.")
    p.add_argument("--out", default="data/processed/ontology.json")
    p.add_argument("--max-concepts", type=int, help="Optional cap for MOOCCubeX concept ontology building.")
    p.add_argument("--max-prerequisites", type=int, help="Optional cap for MOOCCubeX prerequisite edges.")
    p.add_argument("--no-abilities", action="store_true", help="Do not include Course-Skill Atlas ability nodes.")
    p.add_argument("--no-tasks", action="store_true", help="Do not include Course-Skill Atlas task nodes.")
    p.set_defaults(func=cmd_build_ontology)

    p = sub.add_parser("prepare", help="Prepare weak query/qrel files from corpus and ontology.")
    p.add_argument("--corpus", required=True)
    p.add_argument("--ontology", required=True)
    p.add_argument("--out", default="data/processed")
    p.add_argument("--max-queries", type=int, default=100)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("prepare-mooccubex", help="Prepare corpus/query/qrel files from MOOCCubeX course and concept links.")
    p.add_argument("--input", required=True, help="MOOCCubeX raw directory.")
    p.add_argument("--ontology", required=True)
    p.add_argument("--out", default="data/processed/mooccubex")
    p.add_argument("--max-queries", type=int, default=1000)
    p.add_argument("--max-courses", type=int)
    p.set_defaults(func=cmd_prepare_mooccubex)

    p = sub.add_parser("prepare-course-skill-atlas", help="Prepare corpus/query/qrel files from Course-Skill Atlas.")
    p.add_argument("--input", required=True, help="Course-Skill Atlas raw directory.")
    p.add_argument("--ontology", required=True)
    p.add_argument("--out", default="data/processed/course_skill_atlas")
    p.add_argument("--max-queries", type=int, default=1000)
    p.add_argument("--max-docs", type=int)
    p.add_argument("--min-syllabi", type=int, default=1)
    p.add_argument("--max-positives-per-query", type=int, default=100, help="Use 0 for no cap.")
    p.set_defaults(func=cmd_prepare_course_skill_atlas)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", required=True)
    common.add_argument("--queries", required=True)
    common.add_argument("--qrels", required=True)
    common.add_argument("--ontology", required=True)
    common.add_argument("--top-k", type=int, default=50)
    common.add_argument("--negatives-per-query", type=int, default=8)
    common.add_argument("--confidence", type=float, default=0.75)
    common.add_argument("--random-seed", type=int, default=13)
    common.add_argument("--dense-model")
    common.add_argument("--dense-backend", choices=["auto", "hash", "tfidf", "sentence-transformers"], default="auto")
    common.add_argument("--dense-batch-size", type=int, default=16, help="Batch size for sentence-transformers encoding.")
    common.add_argument("--dense-device", help="Device for sentence-transformers encoding, e.g. cpu, cuda, cuda:1.")
    common.add_argument("--dense-max-seq-length", type=int, help="Optional max token length for dense encoder inputs.")
    common.add_argument(
        "--candidate-fusion",
        choices=["max_score", "rrf"],
        default="max_score",
        help="Fusion used for BM25+dense CA-HNM candidate pools.",
    )
    common.add_argument(
        "--selection-policy",
        choices=["balanced", "retrieval_aware", "top_ranked"],
        default="balanced",
        help="Policy used to select CA-HNM candidates after constraint classification.",
    )
    common.add_argument("--retrieval-weight", type=float, default=0.55, help="Retrieval-hardness weight for retrieval-aware CA-HNM selection.")
    common.add_argument("--constraint-weight", type=float, default=0.30, help="Constraint-confidence weight for retrieval-aware CA-HNM selection.")
    common.add_argument("--ontology-weight", type=float, default=0.15, help="Ontology-hardness weight for retrieval-aware CA-HNM selection.")
    common.add_argument("--diversity-penalty", type=float, default=0.06, help="Penalty for repeatedly selecting the same violation type.")
    common.add_argument("--judge", choices=["heuristic", "openai", "openai-compatible", "lmstudio"], default="heuristic")
    common.add_argument("--llm-model")
    common.add_argument("--llm-base-url", help="OpenAI-compatible API base URL, e.g. http://localhost:1234/v1 for LM Studio.")
    common.add_argument("--llm-cache", help="Optional JSONL cache for LLM constraint decisions.")
    common.add_argument("--llm-candidate-chars", type=int, default=1200, help="Maximum candidate text characters sent to the LLM judge.")
    common.add_argument("--llm-context-items", type=int, default=10, help="Maximum ontology labels per relation type sent to the LLM judge.")
    common.add_argument("--llm-label-chars", type=int, default=80, help="Maximum characters per ontology label sent to the LLM judge.")
    common.add_argument("--llm-retries", type=int, default=2, help="Retry count for transient LLM server connection failures.")
    common.add_argument("--llm-retry-sleep", type=float, default=10.0, help="Seconds to sleep between LLM connection retries.")

    p = sub.add_parser("mine", parents=[common], help="Mine CA-HNM negatives.")
    p.add_argument("--out", default="runs/CA-HNM.negatives.jsonl")
    p.set_defaults(func=cmd_mine)

    p = sub.add_parser("compare", parents=[common], help="Run baseline negative mining and retrieval comparison.")
    p.add_argument("--out", default="runs/comparison")
    p.add_argument("--strategies", nargs="*")
    p.add_argument("--eval-top-k", type=int, default=100)
    p.add_argument("--train", action="store_true", help="Train one sentence-transformers model per strategy.")
    p.add_argument("--train-model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--train-device", help="Device for fine-tuning, e.g. cuda, cuda:1, cpu.")
    p.add_argument("--train-max-seq-length", type=int, help="Optional max token length during triplet fine-tuning.")
    p.add_argument("--train-use-amp", action="store_true", help="Use mixed precision during sentence-transformers fine-tuning when supported.")
    p.add_argument(
        "--loss",
        choices=["triplet", "mnrl", "multiple-negatives", "hybrid"],
        default="triplet",
        help="Sentence-transformers training loss.",
    )
    p.add_argument("--learning-rate", type=float, help="Optional optimizer learning rate for sentence-transformers fit().")
    p.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio used when fit() supports warmup_steps.")
    p.add_argument("--no-eval-trained-models", action="store_true", help="Skip retrieval evaluation of trained models.")
    p.add_argument("--resume", action="store_true", help="Reuse existing negatives, triplets, and trained model directories when present.")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("eval-trained-models", help="Evaluate saved sentence-transformers retriever models.")
    p.add_argument("--corpus", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--qrels", required=True)
    p.add_argument("--ontology", required=True)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--strategies", nargs="*", help="Model subdirectories to evaluate. Defaults to all subdirectories.")
    p.add_argument("--eval-top-k", type=int, default=100)
    p.add_argument("--constraint-k", type=int, default=10)
    p.add_argument("--dense-batch-size", type=int, default=16, help="Batch size for sentence-transformers encoding.")
    p.add_argument("--dense-device", help="Device for sentence-transformers encoding, e.g. cpu, cuda, cuda:1.")
    p.add_argument("--dense-max-seq-length", type=int, help="Optional max token length for dense encoder inputs.")
    p.add_argument("--judge", choices=["heuristic", "openai", "openai-compatible", "lmstudio"], default="heuristic")
    p.add_argument("--llm-model")
    p.add_argument("--llm-base-url", help="OpenAI-compatible API base URL, e.g. http://localhost:1234/v1 for LM Studio.")
    p.add_argument("--llm-cache", help="Optional JSONL cache for LLM constraint decisions.")
    p.add_argument("--llm-candidate-chars", type=int, default=1200, help="Maximum candidate text characters sent to the LLM judge.")
    p.add_argument("--llm-context-items", type=int, default=10, help="Maximum ontology labels per relation type sent to the LLM judge.")
    p.add_argument("--llm-label-chars", type=int, default=80, help="Maximum characters per ontology label sent to the LLM judge.")
    p.add_argument("--llm-retries", type=int, default=2, help="Retry count for transient LLM server connection failures.")
    p.add_argument("--llm-retry-sleep", type=float, default=10.0, help="Seconds to sleep between LLM connection retries.")
    p.add_argument("--out", default="runs/trained_retrieval_metrics.json")
    p.set_defaults(func=cmd_eval_trained_models)

    p = sub.add_parser("eval-run", help="Evaluate an external official retriever run.")
    p.add_argument("--corpus", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--qrels", required=True)
    p.add_argument("--ontology", required=True)
    p.add_argument("--run", required=True, help="TREC, DPR JSON, or BEIR-style run file.")
    p.add_argument("--run-format", choices=["auto", "trec", "json", "jsonl", "dpr", "beir"], default="auto")
    p.add_argument("--name", help="Run name to write into metrics.")
    p.add_argument("--k", type=int, nargs="+", default=[10, 100])
    p.add_argument("--constraint-k", type=int, default=10)
    p.add_argument("--judge", choices=["heuristic", "openai", "openai-compatible", "lmstudio"], default="heuristic")
    p.add_argument("--llm-model")
    p.add_argument("--llm-base-url", help="OpenAI-compatible API base URL, e.g. http://localhost:1234/v1 for LM Studio.")
    p.add_argument("--llm-cache", help="Optional JSONL cache for LLM constraint decisions.")
    p.add_argument("--llm-candidate-chars", type=int, default=1200, help="Maximum candidate text characters sent to the LLM judge.")
    p.add_argument("--llm-context-items", type=int, default=10, help="Maximum ontology labels per relation type sent to the LLM judge.")
    p.add_argument("--llm-label-chars", type=int, default=80, help="Maximum characters per ontology label sent to the LLM judge.")
    p.add_argument("--llm-retries", type=int, default=2, help="Retry count for transient LLM server connection failures.")
    p.add_argument("--llm-retry-sleep", type=float, default=10.0, help="Seconds to sleep between LLM connection retries.")
    p.add_argument("--out", default="runs/external_run_metrics.json")
    p.set_defaults(func=cmd_eval_run)

    p = sub.add_parser("normalize-negatives", help="Convert external negatives/runs into CA-HNM JSONL records.")
    p.add_argument("--input", required=True)
    p.add_argument("--input-format", choices=["auto", "trec", "json", "jsonl", "table", "tsv", "csv"], default="auto")
    p.add_argument("--source", required=True, help="Baseline name, e.g. ANCE or GPL.")
    p.add_argument("--label", default="HardNeg")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_normalize_negatives)

    p = sub.add_parser("validate-negatives", help="Validate mined negatives with a heuristic or LLM constraint judge.")
    p.add_argument("--corpus", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--qrels", required=True)
    p.add_argument("--ontology", required=True)
    p.add_argument("--negatives", required=True)
    p.add_argument("--source", help="Optional negative source/strategy filter.")
    p.add_argument("--sample-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--judge", choices=["heuristic", "openai", "openai-compatible", "lmstudio"], default="heuristic")
    p.add_argument("--llm-model")
    p.add_argument("--llm-base-url", help="OpenAI-compatible API base URL, e.g. http://localhost:1234/v1 for LM Studio.")
    p.add_argument("--llm-cache", help="Optional JSONL cache for LLM constraint decisions.")
    p.add_argument("--llm-candidate-chars", type=int, default=1200, help="Maximum candidate text characters sent to the LLM judge.")
    p.add_argument("--llm-context-items", type=int, default=10, help="Maximum ontology labels per relation type sent to the LLM judge.")
    p.add_argument("--llm-label-chars", type=int, default=80, help="Maximum characters per ontology label sent to the LLM judge.")
    p.add_argument("--llm-retries", type=int, default=2, help="Retry count for transient LLM server connection failures.")
    p.add_argument("--llm-retry-sleep", type=float, default=10.0, help="Seconds to sleep between LLM connection retries.")
    p.add_argument("--out", default="runs/negative_validation_summary.json")
    p.add_argument("--decisions-out", help="Optional JSONL path for per-negative judge decisions.")
    p.set_defaults(func=cmd_validate_negatives)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
