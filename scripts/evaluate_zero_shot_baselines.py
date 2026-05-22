#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cahnm.evaluation import constraint_violation_at_k, evaluate_run
from cahnm.io_utils import load_corpus, load_qrels, load_queries
from cahnm.judges import HeuristicConstraintJudge
from cahnm.ontology import Ontology
from cahnm.retrievers import BM25Retriever, DenseRetriever
from cahnm.schemas import Query, RetrievalRun


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BM25, zero-shot dense, and BM25+dense RRF baselines.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--dense-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--dense-backend", choices=["auto", "hash", "tfidf", "sentence-transformers"], default="sentence-transformers")
    parser.add_argument("--dense-batch-size", type=int, default=16)
    parser.add_argument("--dense-device")
    parser.add_argument("--dense-max-seq-length", type=int)
    parser.add_argument("--eval-top-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    docs = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    ontology = Ontology.load(args.ontology)
    judge = HeuristicConstraintJudge()

    bm25 = BM25Retriever(docs).run(queries, top_k=args.eval_top_k)
    dense = DenseRetriever(
        docs,
        model_name=args.dense_model,
        backend=args.dense_backend,
        batch_size=args.dense_batch_size,
        device=args.dense_device,
        max_seq_length=args.dense_max_seq_length,
    ).run(queries, top_k=args.eval_top_k)
    hybrid = _rrf_run("hybrid:bm25+dense-rrf", queries, [bm25, dense], top_k=args.eval_top_k, rrf_k=args.rrf_k)

    rows = []
    for run in [bm25, dense, hybrid]:
        row = {"run": run.name}
        row.update(evaluate_run(run, qrels, ks=(10, 100)))
        row.update(constraint_violation_at_k(run, queries, docs, qrels, ontology, k=10, judge=judge))
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote zero-shot baseline metrics to {out}")


def _rrf_run(name: str, queries: list[Query], runs: list[RetrievalRun], top_k: int, rrf_k: int) -> RetrievalRun:
    rankings: dict[str, list[tuple[str, float]]] = {}
    for query in queries:
        scores: dict[str, float] = {}
        for run in runs:
            for rank, (doc_id, _score) in enumerate(run.rankings.get(query.id, []), start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
        rankings[query.id] = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return RetrievalRun(name, rankings)


if __name__ == "__main__":
    main()
