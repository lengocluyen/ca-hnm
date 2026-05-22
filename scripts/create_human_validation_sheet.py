#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cahnm.evaluation import qrels_to_dict
from cahnm.io_utils import load_corpus, load_negatives, load_qrels, load_queries
from cahnm.ontology import Ontology
from cahnm.schemas import NegativeRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a blinded CSV sheet for human validation of mined negatives.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--negatives", nargs="+", required=True)
    parser.add_argument("--sample-size", type=int, default=300, help="Samples per negatives file.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--doc-chars", type=int, default=900)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    docs = {doc.id: doc for doc in load_corpus(args.corpus)}
    queries = {query.id: query for query in load_queries(args.queries)}
    qrels = qrels_to_dict(load_qrels(args.qrels))
    ontology = Ontology.load(args.ontology)
    rng = random.Random(args.seed)

    rows = []
    for negatives_path in args.negatives:
        path = Path(negatives_path)
        negatives = [item for item in load_negatives(path) if item.query_id in queries and item.doc_id in docs]
        sampled = _stratified_sample(negatives, args.sample_size, rng)
        for idx, negative in enumerate(sampled, start=1):
            query = queries[negative.query_id]
            doc = docs[negative.doc_id]
            context = ontology.context(query.target_concept)
            rows.append(
                {
                    "item_id": f"{path.stem}:{idx}",
                    "strategy_file": path.name,
                    "strategy": negative.source or path.stem.replace(".negatives", ""),
                    "query_id": query.id,
                    "doc_id": doc.id,
                    "query_text": query.text,
                    "target_concept": query.target_concept or "",
                    "ontology_context": _context_summary(context),
                    "doc_title": doc.title,
                    "doc_text_snippet": _snippet(doc.text, args.doc_chars),
                    "system_label": negative.label,
                    "system_violations": ", ".join(negative.violation_types),
                    "system_evidence": negative.evidence,
                    "known_positive_in_qrels": int(qrels.get(query.id, {}).get(doc.id, 0) > 0),
                    "annotator1_label": "",
                    "annotator2_label": "",
                    "adjudicated_label": "",
                    "notes": "",
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else _fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote human validation sheet with {len(rows)} rows to {out}")


def _stratified_sample(negatives: list[NegativeRecord], sample_size: int, rng: random.Random) -> list[NegativeRecord]:
    if len(negatives) <= sample_size:
        return list(negatives)
    buckets: dict[str, list[NegativeRecord]] = {}
    for item in negatives:
        key = item.violation_types[0] if item.violation_types else item.label
        buckets.setdefault(key, []).append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[NegativeRecord] = []
    keys = sorted(buckets)
    while len(selected) < sample_size and keys:
        next_keys = []
        for key in keys:
            bucket = buckets[key]
            if bucket and len(selected) < sample_size:
                selected.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def _context_summary(context) -> str:
    parts = []
    if context.target:
        parts.append(f"target={context.target.label}")
    for name, nodes in [
        ("broader", context.broader),
        ("narrower", context.narrower),
        ("siblings", context.siblings),
        ("prereq", context.prerequisites),
        ("postreq", context.postrequisites),
        ("related", context.related),
    ]:
        labels = [node.label for node in nodes[:5]]
        if labels:
            parts.append(f"{name}={'; '.join(labels)}")
    return " | ".join(parts)


def _snippet(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _fieldnames() -> list[str]:
    return [
        "item_id",
        "strategy_file",
        "strategy",
        "query_id",
        "doc_id",
        "query_text",
        "target_concept",
        "ontology_context",
        "doc_title",
        "doc_text_snippet",
        "system_label",
        "system_violations",
        "system_evidence",
        "known_positive_in_qrels",
        "annotator1_label",
        "annotator2_label",
        "adjudicated_label",
        "notes",
    ]


if __name__ == "__main__":
    main()
