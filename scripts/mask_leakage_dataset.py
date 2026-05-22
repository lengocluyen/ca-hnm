#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cahnm.io_utils import load_corpus, load_queries, write_corpus, write_queries
from cahnm.schemas import Document, Query


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create leakage-control dataset variants by masking ontology labels/aliases in text."
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--qrels", help="Optional qrels file to copy into the output directory.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--text-scope",
        choices=["none", "query-target", "all-text"],
        default="query-target",
        help="query-target masks only each query's target labels in query text; all-text masks every ontology label in queries and corpus.",
    )
    parser.add_argument("--strip-ontology-aliases", action="store_true")
    parser.add_argument("--mask-ontology-labels", action="store_true", help="Replace ontology node labels with stable concept ids.")
    parser.add_argument("--mask-token", default="[MASKED_CONCEPT]")
    parser.add_argument("--min-label-chars", type=int, default=3)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ontology_payload = json.loads(Path(args.ontology).read_text(encoding="utf-8"))
    nodes = list(ontology_payload.get("nodes", []))
    labels_by_id = _labels_by_id(nodes, min_chars=args.min_label_chars)
    all_labels = sorted({label for labels in labels_by_id.values() for label in labels}, key=len, reverse=True)

    docs = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    report = {
        "text_scope": args.text_scope,
        "strip_ontology_aliases": args.strip_ontology_aliases,
        "mask_ontology_labels": args.mask_ontology_labels,
        "query_replacements": 0,
        "document_replacements": 0,
        "labels_considered": len(all_labels),
    }

    if args.text_scope == "query-target":
        masked_queries = []
        for query in queries:
            labels = labels_by_id.get(str(query.target_concept), [])
            text, count = _replace_labels(query.text, labels, args.mask_token)
            report["query_replacements"] += count
            metadata = dict(query.metadata)
            metadata["leakage_mask"] = {"scope": args.text_scope, "replacements": count}
            masked_queries.append(Query(query.id, text, query.target_concept, metadata))
        masked_docs = docs
    elif args.text_scope == "all-text":
        patterns = _compile_label_chunks(all_labels)
        masked_queries = []
        for query in queries:
            text, count = _replace_with_patterns(query.text, patterns, args.mask_token)
            report["query_replacements"] += count
            metadata = dict(query.metadata)
            metadata["leakage_mask"] = {"scope": args.text_scope, "replacements": count}
            masked_queries.append(Query(query.id, text, query.target_concept, metadata))
        masked_docs = []
        for doc in docs:
            title, title_count = _replace_with_patterns(doc.title, patterns, args.mask_token)
            text, text_count = _replace_with_patterns(doc.text, patterns, args.mask_token)
            count = title_count + text_count
            report["document_replacements"] += count
            metadata = dict(doc.metadata)
            metadata["leakage_mask"] = {"scope": args.text_scope, "replacements": count}
            masked_docs.append(Document(doc.id, title, text, metadata))
    else:
        masked_queries = queries
        masked_docs = docs

    if args.strip_ontology_aliases or args.mask_ontology_labels:
        for node in nodes:
            if args.strip_ontology_aliases:
                node["aliases"] = []
            if args.mask_ontology_labels:
                node["label"] = f"concept_{node['id']}"
                node["aliases"] = []

    ontology_payload["nodes"] = nodes
    write_corpus(out_dir / "corpus.jsonl", masked_docs)
    write_queries(out_dir / "queries.jsonl", masked_queries)
    (out_dir / "ontology.json").write_text(json.dumps(ontology_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.qrels:
        shutil.copyfile(args.qrels, out_dir / "qrels.tsv")
    (out_dir / "masking_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote leakage-control dataset to {out_dir}")


def _labels_by_id(nodes: list[dict], min_chars: int) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for node in nodes:
        labels = [str(node.get("label", "")), *(str(alias) for alias in node.get("aliases", []))]
        labels = [label.strip() for label in labels if len(label.strip()) >= min_chars]
        result[str(node["id"])] = sorted(set(labels), key=len, reverse=True)
    return result


def _replace_labels(text: str, labels: list[str], mask_token: str) -> tuple[str, int]:
    if not labels:
        return text, 0
    return _replace_with_patterns(text, _compile_label_chunks(labels), mask_token)


def _compile_label_chunks(labels: list[str], chunk_size: int = 400) -> list[re.Pattern]:
    patterns = []
    unique_labels = sorted(set(labels), key=len, reverse=True)
    for start in range(0, len(unique_labels), chunk_size):
        chunk = unique_labels[start : start + chunk_size]
        escaped = [re.escape(label) for label in chunk if label.strip()]
        if escaped:
            patterns.append(re.compile(r"(?<!\w)(" + "|".join(escaped) + r")(?!\w)", flags=re.IGNORECASE))
    return patterns


def _replace_with_patterns(text: str, patterns: list[re.Pattern], mask_token: str) -> tuple[str, int]:
    count = 0
    updated = text
    for pattern in patterns:
        updated, n = pattern.subn(mask_token, updated)
        count += n
    return updated, count


if __name__ == "__main__":
    main()
