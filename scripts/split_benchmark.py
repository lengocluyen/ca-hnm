#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cahnm.io_utils import load_qrels, load_queries, write_qrels, write_queries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic, target-disjoint train/dev/test query and qrel splits."
    )
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    args = parser.parse_args()

    ratios = {
        "train": args.train_ratio,
        "dev": args.dev_ratio,
        "test": args.test_ratio,
    }
    if any(value <= 0 for value in ratios.values()):
        parser.error("all split ratios must be positive")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        parser.error("train, dev, and test ratios must sum to 1.0")

    query_path = Path(args.queries)
    qrels_path = Path(args.qrels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = load_queries(query_path)
    qrels = load_qrels(qrels_path)
    query_ids = {query.id for query in queries}
    if len(query_ids) != len(queries):
        raise ValueError("query IDs must be unique before splitting")

    orphan_qrels = sorted({qrel.query_id for qrel in qrels if qrel.query_id not in query_ids})
    if orphan_qrels:
        raise ValueError(f"qrels reference {len(orphan_qrels)} unknown queries; first={orphan_qrels[0]!r}")

    positive_counts: dict[str, int] = defaultdict(int)
    for qrel in qrels:
        if qrel.relevance > 0:
            positive_counts[qrel.query_id] += 1
    missing_positive = sorted(query_id for query_id in query_ids if positive_counts[query_id] == 0)
    if missing_positive:
        raise ValueError(
            f"{len(missing_positive)} queries have no positive judgment; first={missing_positive[0]!r}"
        )

    groups: dict[str, list] = defaultdict(list)
    for query in queries:
        group_key = str(query.target_concept or query.id)
        groups[group_key].append(query)

    strata: dict[str, list[str]] = defaultdict(list)
    for group_key, group_queries in groups.items():
        count = sum(positive_counts[query.id] for query in group_queries)
        strata[_count_bucket(count)].append(group_key)

    rng = random.Random(args.seed)
    split_group_keys: dict[str, list[str]] = {name: [] for name in ratios}
    for bucket in sorted(strata):
        keys = sorted(strata[bucket])
        rng.shuffle(keys)
        counts = _allocate_counts(len(keys), ratios)
        offset = 0
        for split_name in ("train", "dev", "test"):
            end = offset + counts[split_name]
            split_group_keys[split_name].extend(keys[offset:end])
            offset = end

    split_query_ids: dict[str, set[str]] = {}
    output_files: dict[str, dict[str, str]] = {}
    for split_name in ("train", "dev", "test"):
        selected_groups = set(split_group_keys[split_name])
        selected_queries = [
            query for query in queries if str(query.target_concept or query.id) in selected_groups
        ]
        selected_ids = {query.id for query in selected_queries}
        selected_qrels = [qrel for qrel in qrels if qrel.query_id in selected_ids]
        split_query_ids[split_name] = selected_ids

        queries_out = out_dir / f"{split_name}.queries.jsonl"
        qrels_out = out_dir / f"{split_name}.qrels.tsv"
        write_queries(queries_out, selected_queries)
        write_qrels(qrels_out, selected_qrels)
        output_files[split_name] = {
            "queries": str(queries_out),
            "qrels": str(qrels_out),
            "queries_sha256": _sha256(queries_out),
            "qrels_sha256": _sha256(qrels_out),
        }

    overlaps = {
        "train_dev": len(split_query_ids["train"] & split_query_ids["dev"]),
        "train_test": len(split_query_ids["train"] & split_query_ids["test"]),
        "dev_test": len(split_query_ids["dev"] & split_query_ids["test"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"query overlap detected after split: {overlaps}")

    assigned = set().union(*split_query_ids.values())
    if assigned != query_ids:
        raise AssertionError("split assignment did not preserve the complete query set")

    manifest = {
        "schema_version": 1,
        "strategy": "target-disjoint-qrel-count-stratified",
        "seed": args.seed,
        "ratios": ratios,
        "inputs": {
            "queries": str(query_path),
            "qrels": str(qrels_path),
            "queries_sha256": _sha256(query_path),
            "qrels_sha256": _sha256(qrels_path),
        },
        "counts": {
            split_name: {
                "queries": len(split_query_ids[split_name]),
                "target_groups": len(split_group_keys[split_name]),
                "qrels": sum(1 for qrel in qrels if qrel.query_id in split_query_ids[split_name]),
            }
            for split_name in ("train", "dev", "test")
        },
        "overlap_audit": overlaps,
        "outputs": output_files,
    }
    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    print(f"wrote deterministic split manifest to {manifest_path}")


def _count_bucket(count: int) -> str:
    if count <= 1:
        return "01"
    if count <= 5:
        return "02-05"
    if count <= 20:
        return "06-20"
    if count <= 100:
        return "21-100"
    return "101+"


def _allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(ratios, key=lambda name: (raw[name] - counts[name], ratios[name], name), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
