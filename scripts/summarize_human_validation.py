#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


HARD_LABELS = {"hard_negative", "true_hard_negative", "hardneg", "valid_hard", "valid-hard"}
FALSE_NEG_LABELS = {"false_negative", "positive", "true_positive", "falseneg"}
AMBIGUOUS_LABELS = {"ambiguous", "partial", "unclear"}
EASY_LABELS = {"easy_negative", "easyneg", "irrelevant"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed human validation annotations.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--label-column", default="adjudicated_label")
    parser.add_argument("--fallback-label-columns", nargs="*", default=["annotator1_label", "annotator2_label"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = _read_rows(Path(args.annotations))
    summary = _summarize(rows, args.label_column, args.fallback_label_columns)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(out.with_suffix(".csv"), summary["by_strategy"])
    print(f"wrote human validation summary to {out}")


def _summarize(rows: list[dict[str, str]], label_column: str, fallback_columns: list[str]) -> dict:
    by_strategy: dict[str, list[str]] = defaultdict(list)
    missing = 0
    for row in rows:
        label = _normalized_label(row.get(label_column, ""))
        if not label:
            for column in fallback_columns:
                label = _normalized_label(row.get(column, ""))
                if label:
                    break
        if not label:
            missing += 1
            continue
        by_strategy[row.get("strategy", "unknown")].append(label)

    strategy_rows = []
    for strategy, labels in sorted(by_strategy.items()):
        counts = Counter(labels)
        n = len(labels)
        strategy_rows.append(
            {
                "strategy": strategy,
                "n": n,
                "valid_hard_rate": _rate(labels, HARD_LABELS),
                "false_negative_rate": _rate(labels, FALSE_NEG_LABELS),
                "ambiguous_rate": _rate(labels, AMBIGUOUS_LABELS),
                "easy_negative_rate": _rate(labels, EASY_LABELS),
                "label_counts": dict(sorted(counts.items())),
            }
        )

    kappa = None
    if rows and "annotator1_label" in rows[0] and "annotator2_label" in rows[0]:
        pairs = [
            (_normalized_label(row.get("annotator1_label", "")), _normalized_label(row.get("annotator2_label", "")))
            for row in rows
        ]
        pairs = [(a, b) for a, b in pairs if a and b]
        if pairs:
            kappa = _cohen_kappa(pairs)

    return {
        "n_rows": len(rows),
        "n_annotated": sum(len(labels) for labels in by_strategy.values()),
        "n_missing": missing,
        "cohen_kappa_annotator1_annotator2": kappa,
        "by_strategy": strategy_rows,
        "accepted_labels": {
            "hard": sorted(HARD_LABELS),
            "false_negative": sorted(FALSE_NEG_LABELS),
            "ambiguous": sorted(AMBIGUOUS_LABELS),
            "easy": sorted(EASY_LABELS),
        },
    }


def _rate(labels: list[str], target: set[str]) -> float:
    return sum(1 for label in labels if label in target) / len(labels) if labels else 0.0


def _cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    labels = sorted({label for pair in pairs for label in pair})
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    a_counts = Counter(a for a, _ in pairs)
    b_counts = Counter(b for _, b in pairs)
    expected = sum((a_counts[label] / n) * (b_counts[label] / n) for label in labels)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def _normalized_label(value: str | None) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    flattened = []
    for row in rows:
        row = dict(row)
        row["label_counts"] = json.dumps(row["label_counts"], sort_keys=True)
        flattened.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


if __name__ == "__main__":
    main()
