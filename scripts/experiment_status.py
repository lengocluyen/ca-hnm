#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment completion without reading model checkpoints.")
    parser.add_argument("--root", default="runs/experiments")
    parser.add_argument("--out")
    args = parser.parse_args()

    root = Path(args.root)
    runs = []
    for path in sorted(root.rglob("experiment_run.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["path"] = str(path.parent)
        payload["per_query_metrics_present"] = (path.parent / "trained_per_query_metrics.csv").exists()
        runs.append(payload)
    statuses = Counter(run.get("status", "unknown") for run in runs)
    complete_with_metrics = sum(
        run.get("status") == "complete" and run["per_query_metrics_present"] for run in runs
    )
    report = {
        "schema_version": 1,
        "root": str(root),
        "checked_at": time.time(),
        "runs": len(runs),
        "status_counts": dict(sorted(statuses.items())),
        "complete_with_per_query_metrics": complete_with_metrics,
        "incomplete": [
            {
                "path": run["path"],
                "status": run.get("status", "unknown"),
                "metrics": run["per_query_metrics_present"],
            }
            for run in runs
            if run.get("status") != "complete" or not run["per_query_metrics_present"]
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
