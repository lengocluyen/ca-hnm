#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_experiment_matrix import DATASETS, MODELS, PROFILES


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit a deterministic server/Slurm experiment job matrix.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), required=True)
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), required=True)
    parser.add_argument("--losses", nargs="+", choices=["triplet", "cached-mnrl"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="core")
    parser.add_argument("--eval-split", choices=["dev", "test"], default="dev")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [
        {
            "job_index": index,
            "dataset": dataset,
            "model": model,
            "loss": loss,
            "seed": seed,
            "profile": args.profile,
            "eval_split": args.eval_split,
            "model_fits": len(PROFILES[args.profile]),
        }
        for index, (dataset, model, loss, seed) in enumerate(
            (dataset, model, loss, seed)
            for dataset in args.datasets
            for model in args.models
            for loss in args.losses
            for seed in args.seeds
        )
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "jobs": len(rows),
        "model_fits": sum(row["model_fits"] for row in rows),
        "datasets": args.datasets,
        "models": args.models,
        "losses": args.losses,
        "seeds": args.seeds,
        "profile": args.profile,
        "eval_split": args.eval_split,
        "output": str(out),
    }
    out.with_name(f"{out.stem}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
