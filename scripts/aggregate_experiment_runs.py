#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_METRICS = ["NDCG@10", "MRR@10", "MAP@100", "Recall@100"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate multi-seed CA-HNM runs with hierarchical bootstrap intervals."
    )
    parser.add_argument("--runs-root", default="runs/experiments/training")
    parser.add_argument("--baseline", default="CA-HNM-rank-matched")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--out-dir", default="runs/experiments/analysis")
    args = parser.parse_args()

    runs = _load_runs(Path(args.runs_root))
    if not runs:
        raise FileNotFoundError(f"no complete experiment runs with per-query metrics under {args.runs_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = _seed_metric_rows(runs, args.metrics)
    _write_csv(out_dir / "seed_metrics.csv", seed_rows)

    comparison_rows = _comparison_rows(
        runs,
        baseline=args.baseline,
        metrics=args.metrics,
        bootstrap_samples=args.bootstrap_samples,
        random_seed=args.seed,
    )
    _write_csv(out_dir / "comparisons.csv", comparison_rows)
    (out_dir / "comparisons.json").write_text(
        json.dumps(comparison_rows, indent=2, sort_keys=True), encoding="utf-8"
    )

    manifest = {
        "schema_version": 1,
        "runs_root": str(args.runs_root),
        "complete_runs": len(runs),
        "baseline": args.baseline,
        "metrics": args.metrics,
        "bootstrap_samples": args.bootstrap_samples,
        "random_seed": args.seed,
        "inference_unit": "training seed with queries nested within seed",
    }
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {len(comparison_rows)} comparisons from {len(runs)} complete runs to {out_dir}")


def _load_runs(root: Path) -> list[dict]:
    runs = []
    for metadata_path in sorted(root.rglob("experiment_run.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metrics_path = metadata_path.parent / "trained_per_query_metrics.csv"
        if metadata.get("status") != "complete" or not metrics_path.exists():
            continue
        per_query: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                strategy = row.pop("strategy")
                query_id = row.pop("query_id")
                per_query[strategy][query_id] = {
                    key: float(value) for key, value in row.items() if value not in {"", None}
                }
        runs.append({"metadata": metadata, "per_query": dict(per_query), "path": metadata_path.parent})
    return runs


def _seed_metric_rows(runs: list[dict], metrics: list[str]) -> list[dict]:
    rows = []
    for run in runs:
        metadata = run["metadata"]
        for strategy, query_rows in sorted(run["per_query"].items()):
            for metric in metrics:
                values = [row[metric] for row in query_rows.values() if metric in row]
                if not values:
                    continue
                rows.append(
                    {
                        "dataset": metadata["dataset"],
                        "model_key": metadata["model_key"],
                        "loss": metadata["loss"],
                        "eval_split": metadata["eval_split"],
                        "training_seed": metadata["training_seed"],
                        "strategy": strategy,
                        "metric": metric,
                        "n_queries": len(values),
                        "mean": statistics.fmean(values),
                    }
                )
    return rows


def _comparison_rows(
    runs: list[dict],
    baseline: str,
    metrics: list[str],
    bootstrap_samples: int,
    random_seed: int,
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs:
        metadata = run["metadata"]
        key = (metadata["dataset"], metadata["model_key"], metadata["loss"], metadata["eval_split"])
        grouped[key].append(run)

    rows = []
    for group_key, group_runs in sorted(grouped.items()):
        dataset, model_key, loss, eval_split = group_key
        strategies = sorted(set.intersection(*(set(run["per_query"]) for run in group_runs)))
        if baseline not in strategies:
            continue
        for strategy in strategies:
            if strategy == baseline:
                continue
            for metric in metrics:
                diffs_by_seed: dict[int, list[float]] = {}
                baseline_by_seed: dict[int, list[float]] = {}
                strategy_by_seed: dict[int, list[float]] = {}
                for run in group_runs:
                    seed = int(run["metadata"]["training_seed"])
                    base_rows = run["per_query"][baseline]
                    strategy_rows = run["per_query"][strategy]
                    query_ids = sorted(set(base_rows) & set(strategy_rows))
                    diffs = [
                        strategy_rows[query_id][metric] - base_rows[query_id][metric]
                        for query_id in query_ids
                        if metric in strategy_rows[query_id] and metric in base_rows[query_id]
                    ]
                    if diffs:
                        diffs_by_seed[seed] = diffs
                        baseline_by_seed[seed] = [
                            base_rows[query_id][metric]
                            for query_id in query_ids
                            if metric in strategy_rows[query_id] and metric in base_rows[query_id]
                        ]
                        strategy_by_seed[seed] = [
                            strategy_rows[query_id][metric]
                            for query_id in query_ids
                            if metric in strategy_rows[query_id] and metric in base_rows[query_id]
                        ]
                if not diffs_by_seed:
                    continue
                seed_means = [statistics.fmean(diffs_by_seed[seed]) for seed in sorted(diffs_by_seed)]
                observed = statistics.fmean(seed_means)
                rng = random.Random(_stable_seed(random_seed, group_key, strategy, metric))
                ci_low, ci_high = _hierarchical_bootstrap_ci(
                    diffs_by_seed, bootstrap_samples=bootstrap_samples, rng=rng
                )
                sd = statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0
                effect_dz = observed / sd if sd > 0 else None
                rows.append(
                    {
                        "dataset": dataset,
                        "model_key": model_key,
                        "loss": loss,
                        "eval_split": eval_split,
                        "baseline": baseline,
                        "strategy": strategy,
                        "metric": metric,
                        "n_seeds": len(seed_means),
                        "n_queries_min": min(len(values) for values in diffs_by_seed.values()),
                        "baseline_mean": statistics.fmean(
                            statistics.fmean(baseline_by_seed[seed]) for seed in baseline_by_seed
                        ),
                        "strategy_mean": statistics.fmean(
                            statistics.fmean(strategy_by_seed[seed]) for seed in strategy_by_seed
                        ),
                        "mean_diff": observed,
                        "seed_sd_diff": sd,
                        "paired_effect_dz": effect_dz,
                        "seed_win_rate": sum(value > 0 for value in seed_means) / len(seed_means),
                        "hierarchical_ci95_low": ci_low,
                        "hierarchical_ci95_high": ci_high,
                    }
                )
    return rows


def _hierarchical_bootstrap_ci(
    diffs_by_seed: dict[int, list[float]],
    bootstrap_samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    seeds = sorted(diffs_by_seed)
    if bootstrap_samples <= 0:
        value = statistics.fmean(statistics.fmean(diffs_by_seed[seed]) for seed in seeds)
        return value, value
    sampled_means = []
    for _ in range(bootstrap_samples):
        sampled_seed_means = []
        for _ in seeds:
            seed = seeds[rng.randrange(len(seeds))]
            query_diffs = diffs_by_seed[seed]
            sampled_seed_means.append(
                statistics.fmean(query_diffs[rng.randrange(len(query_diffs))] for _ in query_diffs)
            )
        sampled_means.append(statistics.fmean(sampled_seed_means))
    sampled_means.sort()
    return _quantile(sampled_means, 0.025), _quantile(sampled_means, 0.975)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    position = q * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _stable_seed(base: int, group_key: tuple, strategy: str, metric: str) -> int:
    text = "|".join([str(base), *(str(item) for item in group_key), strategy, metric])
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % (2**32)
    return value


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
