#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cahnm.evaluation import evaluate_run_per_query
from cahnm.io_utils import load_corpus, load_qrels, load_queries
from cahnm.retrievers import DenseRetriever
from cahnm.schemas import RetrievalRun


DEFAULT_METRICS = ("NDCG@10", "Recall@100", "MRR@10", "MAP@100")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute paired bootstrap CIs and randomization tests for trained retriever runs."
    )
    parser.add_argument("--corpus")
    parser.add_argument("--queries")
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--models-dir")
    parser.add_argument("--strategies", nargs="+")
    parser.add_argument("--baseline", default="DPR-Random")
    parser.add_argument("--per-query-input", help="Existing per-query metrics CSV. Skips model evaluation.")
    parser.add_argument("--eval-top-k", type=int, default=100)
    parser.add_argument("--dense-batch-size", type=int, default=16)
    parser.add_argument("--dense-device", help="cpu, cuda, cuda:0, etc.")
    parser.add_argument("--dense-max-seq-length", type=int)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--randomization-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qrels = load_qrels(args.qrels)

    if args.per_query_input:
        per_query = _read_per_query(Path(args.per_query_input))
    else:
        if not args.corpus or not args.queries or not args.models_dir or not args.strategies:
            parser.error("--corpus, --queries, --models-dir, and --strategies are required without --per-query-input")
        docs = load_corpus(args.corpus)
        queries = load_queries(args.queries)
        per_query = _compute_per_query(
            docs=docs,
            queries=queries,
            qrels=qrels,
            models_dir=Path(args.models_dir),
            strategies=args.strategies,
            eval_top_k=args.eval_top_k,
            dense_batch_size=args.dense_batch_size,
            dense_device=args.dense_device,
            dense_max_seq_length=args.dense_max_seq_length,
        )
        _write_per_query(out_dir / "per_query_metrics.csv", per_query)

    tests = _paired_tests(
        per_query,
        baseline=args.baseline,
        metrics=args.metrics,
        bootstrap_samples=args.bootstrap_samples,
        randomization_samples=args.randomization_samples,
        seed=args.seed,
    )
    _write_rows(out_dir / "paired_tests.csv", tests)
    _write_latex_table(out_dir / "paired_tests.tex", tests)
    print(f"wrote statistical tests to {out_dir}")


def _compute_per_query(
    docs,
    queries,
    qrels,
    models_dir: Path,
    strategies: list[str],
    eval_top_k: int,
    dense_batch_size: int,
    dense_device: str | None,
    dense_max_seq_length: int | None,
) -> dict[str, dict[str, dict[str, float]]]:
    ks = tuple(sorted({10, 100, eval_top_k}))
    per_query: dict[str, dict[str, dict[str, float]]] = {}
    for strategy in strategies:
        model_dir = models_dir / strategy
        if not model_dir.exists():
            print(f"warning: missing model directory for {strategy}: {model_dir}", file=sys.stderr)
            continue
        retriever = DenseRetriever(
            docs,
            model_name=str(model_dir),
            backend="sentence-transformers",
            batch_size=dense_batch_size,
            device=dense_device,
            max_seq_length=dense_max_seq_length,
        )
        run = retriever.run(queries, top_k=eval_top_k)
        run = RetrievalRun(f"trained:{strategy}", run.rankings)
        per_query[strategy] = evaluate_run_per_query(run, qrels, ks=ks)
    return per_query


def _paired_tests(
    per_query: dict[str, dict[str, dict[str, float]]],
    baseline: str,
    metrics: list[str],
    bootstrap_samples: int,
    randomization_samples: int,
    seed: int,
) -> list[dict[str, str | float | int]]:
    if baseline not in per_query:
        raise ValueError(f"Baseline {baseline!r} not found in per-query metrics")
    rng = random.Random(seed)
    rows: list[dict[str, str | float | int]] = []
    base_rows = per_query[baseline]
    for strategy, strategy_rows in sorted(per_query.items()):
        if strategy == baseline:
            continue
        query_ids = sorted(set(base_rows) & set(strategy_rows))
        for metric in metrics:
            diffs = [strategy_rows[qid].get(metric, 0.0) - base_rows[qid].get(metric, 0.0) for qid in query_ids]
            if not diffs:
                continue
            observed = sum(diffs) / len(diffs)
            ci_low, ci_high = _bootstrap_ci(diffs, bootstrap_samples, rng)
            p_value = _paired_randomization_pvalue(diffs, observed, randomization_samples, rng)
            rows.append(
                {
                    "baseline": baseline,
                    "strategy": strategy,
                    "metric": metric,
                    "n_queries": len(diffs),
                    "baseline_mean": _mean(base_rows[qid].get(metric, 0.0) for qid in query_ids),
                    "strategy_mean": _mean(strategy_rows[qid].get(metric, 0.0) for qid in query_ids),
                    "mean_diff": observed,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "randomization_p": p_value,
                    "significant_0_05": "yes" if p_value < 0.05 and not (ci_low <= 0.0 <= ci_high) else "no",
                }
            )
    return rows


def _bootstrap_ci(diffs: list[float], samples: int, rng: random.Random) -> tuple[float, float]:
    if samples <= 0:
        mean = _mean(diffs)
        return mean, mean
    n = len(diffs)
    means = []
    for _ in range(samples):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return _quantile(means, 0.025), _quantile(means, 0.975)


def _paired_randomization_pvalue(diffs: list[float], observed: float, samples: int, rng: random.Random) -> float:
    if samples <= 0:
        return 1.0
    count = 0
    n = len(diffs)
    threshold = abs(observed)
    for _ in range(samples):
        permuted = sum(diff if rng.random() < 0.5 else -diff for diff in diffs) / n
        if abs(permuted) >= threshold:
            count += 1
    return (count + 1) / (samples + 1)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    pos = q * (len(values) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    frac = pos - lower
    return values[lower] * (1 - frac) + values[upper] * frac


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _read_per_query(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            strategy = row.pop("strategy")
            query_id = row.pop("query_id")
            result.setdefault(strategy, {})[query_id] = {
                key: float(value) for key, value in row.items() if value not in {"", None}
            }
    return result


def _write_per_query(path: Path, per_query: dict[str, dict[str, dict[str, float]]]) -> None:
    metric_names = sorted({metric for rows in per_query.values() for row in rows.values() for metric in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["strategy", "query_id", *metric_names])
        writer.writeheader()
        for strategy, rows in sorted(per_query.items()):
            for query_id, metrics in sorted(rows.items()):
                writer.writerow({"strategy": strategy, "query_id": query_id, **metrics})


def _write_rows(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_latex_table(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Method & Metric & $\\Delta$ & 95\\% CI low & 95\\% CI high & $p$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['strategy']} & {row['metric']} & "
            f"{float(row['mean_diff']):.4f} & {float(row['ci95_low']):.4f} & "
            f"{float(row['ci95_high']):.4f} & {float(row['randomization_p']):.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
