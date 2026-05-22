from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Iterable


def analyze_run(run_dir: str | Path, out_dir: str | Path | None = None) -> Path:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    out_dir = Path(out_dir) if out_dir else run_dir / "analysis"
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    negative_quality = _read_csv(run_dir / "negative_quality.csv")
    training_summary = _read_csv(run_dir / "training_summary.csv")
    retrieval_metrics = _read_json_rows(run_dir / "retrieval_metrics.json")
    trained_metrics = _read_json_rows(run_dir / "trained_retrieval_metrics.json")
    validations = _read_validation_summaries(run_dir)
    artifact_counts = _artifact_counts(run_dir)

    negative_quality_ranked = _rank_negative_quality(negative_quality)
    retrieval_flat = _flatten_metric_rows(retrieval_metrics)
    trained_flat = _flatten_metric_rows(trained_metrics)

    _write_csv(tables_dir / "negative_quality_ranked.csv", negative_quality_ranked)
    _write_csv(tables_dir / "retrieval_metrics.csv", retrieval_flat)
    _write_csv(tables_dir / "trained_retrieval_metrics.csv", trained_flat)
    _write_csv(tables_dir / "training_summary.csv", training_summary)
    _write_csv(tables_dir / "validation_summary.csv", validations)
    _write_csv(tables_dir / "artifact_counts.csv", artifact_counts)

    summary = _make_markdown_summary(
        run_dir=run_dir,
        negative_quality=negative_quality_ranked,
        training_summary=training_summary,
        retrieval_metrics=retrieval_flat,
        trained_metrics=trained_flat,
        validations=validations,
        artifact_counts=artifact_counts,
    )
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    (out_dir / "index.html").write_text(
        _make_html_report(
            run_dir=run_dir,
            negative_quality=negative_quality_ranked,
            training_summary=training_summary,
            retrieval_metrics=retrieval_flat,
            trained_metrics=trained_flat,
            validations=validations,
            artifact_counts=artifact_counts,
        ),
        encoding="utf-8",
    )
    return out_dir


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _read_validation_summaries(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(run_dir.glob("*.json")):
        lower = path.name.lower()
        if "validation" not in lower or "summary" not in lower:
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            row = dict(row)
            row["file"] = path.name
            rows.append(row)
    return rows


def _artifact_counts(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(run_dir.glob("*.negatives.jsonl")):
        strategy = path.name.removesuffix(".negatives.jsonl")
        rows.append({"strategy": strategy, "artifact": "negatives", "rows": _count_lines(path), "file": path.name})
    for path in sorted(run_dir.glob("*.triplets.jsonl")):
        strategy = path.name.removesuffix(".triplets.jsonl")
        rows.append({"strategy": strategy, "artifact": "triplets", "rows": _count_lines(path), "file": path.name})
    return rows


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _rank_negative_quality(rows: list[dict]) -> list[dict]:
    enriched = []
    for row in rows:
        item = dict(row)
        valid = _float(item.get("valid_hard_rate"))
        false_negative = _float(item.get("false_negative_rate"))
        leakage = _float(item.get("target_leakage_rate"))
        item["quality_score"] = f"{valid - false_negative - leakage:.6f}"
        enriched.append(item)
    enriched.sort(key=lambda item: _float(item.get("quality_score")), reverse=True)
    return enriched


def _flatten_metric_rows(rows: list[dict]) -> list[dict]:
    flattened: list[dict] = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            if isinstance(value, (dict, list)):
                flat[key] = json.dumps(value, sort_keys=True)
            else:
                flat[key] = value
        flattened.append(flat)
    return flattened


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _make_markdown_summary(
    run_dir: Path,
    negative_quality: list[dict],
    training_summary: list[dict],
    retrieval_metrics: list[dict],
    trained_metrics: list[dict],
    validations: list[dict],
    artifact_counts: list[dict],
) -> str:
    lines = [f"# CA-HNM Run Analysis: `{run_dir}`", ""]
    lines.extend(_section_table("Artifact Counts", artifact_counts[:20], ["strategy", "artifact", "rows"]))
    lines.extend(_section_table("Negative Quality", negative_quality[:20], ["strategy", "negatives", "valid_hard_rate", "false_negative_rate", "target_leakage_rate", "quality_score"]))
    lines.extend(_section_table("Training Summary", training_summary[:20], ["strategy", "model_name", "triplets", "epochs", "batch_size", "duration_sec"]))
    lines.extend(_section_table("Zero-Shot Retrieval", retrieval_metrics[:20], _metric_columns(retrieval_metrics)))
    lines.extend(_section_table("Trained Retrieval", trained_metrics[:20], _metric_columns(trained_metrics)))
    lines.extend(_section_table("LLM Validation", validations[:20], ["file", "judge", "llm_model", "evaluated", "valid_hard_rate", "false_negative_rate"]))
    lines.append("## Reading Guide")
    lines.append("")
    lines.append("- `valid_hard_rate` should be high: the negative is semantically close but invalid.")
    lines.append("- `false_negative_rate` should be near zero: true positives should not be mined as negatives.")
    lines.append("- `target_leakage_rate` should be low: candidates should not directly satisfy the target concept.")
    lines.append("- Main retrieval evidence comes from `trained_retrieval_metrics.json`, not only `retrieval_metrics.json`.")
    lines.append("")
    return "\n".join(lines)


def _section_table(title: str, rows: list[dict], columns: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows or not columns:
        lines.extend(["No data found.", ""])
        return lines
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_md(row.get(col, "")) for col in columns) + " |")
    lines.append("")
    return lines


def _metric_columns(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    preferred = [
        "run",
        "strategy",
        "NDCG@10",
        "Recall@10",
        "Recall@100",
        "MRR@10",
        "MAP@100",
        "ConstraintViolation@10",
    ]
    present = {key for row in rows for key in row}
    return [key for key in preferred if key in present] or sorted(present)[:8]


def _make_html_report(
    run_dir: Path,
    negative_quality: list[dict],
    training_summary: list[dict],
    retrieval_metrics: list[dict],
    trained_metrics: list[dict],
    validations: list[dict],
    artifact_counts: list[dict],
) -> str:
    sections = [
        _html_table("Artifact Counts", artifact_counts, ["strategy", "artifact", "rows"]),
        _bar_chart("Negative Valid Hard Rate", negative_quality, "strategy", "valid_hard_rate"),
        _bar_chart("Negative False Negative Rate", negative_quality, "strategy", "false_negative_rate"),
        _bar_chart("Trained NDCG@10", trained_metrics, "strategy", "NDCG@10"),
        _bar_chart("Trained Recall@100", trained_metrics, "strategy", "Recall@100"),
        _bar_chart("Training Triplets", training_summary, "strategy", "triplets"),
        _bar_chart("LLM Validation Valid Hard Rate", validations, "file", "valid_hard_rate"),
        _html_table("Negative Quality", negative_quality, ["strategy", "negatives", "valid_hard_rate", "false_negative_rate", "target_leakage_rate", "quality_score"]),
        _html_table("Trained Retrieval", trained_metrics, _metric_columns(trained_metrics)),
        _html_table("Validation", validations, ["file", "judge", "llm_model", "evaluated", "valid_hard_rate", "false_negative_rate"]),
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CA-HNM Run Analysis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ margin: 0 0 12px; }}
    section {{ margin: 28px 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d6d9de; padding: 7px 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    svg {{ max-width: 100%; height: auto; border: 1px solid #e5e7eb; background: #fff; }}
    .muted {{ color: #64748b; }}
  </style>
</head>
<body>
  <h1>CA-HNM Run Analysis</h1>
  <p class="muted">{html.escape(str(run_dir))}</p>
  {''.join(sections)}
</body>
</html>
"""


def _bar_chart(title: str, rows: list[dict], label_key: str, value_key: str) -> str:
    items = [(str(row.get(label_key, "")), _float(row.get(value_key))) for row in rows if row.get(label_key) not in {None, ""}]
    items = [(label, value) for label, value in items if value is not None]
    if not items:
        return f"<section><h2>{html.escape(title)}</h2><p class='muted'>No data.</p></section>"
    items = items[:20]
    max_value = max(value for _, value in items) or 1.0
    width = 920
    row_h = 30
    left = 220
    right = 80
    height = 50 + row_h * len(items)
    bars = []
    for idx, (label, value) in enumerate(items):
        y = 35 + idx * row_h
        bar_w = int((width - left - right) * value / max_value)
        bars.append(f"<text x='8' y='{y + 16}' font-size='12'>{html.escape(label[:34])}</text>")
        bars.append(f"<rect x='{left}' y='{y}' width='{bar_w}' height='18' fill='#2563eb'></rect>")
        bars.append(f"<text x='{left + bar_w + 6}' y='{y + 14}' font-size='12'>{value:.4g}</text>")
    return f"<section><h2>{html.escape(title)}</h2><svg width='{width}' height='{height}' role='img'>{''.join(bars)}</svg></section>"


def _html_table(title: str, rows: list[dict], columns: list[str]) -> str:
    if not rows or not columns:
        return f"<section><h2>{html.escape(title)}</h2><p class='muted'>No data.</p></section>"
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = []
    for row in rows[:50]:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in columns) + "</tr>")
    return f"<section><h2>{html.escape(title)}</h2><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></section>"


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _md(value) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize and visualize CA-HNM run artifacts.")
    parser.add_argument("--run", required=True, help="Run directory, e.g. runs/mooccubex_full_bge_base_gpu7.")
    parser.add_argument("--out", help="Output analysis directory. Defaults to <run>/analysis.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    out_dir = analyze_run(args.run, args.out)
    print(f"wrote analysis report to {out_dir}")


if __name__ == "__main__":
    main()
