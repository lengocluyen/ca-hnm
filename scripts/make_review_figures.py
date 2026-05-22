#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


COLORS = {
    "DPR-Random": "#64748b",
    "CA-HNM-mixed": "#92400e",
    "CA-HNM-v2-mixed": "#b45309",
    "HardNeg": "#16a34a",
    "Positive": "#dc2626",
    "Ambiguous": "#f59e0b",
    "EasyNeg": "#94a3b8",
    "grid": "#e2e8f0",
    "text": "#111827",
}

LABELS = {
    "DPR-Random": "DPR-Random",
    "CA-HNM-mixed": "CA-HNM-Mix",
    "CA-HNM-v2-mixed": "CA-HNM-RA-Mix",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create review-response figures that are not table duplicates.")
    parser.add_argument("--old-root", default="runs")
    parser.add_argument("--out", default="runs/paper_figures_review")
    args = parser.parse_args()

    root = Path(args.old_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _write_svg(out / "fig_significance_ci.svg", _significance_ci_figure(root))
    _write_svg(out / "fig_leakage_stress.svg", _leakage_stress_figure(root))
    _write_svg(out / "fig_llm_label_distribution.svg", _llm_label_distribution_figure(root))
    (out / "README.md").write_text(_readme(out), encoding="utf-8")
    print(f"wrote review figures to {out}")


def _significance_ci_figure(root: Path) -> str:
    stats = {
        "MoocCubeX": _read_paired_tests(root / "old_main_stats" / "mooccubex" / "paired_tests.csv"),
        "Course-Skill Atlas": _read_paired_tests(root / "old_main_stats" / "course_skill_atlas" / "paired_tests.csv"),
    }
    methods = ["CA-HNM-mixed", "CA-HNM-v2-mixed"]
    metrics = ["NDCG@10", "MRR@10", "MAP@100"]
    width, height = 1040, 600
    parts = [_title("Paired improvement over DPR-Random with 95% bootstrap CI")]
    panel_w, panel_h = 460, 230
    for col, (dataset, rows) in enumerate(stats.items()):
        x0 = 70 + col * 500
        y0 = 78
        parts.extend(_ci_panel(dataset, rows, methods, metrics, x0, y0, panel_w, panel_h))
    parts.append(_legend(700, 540))
    return _svg(width, height, parts)


def _ci_panel(title: str, rows: dict[tuple[str, str], dict], methods: list[str], metrics: list[str], x: int, y: int, w: int, h: int) -> list[str]:
    parts = [f"<text x='{x}' y='{y}' font-size='18' font-weight='700'>{_esc(title)}</text>"]
    left = x + 120
    right = x + w - 20
    top = y + 35
    row_h = 28
    min_v, max_v = -0.25, 0.25
    zero_x = _scale(0.0, min_v, max_v, left, right)
    parts.append(f"<line x1='{left}' y1='{top - 12}' x2='{right}' y2='{top - 12}' stroke='{COLORS['grid']}'/>")
    parts.append(f"<line x1='{zero_x}' y1='{top - 20}' x2='{zero_x}' y2='{top + len(methods)*len(metrics)*row_h + 8}' stroke='#111827' stroke-dasharray='4 4'/>")
    for tick in [-0.2, -0.1, 0.0, 0.1, 0.2]:
        tx = _scale(tick, min_v, max_v, left, right)
        parts.append(f"<line x1='{tx}' y1='{top - 18}' x2='{tx}' y2='{top + len(methods)*len(metrics)*row_h + 5}' stroke='{COLORS['grid']}'/>")
        parts.append(f"<text x='{tx - 12}' y='{top + len(methods)*len(metrics)*row_h + 24}' font-size='11' fill='#64748b'>{tick:.1f}</text>")
    idx = 0
    for metric in metrics:
        for method in methods:
            row = rows.get((method, metric))
            if not row:
                continue
            cy = top + idx * row_h
            mean = float(row["mean_diff"])
            lo = float(row["ci95_low"])
            hi = float(row["ci95_high"])
            p = float(row["randomization_p"])
            x_mean = _scale(mean, min_v, max_v, left, right)
            x_lo = _scale(lo, min_v, max_v, left, right)
            x_hi = _scale(hi, min_v, max_v, left, right)
            color = COLORS[method]
            label = metric.replace("@10", "").replace("@100", "")
            if method == methods[0]:
                parts.append(f"<text x='{x}' y='{cy + 4}' font-size='12'>{_esc(label)}</text>")
            parts.append(f"<line x1='{x_lo}' y1='{cy}' x2='{x_hi}' y2='{cy}' stroke='{color}' stroke-width='3'/>")
            parts.append(f"<circle cx='{x_mean}' cy='{cy}' r='5' fill='{color}'/>")
            star = "*" if p < 0.05 else ""
            parts.append(f"<text x='{x_hi + 6}' y='{cy + 4}' font-size='11'>{mean:+.3f}{star}</text>")
            idx += 1
    parts.append(f"<text x='{left + 72}' y='{top + len(methods)*len(metrics)*row_h + 48}' font-size='12'>Mean paired difference vs DPR-Random</text>")
    return parts


def _leakage_stress_figure(root: Path) -> str:
    originals = {
        "MoocCubeX": root / "e1_mooccubex_v2_heuristic_bge_base_top100_neg4_mnrl" / "mooccubex_heuristic_baai_bge_base_en_v1_5",
        "Course-Skill Atlas": root / "e1_course_skill_atlas_v2_heuristic_bge_base_top100_neg4_mnrl" / "course_skill_atlas_heuristic_baai_bge_base_en_v1_5",
    }
    masked = {
        "MoocCubeX": root / "leakage_mooccubex_oldcfg_query_target_masked",
        "Course-Skill Atlas": root / "leakage_course_skill_atlas_oldcfg_query_target_masked",
    }
    methods = ["DPR-Random", "CA-HNM-mixed", "CA-HNM-v2-mixed"]
    width, height = 1040, 520
    parts = [_title("Leakage-control stress test: original vs target-label masked")]
    for idx, dataset in enumerate(originals):
        x = 70 + idx * 500
        y = 82
        orig = _read_retrieval(originals[dataset] / "trained_retrieval_metrics.json")
        mask = _read_retrieval(masked[dataset] / "trained_retrieval_metrics.json")
        parts.extend(_paired_bar_panel(dataset, orig, mask, methods, x, y, 430, 330))
    parts.append(_small_note(70, 472, "Masking removes direct query target labels and strips ontology aliases; the figure is a stress test, not a main retrieval setting."))
    return _svg(width, height, parts)


def _paired_bar_panel(title: str, original: dict[str, dict], masked: dict[str, dict], methods: list[str], x: int, y: int, w: int, h: int) -> list[str]:
    parts = [f"<text x='{x}' y='{y}' font-size='18' font-weight='700'>{_esc(title)}</text>"]
    label_w = 120
    plot_w = w - label_w - 35
    max_v = max([_metric(original, m, "NDCG@10") for m in methods] + [_metric(masked, m, "NDCG@10") for m in methods] + [0.1])
    max_v = min(1.0, max_v * 1.15)
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        tx = x + label_w + int(plot_w * tick)
        parts.append(f"<line x1='{tx}' y1='{y + 26}' x2='{tx}' y2='{y + h}' stroke='{COLORS['grid']}'/>")
        parts.append(f"<text x='{tx - 9}' y='{y + h + 18}' font-size='11' fill='#64748b'>{tick*max_v:.2f}</text>")
    for i, method in enumerate(methods):
        row_y = y + 48 + i * 82
        parts.append(f"<text x='{x}' y='{row_y + 28}' font-size='12'>{_esc(LABELS[method])}</text>")
        for j, (name, source, opacity) in enumerate([("original", original, 1.0), ("masked", masked, 0.45)]):
            value = _metric(source, method, "NDCG@10")
            bar_w = int(plot_w * value / max_v)
            by = row_y + j * 25
            parts.append(f"<rect x='{x + label_w}' y='{by}' width='{bar_w}' height='18' fill='{COLORS[method]}' opacity='{opacity}'/>")
            parts.append(f"<text x='{x + label_w + bar_w + 5}' y='{by + 14}' font-size='11'>{value:.3f}</text>")
    parts.append(f"<text x='{x + label_w}' y='{y + h + 42}' font-size='12'>NDCG@10</text>")
    parts.append(f"<rect x='{x + 235}' y='{y + h + 32}' width='12' height='12' fill='#64748b' opacity='1'/>")
    parts.append(f"<text x='{x + 253}' y='{y + h + 43}' font-size='11'>original</text>")
    parts.append(f"<rect x='{x + 315}' y='{y + h + 32}' width='12' height='12' fill='#64748b' opacity='0.45'/>")
    parts.append(f"<text x='{x + 333}' y='{y + h + 43}' font-size='11'>masked</text>")
    return parts


def _llm_label_distribution_figure(root: Path) -> str:
    validations = {
        "MoocCubeX": root
        / "e1_mooccubex_v2_heuristic_bge_base_top100_neg4_mnrl"
        / "mooccubex_heuristic_baai_bge_base_en_v1_5"
        / "CA-HNM-mixed.gpt-oss-20b_validation_summary.json",
        "Course-Skill Atlas": root
        / "e1_course_skill_atlas_v2_heuristic_bge_base_top100_neg4_mnrl"
        / "course_skill_atlas_heuristic_baai_bge_base_en_v1_5"
        / "CA-HNM-mixed.gpt-oss-20b_validation_summary.json",
    }
    width, height = 880, 330
    parts = [_title("LLM-assisted validation label distribution for CA-HNM-Mix")]
    x, y = 100, 95
    bar_w = 620
    labels = ["HardNeg", "Positive", "Ambiguous", "EasyNeg"]
    for i, (dataset, path) in enumerate(validations.items()):
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        dist = {key: int(value) for key, value in data.get("label_distribution", {}).items()}
        total = max(1, sum(dist.values()))
        row_y = y + i * 78
        parts.append(f"<text x='{x - 10}' y='{row_y + 18}' text-anchor='end' font-size='13' font-weight='700'>{_esc(dataset)}</text>")
        cursor = x
        for label in labels:
            count = dist.get(label, 0)
            seg_w = int(bar_w * count / total)
            if seg_w:
                parts.append(f"<rect x='{cursor}' y='{row_y}' width='{seg_w}' height='26' fill='{COLORS[label]}'/>")
            cursor += seg_w
        valid = float(data.get("valid_hard_rate", 0.0))
        parts.append(f"<text x='{x + bar_w + 12}' y='{row_y + 19}' font-size='12'>valid-hard={valid:.3f}</text>")
    lx = x
    for label in labels:
        parts.append(f"<rect x='{lx}' y='{260}' width='13' height='13' fill='{COLORS[label]}'/>")
        parts.append(f"<text x='{lx + 18}' y='{271}' font-size='12'>{_esc(label)}</text>")
        lx += 118
    return _svg(width, height, parts)


def _read_paired_tests(path: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row["strategy"], row["metric"])] = row
    return rows


def _read_retrieval(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row.get("strategy", row.get("run", "")): row for row in payload if isinstance(row, dict)}


def _metric(rows: dict[str, dict], method: str, metric: str) -> float:
    try:
        return float(rows.get(method, {}).get(metric, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _scale(value: float, min_v: float, max_v: float, left: int, right: int) -> int:
    value = max(min_v, min(max_v, value))
    return left + int((value - min_v) * (right - left) / (max_v - min_v))


def _title(text: str) -> str:
    return f"<text x='40' y='34' font-size='22' font-weight='700'>{_esc(text)}</text>"


def _legend(x: int, y: int) -> str:
    parts = []
    for i, method in enumerate(["CA-HNM-mixed", "CA-HNM-v2-mixed"]):
        lx = x + i * 150
        parts.append(f"<circle cx='{lx}' cy='{y}' r='6' fill='{COLORS[method]}'/>")
        parts.append(f"<text x='{lx + 12}' y='{y + 4}' font-size='12'>{_esc(LABELS[method])}</text>")
    parts.append(f"<text x='{x}' y='{y + 32}' font-size='11' fill='#64748b'>Asterisk indicates paired randomization p &lt; 0.05.</text>")
    return "".join(parts)


def _small_note(x: int, y: int, text: str) -> str:
    return f"<text x='{x}' y='{y}' font-size='12' fill='#64748b'>{_esc(text)}</text>"


def _svg(width: int, height: int, parts: list[str]) -> str:
    style = "<style>text{font-family:Arial,Helvetica,sans-serif;fill:#111827}</style>"
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>{style}{''.join(parts)}</svg>\n"
    )


def _write_svg(path: Path, svg: str) -> None:
    path.write_text(svg, encoding="utf-8")


def _esc(text: str) -> str:
    return html.escape(str(text))


def _readme(out: Path) -> str:
    return f"""# Review-response figures

Generated in `{out}`.

- `fig_significance_ci.svg`: statistical uncertainty not shown in metric tables.
- `fig_leakage_stress.svg`: leakage-control stress test, suitable for discussion or appendix.
- `fig_llm_label_distribution.svg`: LLM validation label composition beyond the valid-hard summary table.
"""


if __name__ == "__main__":
    main()
