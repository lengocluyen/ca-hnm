#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_MOOCCUBEX_RUN = (
    "runs/e1_mooccubex_v2_heuristic_bge_base_top100_neg4_mnrl/"
    "mooccubex_heuristic_baai_bge_base_en_v1_5"
)
DEFAULT_CSA_RUN = (
    "runs/e1_course_skill_atlas_v2_heuristic_bge_base_top100_neg4_mnrl/"
    "course_skill_atlas_heuristic_baai_bge_base_en_v1_5"
)
DEFAULT_MOOCCUBEX_ONTOLOGY = "data/processed/mooccubex_ontology_full.json"
DEFAULT_CSA_ONTOLOGY = "data/processed/course_skill_atlas_ontology.json"

METHOD_LABELS = {
    "DPR-Random": "DPR-Random",
    "ANCE": "ANCE",
    "ADORE": "ADORE",
    "RocketQA-Denoised": "RocketQA",
    "TAS-Balanced": "TAS",
    "GPL-Pseudo": "GPL",
    "SyNeg": "SyNeg",
    "CA-HNM-full": "CA-HNM-C",
    "CA-HNM-mixed": "CA-HNM-Mix",
    "CA-HNM-v2": "CA-HNM-RA",
    "CA-HNM-v2-mixed": "CA-HNM-RA-Mix",
    "CA-HNM-no-ontology": "No Ont.",
    "CA-HNM-no-reasoning": "No Reas.",
    "CA-HNM-no-fn-filter": "No FN Filter",
    "CA-HNM-prereq-only": "Prereq Only",
    "CA-HNM-sibling-only": "Sibling Only",
}

MAIN_METHODS = [
    "DPR-Random",
    "ANCE",
    "TAS-Balanced",
    "SyNeg",
    "CA-HNM-mixed",
    "CA-HNM-v2-mixed",
]

ABLATION_METHODS = [
    "CA-HNM-full",
    "CA-HNM-mixed",
    "CA-HNM-v2",
    "CA-HNM-v2-mixed",
    "CA-HNM-no-ontology",
    "CA-HNM-no-reasoning",
    "CA-HNM-no-fn-filter",
]

COLORS = {
    "baseline": "#64748b",
    "dense": "#2563eb",
    "ours": "#b45309",
    "ours_dark": "#92400e",
    "ablation": "#0891b2",
    "quality": "#16a34a",
    "danger": "#dc2626",
    "muted": "#94a3b8",
    "grid": "#e2e8f0",
    "text": "#111827",
}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create paper-grade SVG figures for CA-HNM runs.")
    parser.add_argument("--mooccubex-run", default=DEFAULT_MOOCCUBEX_RUN)
    parser.add_argument("--course-skill-atlas-run", default=DEFAULT_CSA_RUN)
    parser.add_argument("--mooccubex-ontology", default=DEFAULT_MOOCCUBEX_ONTOLOGY)
    parser.add_argument("--course-skill-atlas-ontology", default=DEFAULT_CSA_ONTOLOGY)
    parser.add_argument("--out", default="runs/paper_figures_base")
    parser.add_argument(
        "--main-methods",
        nargs="+",
        default=MAIN_METHODS,
        help="Methods to include in main comparison figures. Defaults to grouped representative baselines.",
    )
    parser.add_argument(
        "--ablation-methods",
        nargs="+",
        default=ABLATION_METHODS,
        help="Methods to include in CA-HNM ablation figures.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    datasets = [
        DatasetBundle("MoocCubeX", Path(args.mooccubex_run), Path(args.mooccubex_ontology)),
        DatasetBundle("Course-Skill Atlas", Path(args.course_skill_atlas_run), Path(args.course_skill_atlas_ontology)),
    ]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bundle in datasets:
        bundle.load()

    _write_svg(out_dir / "fig_main_ndcg.svg", _main_metric_figure(datasets, "NDCG@10", args.main_methods))
    _write_svg(out_dir / "fig_main_mrr.svg", _main_metric_figure(datasets, "MRR@10", args.main_methods))
    _write_svg(out_dir / "fig_quality_tradeoff.svg", _quality_tradeoff_figure(datasets, args.main_methods))
    _write_svg(out_dir / "fig_ablation_ndcg.svg", _ablation_metric_figure(datasets, "NDCG@10", args.ablation_methods))
    _write_svg(out_dir / "fig_ablation_quality.svg", _ablation_quality_figure(datasets, args.ablation_methods))
    _write_svg(out_dir / "fig_training_cost.svg", _training_cost_figure(datasets, args.main_methods))
    _write_svg(out_dir / "fig_ontology_stats.svg", _ontology_stats_figure(datasets))
    _write_svg(out_dir / "fig_ontology_context.svg", _ontology_context_schematic())
    _write_csv_summary(out_dir / "paper_metrics_summary.csv", datasets)
    (out_dir / "README.md").write_text(_readme(out_dir), encoding="utf-8")
    print(f"wrote paper figures to {out_dir}")


class DatasetBundle:
    def __init__(self, name: str, run_dir: Path, ontology_path: Path) -> None:
        self.name = name
        self.run_dir = run_dir
        self.ontology_path = ontology_path
        self.retrieval: dict[str, dict] = {}
        self.quality: dict[str, dict] = {}
        self.training: dict[str, dict] = {}
        self.ontology: dict = {}

    def load(self) -> None:
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Missing run directory: {self.run_dir}")
        self.retrieval = {row.get("strategy", ""): row for row in _read_json_rows(self.run_dir / "trained_retrieval_metrics.json")}
        self.quality = {row.get("strategy", ""): row for row in _read_csv(self.run_dir / "negative_quality.csv")}
        self.training = {row.get("strategy", ""): row for row in _read_csv(self.run_dir / "training_summary.csv")}
        if self.ontology_path.exists():
            self.ontology = json.loads(self.ontology_path.read_text(encoding="utf-8"))


def _read_json_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_svg(path: Path, svg: str) -> None:
    path.write_text(svg, encoding="utf-8")


def _write_csv_summary(path: Path, datasets: list[DatasetBundle]) -> None:
    fieldnames = [
        "dataset",
        "method",
        "paper_method",
        "NDCG@10",
        "Recall@100",
        "MRR@10",
        "MAP@100",
        "ConstraintViolation@10",
        "valid_hard_rate",
        "false_negative_rate",
        "target_leakage_rate",
        "triplets",
        "duration_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for bundle in datasets:
            methods = sorted(set(bundle.retrieval) | set(bundle.quality) | set(bundle.training))
            for method in methods:
                retrieval = bundle.retrieval.get(method, {})
                quality = bundle.quality.get(method, {})
                training = bundle.training.get(method, {})
                writer.writerow(
                    {
                        "dataset": bundle.name,
                        "method": method,
                        "paper_method": _label(method),
                        "NDCG@10": retrieval.get("NDCG@10", ""),
                        "Recall@100": retrieval.get("Recall@100", ""),
                        "MRR@10": retrieval.get("MRR@10", ""),
                        "MAP@100": retrieval.get("MAP@100", ""),
                        "ConstraintViolation@10": retrieval.get("ConstraintViolation@10", ""),
                        "valid_hard_rate": quality.get("valid_hard_rate", ""),
                        "false_negative_rate": quality.get("false_negative_rate", ""),
                        "target_leakage_rate": quality.get("target_leakage_rate", ""),
                        "triplets": training.get("triplets", ""),
                        "duration_sec": training.get("duration_sec", ""),
                    }
                )


def _main_metric_figure(datasets: list[DatasetBundle], metric: str, methods: list[str]) -> str:
    panels = []
    width = 1000
    panel_h = 290
    for idx, bundle in enumerate(datasets):
        y0 = 70 + idx * panel_h
        values = [(method, _metric(bundle, method, metric)) for method in methods]
        panels.append(_horizontal_bars(bundle.name, values, x=170, y=y0, width=760, height=220, max_value=1.0))
    return _svg(
        width,
        70 + panel_h * len(datasets),
        [
            _title(f"Main retrieval effectiveness: {metric}"),
            *_legend(725, 22),
            *panels,
        ],
    )


def _quality_tradeoff_figure(datasets: list[DatasetBundle], methods: list[str]) -> str:
    width = 1040
    height = 470
    parts = [_title("Retrieval effectiveness vs. negative quality")]
    for idx, bundle in enumerate(datasets):
        x0 = 70 + idx * 510
        y0 = 80
        parts.append(_scatter_panel(bundle, x0, y0, 430, 320, methods))
    return _svg(width, height, parts)


def _ablation_metric_figure(datasets: list[DatasetBundle], metric: str, methods: list[str]) -> str:
    panels = []
    width = 1000
    panel_h = 290
    for idx, bundle in enumerate(datasets):
        y0 = 70 + idx * panel_h
        values = [(method, _metric(bundle, method, metric)) for method in methods]
        panels.append(_horizontal_bars(bundle.name, values, x=170, y=y0, width=760, height=220, max_value=1.0))
    return _svg(width, 70 + panel_h * len(datasets), [_title(f"CA-HNM ablation: {metric}"), *panels])


def _ablation_quality_figure(datasets: list[DatasetBundle], methods: list[str]) -> str:
    panels = []
    width = 1000
    panel_h = 290
    for idx, bundle in enumerate(datasets):
        y0 = 70 + idx * panel_h
        values = [(method, _quality(bundle, method, "valid_hard_rate")) for method in methods]
        panels.append(_horizontal_bars(bundle.name, values, x=170, y=y0, width=760, height=220, max_value=1.0, color_key="quality"))
    return _svg(width, 70 + panel_h * len(datasets), [_title("CA-HNM ablation: valid hard negative rate"), *panels])


def _training_cost_figure(datasets: list[DatasetBundle], main_methods: list[str]) -> str:
    width = 1000
    panel_h = 290
    parts = [_title("Training cost by negative mining strategy")]
    methods = list(dict.fromkeys([*main_methods, "CA-HNM-full", "CA-HNM-v2"]))
    for idx, bundle in enumerate(datasets):
        y0 = 70 + idx * panel_h
        values = [(method, _float(bundle.training.get(method, {}).get("duration_sec")) / 60.0) for method in methods]
        panels = _horizontal_bars(bundle.name, values, x=170, y=y0, width=760, height=220, max_value=None, suffix=" min")
        parts.append(panels)
    return _svg(width, 70 + panel_h * len(datasets), parts)


def _ontology_stats_figure(datasets: list[DatasetBundle]) -> str:
    width = 1040
    height = 560
    parts = [_title("Ontology structure used for constraint-aware mining")]
    for idx, bundle in enumerate(datasets):
        x = 60 + idx * 510
        y = 80
        nodes = bundle.ontology.get("nodes", []) if isinstance(bundle.ontology, dict) else []
        relations = bundle.ontology.get("relations", []) if isinstance(bundle.ontology, dict) else []
        rel_types = Counter(str(rel.get("type", "unknown")) for rel in relations if isinstance(rel, dict))
        parts.append(f"<text x='{x}' y='{y}' font-size='18' font-weight='700'>{_esc(bundle.name)}</text>")
        parts.append(f"<text x='{x}' y='{y + 30}' font-size='13' fill='{COLORS['text']}'>Nodes: {len(nodes):,}</text>")
        parts.append(f"<text x='{x}' y='{y + 50}' font-size='13' fill='{COLORS['text']}'>Relations: {len(relations):,}</text>")
        values = [(name, count) for name, count in rel_types.most_common(8)]
        max_value = max([count for _, count in values], default=1)
        bar_x = x + 120
        for i, (name, count) in enumerate(values):
            row_y = y + 85 + i * 32
            bar_w = int(300 * count / max_value)
            parts.append(f"<text x='{x}' y='{row_y + 14}' font-size='12'>{_esc(name[:18])}</text>")
            parts.append(f"<rect x='{bar_x}' y='{row_y}' width='{bar_w}' height='18' fill='{COLORS['ablation']}'></rect>")
            parts.append(f"<text x='{bar_x + bar_w + 6}' y='{row_y + 14}' font-size='12'>{count:,}</text>")
    return _svg(width, height, parts)


def _ontology_context_schematic() -> str:
    width = 980
    height = 420
    cx = 490
    cy = 205
    nodes = [
        ("Broader concept", 250, 90, "#2563eb"),
        ("Sibling concept", 730, 90, "#2563eb"),
        ("Prerequisite", 250, 320, "#0891b2"),
        ("Related concept", 730, 320, "#0891b2"),
        ("Positive resource", 490, 70, "#16a34a"),
        ("Candidate negative", 490, 340, "#dc2626"),
    ]
    parts = [_title("Ontology context used by CA-HNM")]
    parts.append(_node(cx, cy, "Target concept", "#b45309", w=190))
    for label, x, y, color in nodes:
        parts.append(_edge(cx, cy, x, y))
        parts.append(_node(x, y, label, color, w=170))
    parts.append(
        f"<text x='110' y='390' font-size='13' fill='{COLORS['text']}'>"
        "The target concept neighborhood defines constraint checks: target mismatch, sibling confusion, prerequisite mismatch, context mismatch, and missing evidence."
        "</text>"
    )
    return _svg(width, height, parts)


def _horizontal_bars(
    title: str,
    values: list[tuple[str, float]],
    x: int,
    y: int,
    width: int,
    height: int,
    max_value: float | None = 1.0,
    color_key: str | None = None,
    suffix: str = "",
) -> str:
    values = [(method, value) for method, value in values if value is not None]
    if not values:
        return ""
    computed_max = max(value for _, value in values) or 1.0
    max_value = computed_max if max_value is None else max(max_value, computed_max)
    label_w = 155
    plot_w = width - label_w - 70
    row_h = min(28, max(20, int((height - 40) / max(1, len(values)))))
    parts = [f"<text x='{x}' y='{y}' font-size='18' font-weight='700'>{_esc(title)}</text>"]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0] if max_value <= 1.0 else [0.0, 0.25, 0.5, 0.75, 1.0]:
        tx = x + label_w + int(plot_w * tick)
        parts.append(f"<line x1='{tx}' y1='{y + 20}' x2='{tx}' y2='{y + height - 8}' stroke='{COLORS['grid']}'/>")
        tick_value = tick * max_value
        tick_label = f"{tick_value:.2f}" if max_value <= 1.0 else f"{tick_value:.0f}"
        parts.append(f"<text x='{tx - 8}' y='{y + height + 10}' font-size='11' fill='{COLORS['muted']}'>{tick_label}</text>")
    for idx, (method, value) in enumerate(values):
        row_y = y + 34 + idx * row_h
        label = _label(method)
        bar_w = int(plot_w * value / max_value)
        color = _method_color(method, color_key)
        parts.append(f"<text x='{x}' y='{row_y + 14}' font-size='12'>{_esc(label)}</text>")
        parts.append(f"<rect x='{x + label_w}' y='{row_y}' width='{bar_w}' height='17' fill='{color}'></rect>")
        parts.append(f"<text x='{x + label_w + bar_w + 6}' y='{row_y + 13}' font-size='12'>{value:.4f}{suffix}</text>")
    return "".join(parts)


def _scatter_panel(bundle: DatasetBundle, x: int, y: int, width: int, height: int, main_methods: list[str]) -> str:
    methods = list(dict.fromkeys([*main_methods, "CA-HNM-full", "CA-HNM-v2"]))
    left = x + 48
    top = y + 28
    plot_w = width - 72
    plot_h = height - 72
    parts = [f"<text x='{x}' y='{y}' font-size='18' font-weight='700'>{_esc(bundle.name)}</text>"]
    parts.append(f"<rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' fill='white' stroke='{COLORS['grid']}'/>")
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        tx = left + int(plot_w * tick)
        ty = top + plot_h - int(plot_h * tick)
        parts.append(f"<line x1='{tx}' y1='{top}' x2='{tx}' y2='{top + plot_h}' stroke='{COLORS['grid']}'/>")
        parts.append(f"<line x1='{left}' y1='{ty}' x2='{left + plot_w}' y2='{ty}' stroke='{COLORS['grid']}'/>")
        parts.append(f"<text x='{tx - 8}' y='{top + plot_h + 18}' font-size='11' fill='{COLORS['muted']}'>{tick:.2g}</text>")
        parts.append(f"<text x='{left - 38}' y='{ty + 4}' font-size='11' fill='{COLORS['muted']}'>{tick:.2g}</text>")
    for method in methods:
        q = _quality(bundle, method, "valid_hard_rate")
        m = _metric(bundle, method, "NDCG@10")
        px = left + int(plot_w * min(1.0, max(0.0, q)))
        py = top + plot_h - int(plot_h * min(1.0, max(0.0, m)))
        color = _method_color(method)
        parts.append(f"<circle cx='{px}' cy='{py}' r='5' fill='{color}'/>")
        parts.append(f"<text x='{px + 7}' y='{py - 7}' font-size='10'>{_esc(_label(method)[:14])}</text>")
    parts.append(f"<text x='{left + 90}' y='{top + plot_h + 42}' font-size='12'>Valid-hard rate</text>")
    parts.append(f"<text x='{left - 44}' y='{top - 8}' font-size='12'>NDCG@10</text>")
    return "".join(parts)


def _legend(x: int, y: int) -> list[str]:
    return [
        f"<rect x='{x}' y='{y}' width='14' height='14' fill='{COLORS['baseline']}'/>",
        f"<text x='{x + 20}' y='{y + 12}' font-size='12'>baseline</text>",
        f"<rect x='{x + 100}' y='{y}' width='14' height='14' fill='{COLORS['ours']}'/>",
        f"<text x='{x + 120}' y='{y + 12}' font-size='12'>CA-HNM</text>",
    ]


def _node(x: int, y: int, label: str, color: str, w: int = 160, h: int = 44) -> str:
    return (
        f"<rect x='{x - w // 2}' y='{y - h // 2}' width='{w}' height='{h}' rx='8' fill='{color}'/>"
        f"<text x='{x}' y='{y + 5}' font-size='14' text-anchor='middle' fill='white' font-weight='700'>{_esc(label)}</text>"
    )


def _edge(x1: int, y1: int, x2: int, y2: int) -> str:
    return f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{COLORS['muted']}' stroke-width='2'/>"


def _title(text: str) -> str:
    return f"<text x='40' y='34' font-size='22' font-weight='700'>{_esc(text)}</text>"


def _svg(width: int, height: int, parts: list[str]) -> str:
    style = (
        "<style>"
        "text{font-family:Arial,Helvetica,sans-serif;fill:#111827}"
        "</style>"
    )
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>{style}{''.join(parts)}</svg>\n"
    )


def _metric(bundle: DatasetBundle, method: str, metric: str) -> float:
    return _float(bundle.retrieval.get(method, {}).get(metric))


def _quality(bundle: DatasetBundle, method: str, metric: str) -> float:
    return _float(bundle.quality.get(method, {}).get(metric))


def _float(value) -> float:
    try:
        if value in {None, ""}:
            return 0.0
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return 0.0
        return number
    except (TypeError, ValueError):
        return 0.0


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _method_color(method: str, color_key: str | None = None) -> str:
    if color_key:
        return COLORS[color_key]
    if method.startswith("CA-HNM"):
        return COLORS["ours_dark"] if "mixed" in method or "v2-mixed" in method else COLORS["ours"]
    if method in {"ANCE", "ADORE", "RocketQA-Denoised"}:
        return COLORS["dense"]
    return COLORS["baseline"]


def _esc(text: str) -> str:
    return html.escape(str(text))


def _readme(out_dir: Path) -> str:
    return f"""# CA-HNM Paper Figures

Generated figures are in `{out_dir}`.

Recommended paper use:

- `fig_main_ndcg.svg`: main retrieval effectiveness figure with grouped representative baselines.
- `fig_quality_tradeoff.svg`: shows why CA-HNM is not only a retriever-score method; it changes negative quality.
- `fig_ablation_ndcg.svg`: use for the ablation subsection.
- `fig_ablation_quality.svg`: use beside ablation results to show semantic negative quality.
- `fig_training_cost.svg`: use only if you discuss computational cost; it is not a loss curve.
- `fig_ontology_stats.svg`: data-driven ontology statistics.
- `fig_ontology_context.svg`: method schematic showing how ontology context is used.

For CIKM LaTeX, prefer converting SVG to PDF:

```bash
inkscape fig_main_ndcg.svg --export-type=pdf
```

The exact numeric values are also saved in `paper_metrics_summary.csv`.
"""


if __name__ == "__main__":
    main()
