#!/usr/bin/env python
"""Regenerate the paper's RQ1 forest plot from canonical aggregate results."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FACTORIAL_COMPARISONS = ROOT / "results" / "factorial" / "comparisons.csv"
EXPANDED_COMPARISONS = ROOT / "results" / "expanded" / "comparisons.csv"
FOREST_SOURCE_DATA = ROOT / "results" / "effect_forest_source.csv"
OUT_DIR = ROOT / "artifacts" / "figures"

BLUE = "#0077BB"
ORANGE = "#EE7733"
GREY = "#666666"

FOREST_ROW_ORDER = [
    ("course_skill_atlas", "bge-base", "cached-mnrl", "Course | BGE | Cached MNRL"),
    ("course_skill_atlas", "bge-base", "triplet", "Course | BGE | Triplet"),
    ("course_skill_atlas", "e5-base", "cached-mnrl", "Course | E5 | Cached MNRL"),
    ("course_skill_atlas", "e5-base", "triplet", "Course | E5 | Triplet"),
    ("mooccubex", "bge-base", "cached-mnrl", "MOOCCubeX | BGE | Cached MNRL"),
    ("mooccubex", "bge-base", "triplet", "MOOCCubeX | BGE | Triplet"),
    ("mooccubex", "e5-base", "cached-mnrl", "MOOCCubeX | E5 | Cached MNRL"),
    ("mooccubex", "e5-base", "triplet", "MOOCCubeX | E5 | Triplet"),
]

FOREST_EXPORT_FIELDS = [
    "display_order",
    "display_label",
    "dataset",
    "model_key",
    "loss",
    "eval_split",
    "baseline",
    "strategy",
    "metric",
    "n_seeds",
    "n_queries_min",
    "baseline_mean",
    "strategy_mean",
    "mean_diff",
    "seed_sd_diff",
    "paired_effect_dz",
    "seed_win_rate",
    "hierarchical_ci95_low",
    "hierarchical_ci95_high",
    "method_mean_higher",
    "ci_relation_to_zero",
    "source_run",
    "source_artifact",
    "ci_method",
]

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def canonical_forest_rows() -> list[dict[str, str]]:
    """Load and validate the eight Pure-versus-rank-matched NDCG effects.

    The expanded execution is canonical for BGE/cached-MNRL because that run
    contains the mixed and structural controls. The factorial execution supplies
    the other six encoder-loss configurations.
    """
    selected: dict[tuple[str, str, str], dict[str, str]] = {}
    sources = [
        ("factorial", FACTORIAL_COMPARISONS),
        ("expanded", EXPANDED_COMPARISONS),
    ]
    for source_run, source_path in sources:
        with source_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if not (
                    row["baseline"] == "CA-HNM-rank-matched"
                    and row["strategy"] == "CA-HNM-full"
                    and row["metric"] == "NDCG@10"
                ):
                    continue
                if source_run == "expanded" and not (
                    row["model_key"] == "bge-base" and row["loss"] == "cached-mnrl"
                ):
                    continue
                exported = dict(row)
                exported["source_run"] = source_run
                exported["source_artifact"] = source_path.relative_to(ROOT).as_posix()
                selected[(row["dataset"], row["model_key"], row["loss"])] = exported

    rows: list[dict[str, str]] = []
    for display_order, (dataset, model, loss, display_label) in enumerate(
        FOREST_ROW_ORDER, start=1
    ):
        key = (dataset, model, loss)
        if key not in selected:
            raise ValueError(f"Missing forest comparison: {key}")
        row = selected[key]
        effect = float(row["mean_diff"])
        low = float(row["hierarchical_ci95_low"])
        high = float(row["hierarchical_ci95_high"])
        row.update(
            {
                "display_order": str(display_order),
                "display_label": display_label,
                "method_mean_higher": str(effect > 0).lower(),
                "ci_relation_to_zero": (
                    "above" if low > 0 else "below" if high < 0 else "spans"
                ),
                "ci_method": "hierarchical bootstrap over seeds and queries",
            }
        )
        rows.append(row)

    if len(rows) != 8:
        raise ValueError(f"Expected 8 forest rows, found {len(rows)}")
    for row in rows:
        baseline = float(row["baseline_mean"])
        strategy = float(row["strategy_mean"])
        effect = float(row["mean_diff"])
        low = float(row["hierarchical_ci95_low"])
        high = float(row["hierarchical_ci95_high"])
        if abs((strategy - baseline) - effect) > 1e-12:
            raise ValueError(f"Inconsistent mean difference: {row['display_label']}")
        if not low <= effect <= high:
            raise ValueError(f"Point estimate outside CI: {row['display_label']}")

    positive_means = sum(float(row["mean_diff"]) > 0 for row in rows)
    intervals_above_zero = sum(
        float(row["hierarchical_ci95_low"]) > 0 for row in rows
    )
    if (positive_means, intervals_above_zero) != (6, 4):
        raise ValueError(
            "Paper-claim mismatch: "
            f"positive_means={positive_means}, intervals_above_zero={intervals_above_zero}"
        )
    return rows


def write_source_data(rows: list[dict[str, str]]) -> None:
    FOREST_SOURCE_DATA.parent.mkdir(parents=True, exist_ok=True)
    with FOREST_SOURCE_DATA.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FOREST_EXPORT_FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in FOREST_EXPORT_FIELDS} for row in rows
        )


def make_effect_forest(rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(6.9, 3.55))
    y_positions = np.arange(len(rows))[::-1]

    for y, row in zip(y_positions, rows):
        loss = "Triplet" if row["loss"] == "triplet" else "Cached MNRL"
        effect = float(row["mean_diff"])
        low = float(row["hierarchical_ci95_low"])
        high = float(row["hierarchical_ci95_high"])
        color = BLUE if loss == "Triplet" else ORANGE
        marker = "o" if loss == "Triplet" else "s"
        ax.errorbar(
            effect,
            y,
            xerr=np.array([[effect - low], [high - effect]]),
            fmt=marker,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5.5,
            capsize=2.5,
            linewidth=1.2,
            zorder=3,
        )
        ax.annotate(
            f"{effect:+.4f}",
            xy=(effect, y),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.10",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
            },
            zorder=4,
        )

    ax.axvline(0, color=GREY, linestyle="--", linewidth=0.8, zorder=1)
    ax.axhline(3.5, color="#BBBBBB", linewidth=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["display_label"] for row in rows])
    ax.set_xlabel(r"$\Delta$ NDCG@10 (CA-HNM-Pure $-$ Rank-matched)")
    ax.set_xlim(-0.025, 0.132)
    ax.set_ylim(-0.45, len(rows) - 0.35)
    ax.set_xticks(np.arange(-0.02, 0.121, 0.02))
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[
            plt.Line2D([], [], color=BLUE, marker="o", linestyle="None", label="Triplet"),
            plt.Line2D([], [], color=ORANGE, marker="s", linestyle="None", label="Cached MNRL"),
        ],
        loc="lower right",
        frameon=False,
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "effect_forest.pdf", format="pdf")
    fig.savefig(OUT_DIR / "effect_forest.png", format="png")
    plt.close(fig)


def main() -> None:
    rows = canonical_forest_rows()
    write_source_data(rows)
    make_effect_forest(rows)
    print(f"Wrote {FOREST_SOURCE_DATA}")
    print(f"Wrote Figure 3 to {OUT_DIR}")


if __name__ == "__main__":
    main()
