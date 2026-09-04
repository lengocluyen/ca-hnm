"""Generate compact LaTeX tables from the completed multi-seed runs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "artifacts" / "tables"

METRICS = ("NDCG@10", "MRR@10", "MAP@100", "Recall@100")
DATASETS = ("course_skill_atlas", "mooccubex")
DATASET_LABELS = {
    "course_skill_atlas": "Course-Skill Atlas",
    "mooccubex": "MOOCCubeX",
}
MODELS = ("bge-base", "e5-base")
MODEL_LABELS = {"bge-base": "BGE", "e5-base": "E5"}
LOSSES = ("cached-mnrl", "triplet")
LOSS_LABELS = {"cached-mnrl": "Cached MNRL", "triplet": "Triplet"}
CORE_STRATEGIES = (
    "DPR-Random",
    "DenseNeg",
    "CA-HNM-rank-matched",
    "CA-HNM-full",
)
HYBRID_STRATEGIES = (
    "DPR-Random",
    "DenseNeg",
    "CA-HNM-mixed",
    "CA-HNM-matched-mixed",
)
ABLATION_STRATEGIES = (
    "CA-HNM-full",
    "CA-HNM-label-only",
    "CA-HNM-shuffled-graph",
    "CA-HNM-no-ontology",
)
STRATEGY_LABELS = {
    "DPR-Random": "DPR-Random",
    "DenseNeg": "DenseNeg",
    "CA-HNM-rank-matched": "Rank-matched",
    "CA-HNM-full": r"\textsc{CA-HNM}-Pure",
    "CA-HNM-mixed": r"\textsc{CA-HNM}-Mixed",
    "CA-HNM-matched-mixed": "Matched-mixed",
    "CA-HNM-label-only": "Label-only",
    "CA-HNM-shuffled-graph": "Shuffled graph",
    "CA-HNM-no-ontology": "No structure",
}


def load_means(path: Path) -> dict[tuple[str, str, str, str, str], float]:
    values: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (
                row["dataset"],
                row["model_key"],
                row["loss"],
                row["strategy"],
                row["metric"],
            )
            values[key].append(float(row["mean"]))
    return {key: sum(items) / len(items) for key, items in values.items()}


def cell(value: float, best: float) -> str:
    rendered = f"{value:.4f}"
    return rf"\textbf{{{rendered}}}" if abs(value - best) < 5e-12 else rendered


def core_table(means: dict[tuple[str, str, str, str, str], float]) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\caption{Development retrieval performance averaged over ten training seeds. Bold marks the best strategy for each encoder--loss--metric setting.}",
        r"\label{tab:core}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Encoder & Loss & Strategy & NDCG@10 & MRR@10 & MAP@100 & R@100 \\",
        r"\midrule",
    ]
    for dataset_index, dataset in enumerate(DATASETS):
        lines.append(rf"\multicolumn{{7}}{{c}}{{\textbf{{{DATASET_LABELS[dataset]}}}}} \\")
        lines.append(r"\midrule")
        for model_index, model in enumerate(MODELS):
            for loss_index, loss in enumerate(LOSSES):
                best = {
                    metric: max(means[(dataset, model, loss, strategy, metric)] for strategy in CORE_STRATEGIES)
                    for metric in METRICS
                }
                for strategy_index, strategy in enumerate(CORE_STRATEGIES):
                    model_cell = MODEL_LABELS[model] if loss_index == 0 and strategy_index == 0 else ""
                    loss_cell = LOSS_LABELS[loss] if strategy_index == 0 else ""
                    result_cells = [cell(means[(dataset, model, loss, strategy, metric)], best[metric]) for metric in METRICS]
                    lines.append(
                        f"{model_cell} & {loss_cell} & {STRATEGY_LABELS[strategy]} & "
                        + " & ".join(result_cells)
                        + r" \\"
                    )
                if not (model_index == len(MODELS) - 1 and loss_index == len(LOSSES) - 1):
                    lines.append(r"\addlinespace[1pt]")
        if dataset_index != len(DATASETS) - 1:
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
    return "\n".join(lines) + "\n"


def two_dataset_table(
    means: dict[tuple[str, str, str, str, str], float],
    strategies: tuple[str, ...],
    caption: str,
    label: str,
) -> str:
    model = "bge-base"
    loss = "cached-mnrl"
    lines = [
        r"\begin{table}[H]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"& \multicolumn{4}{c}{Course-Skill Atlas} & \multicolumn{4}{c}{MOOCCubeX} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        r"Strategy & NDCG & MRR & MAP & R@100 & NDCG & MRR & MAP & R@100 \\",
        r"\midrule",
    ]
    best = {
        dataset: {
            metric: max(means[(dataset, model, loss, strategy, metric)] for strategy in strategies)
            for metric in METRICS
        }
        for dataset in DATASETS
    }
    for strategy in strategies:
        result_cells = []
        for dataset in DATASETS:
            result_cells.extend(
                cell(means[(dataset, model, loss, strategy, metric)], best[dataset][metric])
                for metric in METRICS
            )
        lines.append(
            f"{STRATEGY_LABELS[strategy]} & "
            + " & ".join(result_cells)
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table}"))
    return "\n".join(lines) + "\n"


def hybrid_table(means: dict[tuple[str, str, str, str, str], float]) -> str:
    return two_dataset_table(
        means,
        HYBRID_STRATEGIES,
        r"Main comparison and component ablation of \textsc{CA-HNM}-Mixed with BGE and cached MNRL, averaged over ten training seeds. Matched-mixed replaces its constraint-aware component with rank-matched negatives. NDCG and MRR use cutoff 10; MAP and recall use cutoff 100.",
        "tab:hybrid",
    )


def ablation_table(means: dict[tuple[str, str, str, str, str], float]) -> str:
    return two_dataset_table(
        means,
        ABLATION_STRATEGIES,
        r"Structural ablation of \textsc{CA-HNM}-Pure with BGE and cached MNRL, averaged over ten training seeds.",
        "tab:abl",
    )


def main() -> None:
    core = load_means(ROOT / "results" / "factorial" / "seed_metrics.csv")
    ablation = load_means(ROOT / "results" / "expanded" / "seed_metrics.csv")
    # The expanded execution is the canonical source for every BGE/cached-MNRL
    # cell because it contains the mixed and structural controls.  Override the
    # same configuration in the factorial table so repeated method values are
    # identical across all manuscript tables.
    for key, value in ablation.items():
        _, model, loss, strategy, _ = key
        if model == "bge-base" and loss == "cached-mnrl" and strategy in CORE_STRATEGIES:
            core[key] = value
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "core_results.tex").write_text(core_table(core), encoding="utf-8")
    (TABLE_DIR / "hybrid_results.tex").write_text(hybrid_table(ablation), encoding="utf-8")
    (TABLE_DIR / "ablation_results.tex").write_text(ablation_table(ablation), encoding="utf-8")
    print(f"wrote {TABLE_DIR / 'core_results.tex'}")
    print(f"wrote {TABLE_DIR / 'hybrid_results.tex'}")
    print(f"wrote {TABLE_DIR / 'ablation_results.tex'}")


if __name__ == "__main__":
    main()
