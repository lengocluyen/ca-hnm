#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import shlex
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate run_full_suite commands for CA-HNM sensitivity analysis.")
    parser.add_argument("--datasets", nargs="+", default=["mooccubex", "course_skill_atlas"])
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--dense-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--train-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--dense-batch-size", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--loss", default="mnrl")
    parser.add_argument("--top-k-values", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--negatives-values", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--confidence-values", type=float, nargs="+", default=[0.70, 0.75, 0.80])
    parser.add_argument(
        "--weight-settings",
        nargs="+",
        default=["default:0.55,0.30,0.15,0.06", "retrieval:0.70,0.20,0.10,0.06", "constraint:0.40,0.45,0.15,0.06"],
        help="Named RA weights as name:retrieval,constraint,ontology,diversity.",
    )
    parser.add_argument("--out-root-prefix", default="runs/sensitivity")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    commands = []
    weight_settings = [_parse_weights(item) for item in args.weight_settings]
    for dataset, top_k, negatives, confidence, weights in itertools.product(
        args.datasets,
        args.top_k_values,
        args.negatives_values,
        args.confidence_values,
        weight_settings,
    ):
        name, retrieval_weight, constraint_weight, ontology_weight, diversity_penalty = weights
        out_root = (
            f"{args.out_root_prefix}_{dataset}_top{top_k}_neg{negatives}_"
            f"conf{str(confidence).replace('.', 'p')}_{name}"
        )
        cmd = [
            "python3",
            "scripts/run_full_suite.py",
            "--datasets",
            dataset,
            "--judges",
            "heuristic",
            "--gpu",
            args.gpu,
            "--dense-model",
            args.dense_model,
            "--train-model",
            args.train_model,
            "--dense-batch-size",
            str(args.dense_batch_size),
            "--train-batch-size",
            str(args.train_batch_size),
            "--max-seq-length",
            str(args.max_seq_length),
            "--top-k",
            str(top_k),
            "--negatives-per-query",
            str(negatives),
            "--confidence",
            str(confidence),
            "--epochs",
            str(args.epochs),
            "--loss",
            args.loss,
            "--candidate-fusion",
            "rrf",
            "--selection-policy",
            "retrieval_aware",
            "--retrieval-weight",
            str(retrieval_weight),
            "--constraint-weight",
            str(constraint_weight),
            "--ontology-weight",
            str(ontology_weight),
            "--diversity-penalty",
            str(diversity_penalty),
            "--eval-top-k",
            "100",
            "--skip-llm-validation",
            "--out-root",
            out_root,
        ]
        commands.append(" ".join(shlex.quote(part) for part in cmd))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"wrote {len(commands)} sensitivity commands to {out}")


def _parse_weights(text: str) -> tuple[str, float, float, float, float]:
    name, values = text.split(":", 1)
    parts = [float(part) for part in values.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Invalid weight setting {text!r}; expected name:r,c,o,d")
    return name, parts[0], parts[1], parts[2], parts[3]


if __name__ == "__main__":
    main()
