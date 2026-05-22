from __future__ import annotations

import csv
import inspect
import json
import random
import time
from pathlib import Path

from .io_utils import write_jsonl
from .miner import positives_by_query
from .schemas import Document, NegativeRecord, Qrel, Query


def build_triplets(
    queries: list[Query],
    documents: list[Document],
    qrels: list[Qrel],
    negatives: list[NegativeRecord],
    random_seed: int = 13,
) -> list[dict]:
    rng = random.Random(random_seed)
    doc_by_id = {doc.id: doc for doc in documents}
    query_by_id = {query.id: query for query in queries}
    positives = positives_by_query(qrels)
    neg_by_query: dict[str, list[NegativeRecord]] = {}
    for neg in negatives:
        neg_by_query.setdefault(neg.query_id, []).append(neg)

    rows: list[dict] = []
    for query_id, positive_doc_ids in positives.items():
        positive_doc_ids = list(positive_doc_ids)
        if not positive_doc_ids or query_id not in neg_by_query:
            continue
        for neg in neg_by_query[query_id]:
            pos_id = rng.choice(positive_doc_ids)
            rows.append(
                {
                    "query_id": query_id,
                    "positive_doc_id": pos_id,
                    "negative_doc_id": neg.doc_id,
                    "query": query_by_id[query_id].text,
                    "positive": doc_by_id[pos_id].searchable_text,
                    "negative": doc_by_id[neg.doc_id].searchable_text,
                    "negative_source": neg.source,
                    "violation_types": list(neg.violation_types),
                }
            )
    return rows


def write_triplets(path: str | Path, triplets: list[dict]) -> None:
    write_jsonl(path, triplets)


def train_sentence_transformer_triplets(
    triplets: list[dict],
    output_dir: str | Path,
    model_name: str = "BAAI/bge-base-en-v1.5",
    epochs: int = 1,
    batch_size: int = 16,
    strategy: str | None = None,
    logs_dir: str | Path | None = None,
    device: str | None = None,
    max_seq_length: int | None = None,
    use_amp: bool = False,
    loss_name: str = "triplet",
    learning_rate: float | None = None,
    warmup_ratio: float = 0.1,
) -> dict:
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError(
            "sentence-transformers and torch are required for training. "
            "Install the dense extra or run comparison without --train."
        ) from exc

    examples = [
        InputExample(texts=[row["query"], row["positive"], row["negative"]])
        for row in triplets
    ]
    output_dir = Path(output_dir)
    logs_dir = Path(logs_dir) if logs_dir is not None else output_dir / "training_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name, device=device)
    if max_seq_length:
        model.max_seq_length = max_seq_length
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_objectives = _make_train_objectives(losses, model, loader, loss_name)
    total_steps = len(loader) * epochs
    log_rows: list[dict] = []
    start_time = time.time()
    summary_writer = _make_summary_writer(logs_dir / "tensorboard")

    def callback(score: float, epoch: int, steps: int) -> None:
        elapsed = time.time() - start_time
        row = {
            "strategy": strategy or output_dir.name,
            "epoch": epoch,
            "steps": steps,
            "score": float(score),
            "elapsed_sec": elapsed,
            "total_steps": total_steps,
        }
        log_rows.append(row)
        _append_training_log(logs_dir / "training_log.csv", row)
        _append_jsonl(logs_dir / "training_log.jsonl", row)
        if summary_writer is not None:
            summary_writer.add_scalar("eval/score", float(score), steps)
            summary_writer.add_scalar("train/epoch", epoch, steps)
            summary_writer.add_scalar("time/elapsed_sec", elapsed, steps)

    train_summary = {
        "strategy": strategy or output_dir.name,
        "model_name": model_name,
        "output_dir": str(output_dir),
        "triplets": len(triplets),
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "max_seq_length": max_seq_length,
        "use_amp": use_amp,
        "loss": loss_name,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "steps_per_epoch": len(loader),
        "total_steps": total_steps,
        "started_at": start_time,
    }
    (logs_dir / "training_config.json").write_text(json.dumps(train_summary, indent=2, sort_keys=True), encoding="utf-8")

    fit_kwargs = {
        "train_objectives": train_objectives,
        "epochs": epochs,
        "show_progress_bar": True,
        "callback": callback,
    }
    if "use_amp" in inspect.signature(model.fit).parameters:
        fit_kwargs["use_amp"] = use_amp
    if learning_rate is not None and "optimizer_params" in inspect.signature(model.fit).parameters:
        fit_kwargs["optimizer_params"] = {"lr": learning_rate}
    if "warmup_steps" in inspect.signature(model.fit).parameters:
        fit_kwargs["warmup_steps"] = max(0, int(total_steps * warmup_ratio))
    model.fit(**fit_kwargs)
    model.save(str(output_dir))
    if summary_writer is not None:
        summary_writer.flush()
        summary_writer.close()
    train_summary["finished_at"] = time.time()
    train_summary["duration_sec"] = train_summary["finished_at"] - start_time
    train_summary["logged_events"] = len(log_rows)
    (logs_dir / "training_summary.json").write_text(json.dumps(train_summary, indent=2, sort_keys=True), encoding="utf-8")
    return train_summary


def _make_train_objectives(losses, model, loader, loss_name: str):
    if loss_name == "triplet":
        return [(loader, losses.TripletLoss(model=model))]
    if loss_name in {"mnrl", "multiple-negatives"}:
        return [(loader, losses.MultipleNegativesRankingLoss(model=model))]
    if loss_name == "hybrid":
        return [
            (loader, losses.TripletLoss(model=model)),
            (loader, losses.MultipleNegativesRankingLoss(model=model)),
        ]
    raise ValueError(f"Unknown training loss: {loss_name}")


def _append_training_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["strategy", "epoch", "steps", "score", "elapsed_sec", "total_steps"])
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _make_summary_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(log_dir))
    except Exception:
        try:
            from tensorboardX import SummaryWriter

            return SummaryWriter(logdir=str(log_dir))
        except Exception:
            return None
