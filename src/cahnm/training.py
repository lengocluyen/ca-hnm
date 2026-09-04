from __future__ import annotations

import csv
import json
import os
import platform
import random
import sys
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
    random_seed: int = 13,
    deterministic: bool = True,
    cached_mini_batch_size: int = 4,
    mnrl_scale: float = 20.0,
) -> dict:
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        import sentence_transformers
        import torch
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError(
            "sentence-transformers and torch are required for training. "
            "Install the dense extra or run comparison without --train."
        ) from exc

    _set_global_seed(torch, random_seed, deterministic)
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
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    loader = DataLoader(
        examples,
        shuffle=True,
        batch_size=batch_size,
        generator=generator,
        collate_fn=model.smart_batching_collate,
    )
    train_objectives = _make_train_objectives(
        losses,
        model,
        loader,
        loss_name,
        cached_mini_batch_size=cached_mini_batch_size,
        mnrl_scale=mnrl_scale,
    )
    total_steps = len(loader) * epochs * len(train_objectives)
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
        "learning_rate": learning_rate or 2e-5,
        "warmup_ratio": warmup_ratio,
        "random_seed": random_seed,
        "deterministic": deterministic,
        "cached_mini_batch_size": cached_mini_batch_size,
        "mnrl_scale": mnrl_scale,
        "steps_per_epoch": len(loader),
        "total_steps": total_steps,
        "started_at": start_time,
        "trainer": "explicit-pytorch-loop",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (logs_dir / "training_config.json").write_text(json.dumps(train_summary, indent=2, sort_keys=True), encoding="utf-8")

    _fit_explicit_pytorch(
        torch,
        model,
        train_objectives,
        epochs=epochs,
        total_steps=total_steps,
        learning_rate=learning_rate or 2e-5,
        warmup_steps=max(0, int(total_steps * warmup_ratio)),
        use_amp=use_amp,
        callback=callback,
        summary_writer=summary_writer,
    )
    model.save(str(output_dir))
    if summary_writer is not None:
        summary_writer.flush()
        summary_writer.close()
    train_summary["finished_at"] = time.time()
    train_summary["duration_sec"] = train_summary["finished_at"] - start_time
    train_summary["logged_events"] = len(log_rows)
    (logs_dir / "training_summary.json").write_text(json.dumps(train_summary, indent=2, sort_keys=True), encoding="utf-8")
    return train_summary


def _fit_explicit_pytorch(
    torch,
    model,
    train_objectives,
    epochs: int,
    total_steps: int,
    learning_rate: float,
    warmup_steps: int,
    use_amp: bool,
    callback,
    summary_writer,
) -> None:
    """Train without the optional Hugging Face Trainer dependency stack."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def lr_multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = max(0, total_steps - step)
        decay_steps = max(1, total_steps - warmup_steps)
        return remaining / decay_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    model_device = next(model.parameters()).device
    amp_enabled = bool(use_amp and model_device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    global_step = 0
    model.train()

    for epoch in range(epochs):
        for loader, loss_model in train_objectives:
            loss_model.train()
            for sentence_features, labels in loader:
                sentence_features = [
                    {
                        key: value.to(model_device) if hasattr(value, "to") else value
                        for key, value in features.items()
                    }
                    for features in sentence_features
                ]
                labels = labels.to(model_device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    loss_value = loss_model(sentence_features, labels)
                scaler.scale(loss_value).backward()
                if amp_enabled:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1
                numeric_loss = float(loss_value.detach().cpu())
                callback(-numeric_loss, epoch + 1, global_step)
                if summary_writer is not None:
                    summary_writer.add_scalar("train/loss", numeric_loss, global_step)
                    summary_writer.add_scalar(
                        "train/learning_rate", optimizer.param_groups[0]["lr"], global_step
                    )
        print(f"completed training epoch {epoch + 1}/{epochs} ({global_step}/{total_steps} steps)", flush=True)


def _make_train_objectives(
    losses,
    model,
    loader,
    loss_name: str,
    cached_mini_batch_size: int = 4,
    mnrl_scale: float = 20.0,
):
    if loss_name == "triplet":
        return [(loader, losses.TripletLoss(model=model))]
    if loss_name == "cached-mnrl":
        if cached_mini_batch_size <= 0:
            raise ValueError("cached_mini_batch_size must be positive")
        return [
            (
                loader,
                losses.CachedMultipleNegativesRankingLoss(
                    model=model,
                    scale=mnrl_scale,
                    mini_batch_size=cached_mini_batch_size,
                ),
            )
        ]
    raise ValueError(f"Unknown training loss: {loss_name}")


def _set_global_seed(torch, random_seed: int, deterministic: bool) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(random_seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(random_seed)
    try:
        import numpy as np

        np.random.seed(random_seed)
    except Exception:
        pass
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


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
