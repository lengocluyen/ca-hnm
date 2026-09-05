from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

from .schemas import Document, NegativeRecord, OntologyNode, OntologyRelation, Qrel, Query

T = TypeVar("T")


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_corpus(path: str | Path) -> list[Document]:
    docs: list[Document] = []
    for row in read_jsonl(path):
        docs.append(
            Document(
                id=str(row["_id"] if "_id" in row else row["id"]),
                title=str(row.get("title", "")),
                text=str(row.get("text", "")),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return docs


def write_corpus(path: str | Path, docs: Iterable[Document]) -> None:
    write_jsonl(
        path,
        (
            {
                "_id": doc.id,
                "title": doc.title,
                "text": doc.text,
                "metadata": doc.metadata,
            }
            for doc in docs
        ),
    )


def load_queries(path: str | Path) -> list[Query]:
    queries: list[Query] = []
    for row in read_jsonl(path):
        queries.append(
            Query(
                id=str(row["_id"] if "_id" in row else row["id"]),
                text=str(row.get("text", "")),
                target_concept=row.get("target_concept"),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return queries


def write_queries(path: str | Path, queries: Iterable[Query]) -> None:
    write_jsonl(
        path,
        (
            {
                "_id": query.id,
                "text": query.text,
                "target_concept": query.target_concept,
                "metadata": query.metadata,
            }
            for query in queries
        ),
    )


def load_qrels(path: str | Path) -> list[Qrel]:
    qrels: list[Qrel] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qrels.append(
                Qrel(
                    query_id=str(row["query_id"]),
                    doc_id=str(row["doc_id"]),
                    relevance=int(row.get("relevance", 1)),
                )
            )
    return qrels


def write_qrels(path: str | Path, qrels: Iterable[Qrel]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "doc_id", "relevance"], delimiter="\t")
        writer.writeheader()
        for qrel in qrels:
            writer.writerow({"query_id": qrel.query_id, "doc_id": qrel.doc_id, "relevance": qrel.relevance})


def load_negatives(path: str | Path) -> list[NegativeRecord]:
    negatives: list[NegativeRecord] = []
    for row in read_jsonl(path):
        negatives.append(
            NegativeRecord(
                query_id=str(row["query_id"]),
                doc_id=str(row["doc_id"]),
                source=str(row.get("source", "")),
                label=str(row.get("label", "")),
                violation_types=tuple(row.get("violation_types", [])),
                evidence=str(row.get("evidence", "")),
                confidence=float(row.get("confidence", 0.0)),
                rank=row.get("rank"),
                score=row.get("score"),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return negatives


def write_negatives(path: str | Path, negatives: Iterable[NegativeRecord]) -> None:
    write_jsonl(
        path,
        (
            {
                "query_id": n.query_id,
                "doc_id": n.doc_id,
                "source": n.source,
                "label": n.label,
                "violation_types": list(n.violation_types),
                "evidence": n.evidence,
                "confidence": n.confidence,
                "rank": n.rank,
                "score": n.score,
                "metadata": n.metadata,
            }
            for n in negatives
        ),
    )


def write_ontology(path: str | Path, nodes: Iterable[OntologyNode], relations: Iterable[OntologyRelation]) -> None:
    payload = {
        "nodes": [
            {
                "id": node.id,
                "label": node.label,
                "aliases": list(node.aliases),
                "level": node.level,
                "metadata": node.metadata,
            }
            for node in nodes
        ],
        "relations": [
            {
                "source": rel.source,
                "target": rel.target,
                "type": rel.type,
                "weight": rel.weight,
                "metadata": rel.metadata,
            }
            for rel in relations
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
