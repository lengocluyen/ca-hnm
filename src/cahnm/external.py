from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io_utils import read_jsonl
from .schemas import NegativeRecord, Query, RetrievalRun


QUERY_ID_KEYS = ("query_id", "qid", "_id", "id", "question_id")
DOC_ID_KEYS = ("doc_id", "docid", "pid", "passage_id", "_id", "id")
RESULT_LIST_KEYS = ("ctxs", "contexts", "results", "hits", "docs", "passages")


def load_trec_run(path: str | Path, name: str | None = None) -> RetrievalRun:
    """Load a standard TREC run: qid Q0 docid rank score tag."""
    rankings: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_no} is not a valid TREC run row")
            qid, _, docid, rank_text, score_text, _tag = parts[:6]
            try:
                rank = int(rank_text)
            except ValueError:
                rank = len(rankings[qid]) + 1
            try:
                score = float(score_text)
            except ValueError:
                score = 0.0
            rankings[str(qid)].append((rank, str(docid), score))

    sorted_rankings = {
        qid: [(docid, score) for rank, docid, score in sorted(rows, key=lambda row: (row[0], -row[2]))]
        for qid, rows in rankings.items()
    }
    return RetrievalRun(name or Path(path).stem, sorted_rankings)


def load_json_run(path: str | Path, name: str | None = None, queries: Iterable[Query] | None = None) -> RetrievalRun:
    """Load common dense-retriever output formats.

    Supported shapes:
    - DPR-style list/jsonl rows with `question` and `ctxs`.
    - BEIR-style dict: `{qid: {docid: score}}`.
    - Generic rows: `{qid, docid, score}` or `{qid, results: [...]}`.
    """
    path = Path(path)
    query_list = list(queries or [])
    queries_by_text = {query.text.strip(): query.id for query in query_list}
    queries_by_index = {idx: query.id for idx, query in enumerate(query_list)}

    payload = _load_json_payload(path)
    rankings: dict[str, list[tuple[str, float]]] = defaultdict(list)

    if isinstance(payload, dict) and _looks_like_beir_run(payload):
        for qid, doc_scores in payload.items():
            rows = [(str(docid), _float_or_zero(score)) for docid, score in doc_scores.items()]
            rankings[str(qid)].extend(sorted(rows, key=lambda row: row[1], reverse=True))
        return RetrievalRun(name or path.stem, dict(rankings))

    if isinstance(payload, dict):
        rows = payload.get("data") or payload.get("rows") or payload.get("results") or [payload]
    else:
        rows = payload

    if not isinstance(rows, list):
        raise ValueError(f"Unsupported JSON run structure in {path}")

    for row_idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        qid = _row_query_id(row, row_idx, queries_by_text, queries_by_index)
        if qid is None:
            raise ValueError(
                f"Could not infer query id for row {row_idx} in {path}; "
                "include qid/query_id or pass matching queries for DPR question text/order mapping."
            )

        result_rows = _result_rows(row)
        if result_rows:
            for rank, result in enumerate(result_rows, start=1):
                doc_id = _row_doc_id(result)
                if doc_id is None:
                    continue
                score = _float_or_zero(result.get("score", result.get("retrieval_score", 0.0)))
                rankings[qid].append((doc_id, score if score else float(-rank)))
        else:
            doc_id = _row_doc_id(row)
            if doc_id is None:
                continue
            score = _float_or_zero(row.get("score", row.get("retrieval_score", 0.0)))
            rankings[qid].append((doc_id, score))

    return RetrievalRun(name or path.stem, _dedupe_rankings(rankings))


def load_run(
    path: str | Path,
    run_format: str = "auto",
    name: str | None = None,
    queries: Iterable[Query] | None = None,
) -> RetrievalRun:
    path = Path(path)
    fmt = run_format.lower()
    if fmt == "auto":
        fmt = "trec" if path.suffix.lower() in {".run", ".trec"} else "json"
    if fmt == "trec":
        return load_trec_run(path, name=name)
    if fmt in {"json", "jsonl", "dpr", "beir"}:
        return load_json_run(path, name=name, queries=queries)
    raise ValueError(f"Unsupported run format: {run_format}")


def load_external_negatives(
    path: str | Path,
    source: str,
    input_format: str = "auto",
    label: str = "HardNeg",
) -> list[NegativeRecord]:
    path = Path(path)
    fmt = input_format.lower()
    if fmt == "auto":
        if path.suffix.lower() in {".run", ".trec"}:
            fmt = "trec"
        elif path.suffix.lower() in {".tsv", ".csv"}:
            fmt = "table"
        else:
            fmt = "jsonl"
    if fmt == "trec":
        return _trec_run_to_negatives(path, source=source, label=label)
    if fmt in {"jsonl", "json"}:
        return _json_to_negatives(path, source=source, label=label)
    if fmt in {"table", "tsv", "csv"}:
        return _table_to_negatives(path, source=source, label=label)
    raise ValueError(f"Unsupported negative input format: {input_format}")


def _load_json_payload(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        return list(read_jsonl(path))
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        return json.loads(text)
    return list(read_jsonl(path))


def _looks_like_beir_run(payload: dict[str, Any]) -> bool:
    return bool(payload) and all(isinstance(value, dict) for value in payload.values())


def _row_query_id(
    row: dict[str, Any],
    row_idx: int,
    queries_by_text: dict[str, str],
    queries_by_index: dict[int, str],
) -> str | None:
    for key in QUERY_ID_KEYS:
        if key in row and row[key] is not None:
            return str(row[key])
    question = row.get("question") or row.get("query") or row.get("text")
    if question is not None:
        qid = queries_by_text.get(str(question).strip())
        if qid:
            return qid
    return queries_by_index.get(row_idx)


def _row_doc_id(row: dict[str, Any]) -> str | None:
    for key in DOC_ID_KEYS:
        if key in row and row[key] is not None:
            return str(row[key])
    return None


def _result_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in RESULT_LIST_KEYS:
        value = row.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _dedupe_rankings(rankings: dict[str, list[tuple[str, float]]]) -> dict[str, list[tuple[str, float]]]:
    deduped: dict[str, list[tuple[str, float]]] = {}
    for qid, rows in rankings.items():
        seen: set[str] = set()
        unique: list[tuple[str, float]] = []
        for doc_id, score in rows:
            if doc_id in seen:
                continue
            unique.append((doc_id, score))
            seen.add(doc_id)
        deduped[qid] = unique
    return deduped


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _trec_run_to_negatives(path: Path, source: str, label: str) -> list[NegativeRecord]:
    negatives: list[NegativeRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"{path}:{line_no} is not a valid TREC run row")
            qid, _, docid, rank_text, score_text, tag = parts[:6]
            negatives.append(
                NegativeRecord(
                    query_id=str(qid),
                    doc_id=str(docid),
                    source=source,
                    label=label,
                    rank=_int_or_none(rank_text),
                    score=_float_or_zero(score_text),
                    metadata={"external_tag": tag, "input_format": "trec"},
                )
            )
    return negatives


def _json_to_negatives(path: Path, source: str, label: str) -> list[NegativeRecord]:
    payload = _load_json_payload(path)
    rows = payload if isinstance(payload, list) else payload.get("rows") or payload.get("data") or [payload]
    negatives: list[NegativeRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        qid = _first_present(row, QUERY_ID_KEYS)
        doc_id = _first_present(row, DOC_ID_KEYS)
        if qid is None or doc_id is None:
            continue
        violation_types = row.get("violation_types", row.get("violations", []))
        if isinstance(violation_types, str):
            violation_types = [part.strip() for part in violation_types.split(",") if part.strip()]
        negatives.append(
            NegativeRecord(
                query_id=str(qid),
                doc_id=str(doc_id),
                source=str(row.get("source", source)),
                label=str(row.get("label", label)),
                violation_types=tuple(violation_types),
                evidence=str(row.get("evidence", "")),
                confidence=_float_or_zero(row.get("confidence", 0.0)),
                rank=_int_or_none(row.get("rank")),
                score=_float_or_zero(row.get("score")) if row.get("score") is not None else None,
                metadata=dict(row.get("metadata", {})) | {"input_format": "json"},
            )
        )
    return negatives


def _table_to_negatives(path: Path, source: str, label: str) -> list[NegativeRecord]:
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    negatives: list[NegativeRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            qid = _first_present(row, QUERY_ID_KEYS)
            doc_id = _first_present(row, DOC_ID_KEYS)
            if qid is None or doc_id is None:
                continue
            violations = row.get("violation_types", "")
            negatives.append(
                NegativeRecord(
                    query_id=str(qid),
                    doc_id=str(doc_id),
                    source=str(row.get("source") or source),
                    label=str(row.get("label") or label),
                    violation_types=tuple(part.strip() for part in violations.split(",") if part.strip()),
                    evidence=str(row.get("evidence") or ""),
                    confidence=_float_or_zero(row.get("confidence", 0.0)),
                    rank=_int_or_none(row.get("rank")),
                    score=_float_or_zero(row.get("score")) if row.get("score") else None,
                    metadata={"input_format": "table"},
                )
            )
    return negatives


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in row and row[key] is not None and row[key] != "":
            return row[key]
    return None
