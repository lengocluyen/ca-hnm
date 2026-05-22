#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "Classify retrieval candidates. Return only one valid JSON object with keys "
    "label, violation_types, evidence, confidence. Use double quotes, keep evidence short, "
    "and do not include markdown or any text outside the JSON object."
)

ALLOWED_LABELS = ["Positive", "EasyNeg", "HardNeg", "Ambiguous"]
ALLOWED_VIOLATION_TYPES = [
    "target_concept_mismatch",
    "sibling_concept_confusion",
    "prerequisite_mismatch",
    "postrequisite_mismatch",
    "level_mismatch",
    "wrong_granularity",
    "context_mismatch",
    "temporal_version_mismatch",
    "evidence_missing",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM validation on a human-validation CSV sheet.")
    parser.add_argument("--sheet", required=True, help="CSV produced by create_human_validation_sheet.py")
    parser.add_argument("--out", required=True, help="CSV with appended LLM judgment columns")
    parser.add_argument("--decisions-out", help="Optional JSONL with one LLM decision per row")
    parser.add_argument("--llm-model", default="gpt-oss-20b")
    parser.add_argument("--llm-base-url", default="http://localhost:1234/v1")
    parser.add_argument("--llm-cache", help="Optional JSONL cache")
    parser.add_argument("--max-rows", type=int, help="Debug/smoke-test limit")
    parser.add_argument("--candidate-chars", type=int, default=1200)
    parser.add_argument("--context-chars", type=int, default=1200)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    args = parser.parse_args()

    from openai import OpenAI

    client = OpenAI(api_key="lm-studio", base_url=args.llm_base_url)
    cache_path = Path(args.llm_cache) if args.llm_cache else None
    cache = _load_cache(cache_path)

    rows = _read_rows(Path(args.sheet))
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    out_rows = []
    decision_rows = []
    for idx, row in enumerate(rows, start=1):
        prompt = _build_prompt(row, args.candidate_chars, args.context_chars)
        key = _cache_key(args.llm_model, prompt)
        if key in cache:
            decision = cache[key]
        else:
            decision = _complete(client, args.llm_model, prompt, args.retries, args.retry_sleep)
            cache[key] = decision
            _append_cache(cache_path, key, decision)
        out_row = dict(row)
        out_row.update(
            {
                "llm_label": str(decision.get("label", "Ambiguous")),
                "llm_violation_types": ",".join(str(v) for v in decision.get("violation_types", [])),
                "llm_evidence": str(decision.get("evidence", "")),
                "llm_confidence": str(decision.get("confidence", "")),
            }
        )
        out_rows.append(out_row)
        decision_rows.append(
            {
                "row_index": idx,
                "item_id": row.get("item_id", ""),
                "query_id": row.get("query_id", ""),
                "doc_id": row.get("doc_id", ""),
                "strategy": row.get("strategy", ""),
                "system_label": row.get("system_label", ""),
                "system_violations": row.get("system_violations", ""),
                **out_row,
            }
        )

    _write_csv(Path(args.out), out_rows)
    if args.decisions_out:
        _write_jsonl(Path(args.decisions_out), decision_rows)
    print(f"wrote LLM-validated sheet to {args.out}")


def _build_prompt(row: dict[str, str], candidate_chars: int, context_chars: int) -> dict[str, Any]:
    return {
        "query": _clip(row.get("query_text", ""), 500),
        "candidate_title": _clip(row.get("doc_title", ""), 200),
        "candidate_text": _clip(row.get("doc_text_snippet", ""), candidate_chars),
        "ontology_context": _parse_context(_clip(row.get("ontology_context", ""), context_chars)),
        "allowed_labels": ALLOWED_LABELS,
        "allowed_violation_types": ALLOWED_VIOLATION_TYPES,
    }


def _parse_context(text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "target": None,
        "broader": [],
        "narrower": [],
        "siblings": [],
        "prerequisites": [],
        "postrequisites": [],
        "related": [],
    }
    mapping = {
        "target": "target",
        "broader": "broader",
        "narrower": "narrower",
        "siblings": "siblings",
        "prereq": "prerequisites",
        "postreq": "postrequisites",
        "related": "related",
    }
    for part in text.split("|"):
        if "=" not in part:
            continue
        key, value = [item.strip() for item in part.split("=", 1)]
        mapped = mapping.get(key)
        if not mapped:
            continue
        if mapped == "target":
            payload[mapped] = value
        else:
            payload[mapped] = [item.strip() for item in value.split(";") if item.strip()]
    return payload


def _complete(client, model: str, prompt: dict[str, Any], retries: int, retry_sleep: float) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0,
                )
            except Exception:
                response = client.chat.completions.create(model=model, messages=messages, temperature=0)
            return _loads_json_object(response.choices[0].message.content or "{}")
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(retry_sleep)
    raise RuntimeError(f"LLM validation failed after {retries + 1} attempts") from last_exc


def _loads_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        payload = json.loads(text[start : end + 1])
    return payload if isinstance(payload, dict) else {}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path or not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[str(row["key"])] = dict(row["decision"])
    return cache


def _append_cache(path: Path | None, key: str, decision: dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"key": key, "decision": decision}, ensure_ascii=False, sort_keys=True) + "\n")


def _cache_key(model: str, prompt: dict[str, Any]) -> str:
    payload = json.dumps({"model": model, "prompt": prompt}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clip(text: str, max_chars: int) -> str:
    text = str(text or "")
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


if __name__ == "__main__":
    main()
