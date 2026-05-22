from __future__ import annotations

import json
import os
import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .schemas import ConstraintDecision, Document, OntologyContext, Query
from .text import contains_any, normalize, term_overlap

BEGINNER = {"beginner", "intro", "introduction", "basic", "fundamental", "first"}
INTERMEDIATE = {"intermediate", "practice", "applied"}
ADVANCED = {"advanced", "optimization", "expert", "performance", "architecture", "deep"}


class ConstraintJudge(Protocol):
    def classify(self, query: Query, document: Document, context: OntologyContext) -> ConstraintDecision:
        ...


def infer_level(text: str) -> str | None:
    tokens = set(normalize(text).split())
    if tokens & BEGINNER:
        return "beginner"
    if tokens & ADVANCED:
        return "advanced"
    if tokens & INTERMEDIATE:
        return "intermediate"
    return None


def node_labels(nodes) -> list[str]:
    labels: list[str] = []
    for node in nodes:
        labels.extend(node.labels())
    return labels


@dataclass
class HeuristicConstraintJudge:
    min_overlap_for_hardness: float = 0.04

    def classify(self, query: Query, document: Document, context: OntologyContext) -> ConstraintDecision:
        doc_text = document.searchable_text
        overlap = term_overlap(query.text, doc_text)
        if context.target is None:
            if overlap >= self.min_overlap_for_hardness:
                return ConstraintDecision("HardNeg", ("semantic_similarity_only",), "Text is close to the query.", 0.55)
            return ConstraintDecision("EasyNeg", (), "Low lexical or semantic overlap.", 0.65)

        target_hit = contains_any(doc_text, context.target.labels())
        sibling_hit = contains_any(doc_text, node_labels(context.siblings))
        prereq_hit = contains_any(doc_text, node_labels(context.prerequisites))
        postreq_hit = contains_any(doc_text, node_labels(context.postrequisites))
        broader_hit = contains_any(doc_text, node_labels(context.broader))
        related_hit = contains_any(doc_text, node_labels(context.related))
        narrower_hit = contains_any(doc_text, node_labels(context.narrower))

        q_level = infer_level(query.text)
        doc_level = infer_level(doc_text) or document.metadata.get("level")

        violations: list[str] = []
        evidence: list[str] = []

        if sibling_hit and not target_hit:
            violations.append("sibling_concept_confusion")
            evidence.append(f"mentions sibling concept '{sibling_hit}' rather than target '{context.target.label}'")

        if postreq_hit:
            violations.append("postrequisite_mismatch")
            evidence.append(f"mentions postrequisite concept '{postreq_hit}'")

        if q_level and doc_level and q_level != doc_level:
            if q_level == "beginner" and doc_level in {"intermediate", "advanced"}:
                violations.append("level_mismatch")
                evidence.append(f"query level is {q_level}, document level appears {doc_level}")
            elif q_level == "advanced" and doc_level == "beginner":
                violations.append("level_mismatch")
                evidence.append(f"query level is {q_level}, document level appears {doc_level}")

        if prereq_hit and not target_hit and overlap >= self.min_overlap_for_hardness:
            violations.append("prerequisite_mismatch")
            evidence.append(f"covers prerequisite '{prereq_hit}' but not the requested concept")

        if narrower_hit and not target_hit:
            violations.append("wrong_granularity")
            evidence.append(f"covers narrower concept '{narrower_hit}' without target evidence")

        if broader_hit and not target_hit and not sibling_hit:
            violations.append("wrong_granularity")
            evidence.append(f"covers broader concept '{broader_hit}' without target evidence")

        if related_hit and not target_hit and not violations:
            violations.append("context_mismatch")
            evidence.append(f"mentions related context '{related_hit}' without satisfying the target")

        if not target_hit and not violations and overlap >= self.min_overlap_for_hardness:
            violations.append("target_concept_mismatch")
            evidence.append(f"similar to query but missing target concept '{context.target.label}'")

        if target_hit and not violations:
            return ConstraintDecision("Positive", (), f"document mentions target evidence '{target_hit}'", 0.72)

        if target_hit and violations:
            return ConstraintDecision("Ambiguous", tuple(violations), "; ".join(evidence), 0.62)

        if violations and (overlap >= self.min_overlap_for_hardness or sibling_hit or postreq_hit or narrower_hit):
            confidence = 0.78 + min(0.17, overlap)
            return ConstraintDecision("HardNeg", tuple(dict.fromkeys(violations)), "; ".join(evidence), confidence)

        return ConstraintDecision("EasyNeg", (), "does not satisfy target and is not semantically close enough", 0.68)


class OpenAIConstraintJudge:
    """Optional LLM judge. Falls back to an exception if dependencies are absent."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        base_url: str | None = None,
        cache_path: str | Path | None = None,
        candidate_chars: int = 1200,
        context_items: int = 10,
        label_chars: int = 80,
        max_retries: int = 2,
        retry_sleep: float = 10.0,
    ):
        from openai import OpenAI

        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY") or ("lm-studio" if resolved_base_url else None)
        try:
            self.client = OpenAI(api_key=api_key, base_url=resolved_base_url)
        except TypeError as exc:
            if "proxies" in str(exc):
                raise RuntimeError(
                    "Incompatible openai/httpx package versions. Install the LLM dependencies with "
                    '`python3 -m pip install -e ".[llm]"` or run '
                    '`python3 -m pip install --upgrade "openai>=1.0" "httpx>=0.24,<0.28"`.'
                ) from exc
            raise
        self.model = model
        self.candidate_chars = candidate_chars
        self.context_items = context_items
        self.label_chars = label_chars
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, ConstraintDecision] = {}
        if self.cache_path and self.cache_path.exists():
            self._load_cache(self.cache_path)

    def classify(self, query: Query, document: Document, context: OntologyContext) -> ConstraintDecision:
        prompt = self._build_prompt(query, document, context, self.candidate_chars, self.context_items)
        cache_key = _cache_key(self.model, prompt)
        if cache_key in self.cache:
            return self.cache[cache_key]
        try:
            response = self._complete(prompt)
        except Exception as exc:
            if not _is_context_length_error(exc):
                raise
            prompt = self._build_prompt(query, document, context, candidate_chars=400, context_items=3)
            cache_key = _cache_key(self.model, prompt)
            if cache_key in self.cache:
                return self.cache[cache_key]
            response = self._complete(prompt)
        payload = _loads_json_object(response.choices[0].message.content or "{}")
        decision = ConstraintDecision(
            label=str(payload.get("label", "Ambiguous")),
            violation_types=tuple(payload.get("violation_types", [])),
            evidence=str(payload.get("evidence", "")),
            confidence=float(payload.get("confidence", 0.0)),
        )
        self.cache[cache_key] = decision
        self._append_cache(cache_key, decision)
        return decision

    def _build_prompt(
        self,
        query: Query,
        document: Document,
        context: OntologyContext,
        candidate_chars: int,
        context_items: int,
    ) -> dict:
        ontology_payload = {
            "target": _clip_text(context.target.label, self.label_chars) if context.target else None,
            "broader": _context_labels(context.broader, context_items, self.label_chars),
            "narrower": _context_labels(context.narrower, context_items, self.label_chars),
            "siblings": _context_labels(context.siblings, context_items, self.label_chars),
            "prerequisites": _context_labels(context.prerequisites, context_items, self.label_chars),
            "postrequisites": _context_labels(context.postrequisites, context_items, self.label_chars),
            "related": _context_labels(context.related, context_items, self.label_chars),
        }
        return {
            "query": _clip_text(query.text, 500),
            "candidate_title": _clip_text(document.title, 200),
            "candidate_text": _clip_text(document.text, candidate_chars),
            "ontology_context": ontology_payload,
            "allowed_labels": ["Positive", "EasyNeg", "HardNeg", "Ambiguous"],
            "allowed_violation_types": [
                "target_concept_mismatch",
                "sibling_concept_confusion",
                "prerequisite_mismatch",
                "postrequisite_mismatch",
                "level_mismatch",
                "wrong_granularity",
                "context_mismatch",
                "temporal_version_mismatch",
                "evidence_missing",
            ],
        }

    def _complete(self, prompt: dict):
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify retrieval candidates. Return only one valid JSON object with keys "
                    "label, violation_types, evidence, confidence. Use double quotes, keep evidence short, "
                    "and do not include markdown or any text outside the JSON object."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                try:
                    return self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        response_format={"type": "json_object"},
                        temperature=0,
                    )
                except Exception as exc:
                    if _is_context_length_error(exc) or _is_connection_error(exc):
                        raise
                    return self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0,
                    )
            except Exception as exc:
                last_exc = exc
                if _is_context_length_error(exc) or attempt >= self.max_retries:
                    raise
                if _is_connection_error(exc):
                    print(
                        f"LLM connection failed on attempt {attempt + 1}/{self.max_retries + 1}; "
                        f"retrying in {self.retry_sleep:g}s.",
                        flush=True,
                    )
                time.sleep(self.retry_sleep)
        raise last_exc or RuntimeError("LLM completion failed")

    def _load_cache(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    self.cache[str(row["key"])] = ConstraintDecision(
                        label=str(row.get("label", "Ambiguous")),
                        violation_types=tuple(row.get("violation_types", [])),
                        evidence=str(row.get("evidence", "")),
                        confidence=float(row.get("confidence", 0.0)),
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue

    def _append_cache(self, key: str, decision: ConstraintDecision) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "key": key,
            "label": decision.label,
            "violation_types": list(decision.violation_types),
            "evidence": decision.evidence,
            "confidence": decision.confidence,
        }
        with self.cache_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _loads_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    candidates = [text]
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            pass

        repaired = re.sub(
            r'(?<=[\]"}0-9])\s+(?="(?:label|violation_types|evidence|confidence)"\s*:)',
            ", ",
            candidate,
        )
        try:
            payload = json.loads(repaired)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            pass

    return {}


def _clip_text(text: str | None, limit: int) -> str:
    text = "" if text is None else str(text)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "..."


def _context_labels(nodes, limit: int, label_chars: int) -> list[str]:
    labels = []
    seen = set()
    for node in nodes:
        label = _clip_text(getattr(node, "label", ""), label_chars)
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
        if len(labels) >= limit:
            break
    return labels


def _is_context_length_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "context length" in text or "n_ctx" in text or "tokens to keep" in text


def _is_connection_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "connection refused" in text or "connection error" in text or "api connection" in text


def _cache_key(model: str, prompt: dict) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_judge(
    provider: str = "heuristic",
    model: str | None = None,
    base_url: str | None = None,
    cache_path: str | Path | None = None,
    candidate_chars: int = 1200,
    context_items: int = 10,
    label_chars: int = 80,
    max_retries: int = 2,
    retry_sleep: float = 10.0,
) -> ConstraintJudge:
    if provider == "heuristic":
        return HeuristicConstraintJudge()
    if provider in {"openai", "openai-compatible", "lmstudio"}:
        return OpenAIConstraintJudge(
            model=model or "gpt-4.1-mini",
            base_url=base_url,
            cache_path=cache_path,
            candidate_chars=candidate_chars,
            context_items=context_items,
            label_chars=label_chars,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
    raise ValueError(f"Unknown judge provider: {provider}")
