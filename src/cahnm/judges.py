from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .schemas import ConstraintDecision, Document, OntologyContext, Query
from .text import contains_any, normalize, term_overlap


BEGINNER = {"beginner", "intro", "introduction", "basic", "fundamental", "first"}
INTERMEDIATE = {"intermediate", "practice", "applied"}
ADVANCED = {"advanced", "optimization", "expert", "performance", "architecture", "deep"}


class ConstraintJudge(Protocol):
    def classify(
        self, query: Query, document: Document, context: OntologyContext
    ) -> ConstraintDecision:
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
    """Deterministic structured-domain classifier used in the paper runs."""

    min_overlap_for_hardness: float = 0.04

    def classify(
        self, query: Query, document: Document, context: OntologyContext
    ) -> ConstraintDecision:
        doc_text = document.searchable_text
        overlap = term_overlap(query.text, doc_text)
        if context.target is None:
            if overlap >= self.min_overlap_for_hardness:
                return ConstraintDecision(
                    "HardNeg",
                    ("semantic_similarity_only",),
                    "Text is close to the query.",
                    0.55,
                )
            return ConstraintDecision(
                "EasyNeg", (), "Low lexical or semantic overlap.", 0.65
            )

        target_hit = contains_any(doc_text, context.target.labels())
        sibling_hit = contains_any(doc_text, node_labels(context.siblings))
        prereq_hit = contains_any(doc_text, node_labels(context.prerequisites))
        postreq_hit = contains_any(doc_text, node_labels(context.postrequisites))
        broader_hit = contains_any(doc_text, node_labels(context.broader))
        related_hit = contains_any(doc_text, node_labels(context.related))
        narrower_hit = contains_any(doc_text, node_labels(context.narrower))

        query_level = infer_level(query.text)
        document_level = infer_level(doc_text) or document.metadata.get("level")
        violations: list[str] = []
        evidence: list[str] = []

        if sibling_hit and not target_hit:
            violations.append("sibling_concept_confusion")
            evidence.append(
                f"mentions sibling concept '{sibling_hit}' rather than target "
                f"'{context.target.label}'"
            )
        if postreq_hit:
            violations.append("postrequisite_mismatch")
            evidence.append(f"mentions postrequisite concept '{postreq_hit}'")
        if query_level and document_level and query_level != document_level:
            if query_level == "beginner" and document_level in {
                "intermediate",
                "advanced",
            }:
                violations.append("level_mismatch")
                evidence.append(
                    f"query level is {query_level}, document level appears {document_level}"
                )
            elif query_level == "advanced" and document_level == "beginner":
                violations.append("level_mismatch")
                evidence.append(
                    f"query level is {query_level}, document level appears {document_level}"
                )
        if (
            prereq_hit
            and not target_hit
            and overlap >= self.min_overlap_for_hardness
        ):
            violations.append("prerequisite_mismatch")
            evidence.append(
                f"covers prerequisite '{prereq_hit}' but not the requested concept"
            )
        if narrower_hit and not target_hit:
            violations.append("wrong_granularity")
            evidence.append(
                f"covers narrower concept '{narrower_hit}' without target evidence"
            )
        if broader_hit and not target_hit and not sibling_hit:
            violations.append("wrong_granularity")
            evidence.append(
                f"covers broader concept '{broader_hit}' without target evidence"
            )
        if related_hit and not target_hit and not violations:
            violations.append("context_mismatch")
            evidence.append(
                f"mentions related context '{related_hit}' without satisfying the target"
            )
        if (
            not target_hit
            and not violations
            and overlap >= self.min_overlap_for_hardness
        ):
            violations.append("target_concept_mismatch")
            evidence.append(
                f"similar to query but missing target concept '{context.target.label}'"
            )

        if target_hit and not violations:
            return ConstraintDecision(
                "Positive", (), f"document mentions target evidence '{target_hit}'", 0.72
            )
        if target_hit and violations:
            return ConstraintDecision(
                "Ambiguous", tuple(violations), "; ".join(evidence), 0.62
            )
        if violations and (
            overlap >= self.min_overlap_for_hardness
            or sibling_hit
            or postreq_hit
            or narrower_hit
        ):
            confidence = 0.78 + min(0.17, overlap)
            return ConstraintDecision(
                "HardNeg",
                tuple(dict.fromkeys(violations)),
                "; ".join(evidence),
                confidence,
            )
        return ConstraintDecision(
            "EasyNeg",
            (),
            "does not satisfy target and is not semantically close enough",
            0.68,
        )


def make_judge(provider: str = "heuristic") -> ConstraintJudge:
    if provider != "heuristic":
        raise ValueError(
            "The public paper artifact exposes only the deterministic heuristic judge"
        )
    return HeuristicConstraintJudge()
