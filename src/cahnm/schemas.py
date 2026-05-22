from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    metadata: JsonDict = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        return f"{self.title}\n{self.text}".strip()


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    target_concept: str | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class Qrel:
    query_id: str
    doc_id: str
    relevance: int = 1


@dataclass(frozen=True)
class OntologyNode:
    id: str
    label: str
    aliases: tuple[str, ...] = ()
    level: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def labels(self) -> list[str]:
        return [self.label, *self.aliases]


@dataclass(frozen=True)
class OntologyRelation:
    source: str
    target: str
    type: str
    weight: float = 1.0
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class OntologyContext:
    target: OntologyNode | None
    broader: tuple[OntologyNode, ...] = ()
    narrower: tuple[OntologyNode, ...] = ()
    siblings: tuple[OntologyNode, ...] = ()
    prerequisites: tuple[OntologyNode, ...] = ()
    postrequisites: tuple[OntologyNode, ...] = ()
    related: tuple[OntologyNode, ...] = ()

    def all_nodes(self) -> list[OntologyNode]:
        nodes: list[OntologyNode] = []
        if self.target:
            nodes.append(self.target)
        for group in (
            self.broader,
            self.narrower,
            self.siblings,
            self.prerequisites,
            self.postrequisites,
            self.related,
        ):
            nodes.extend(group)
        return nodes


@dataclass(frozen=True)
class ConstraintDecision:
    label: str
    violation_types: tuple[str, ...] = ()
    evidence: str = ""
    confidence: float = 0.0

    @property
    def is_hard_negative(self) -> bool:
        return self.label == "HardNeg" and bool(self.violation_types)


@dataclass(frozen=True)
class NegativeRecord:
    query_id: str
    doc_id: str
    source: str
    label: str
    violation_types: tuple[str, ...] = ()
    evidence: str = ""
    confidence: float = 0.0
    rank: int | None = None
    score: float | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalRun:
    name: str
    rankings: dict[str, list[tuple[str, float]]]
