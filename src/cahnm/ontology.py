from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .schemas import OntologyContext, OntologyNode, OntologyRelation
from .text import normalize

HIERARCHY_UP_TYPES = {"is_a", "broader", "parent"}
HIERARCHY_DOWN_TYPES = {"has_part", "narrower", "child"}
PREREQUISITE_TYPES = {"requires", "prerequisite"}
POSTREQUISITE_TYPES = {"enables", "postrequisite"}
RELATED_TYPES = {"related", "same_context"}


class Ontology:
    def __init__(self, nodes: list[OntologyNode], relations: list[OntologyRelation]):
        self.nodes = {node.id: node for node in nodes}
        self.relations = relations
        self.outgoing: dict[str, list[OntologyRelation]] = defaultdict(list)
        self.incoming: dict[str, list[OntologyRelation]] = defaultdict(list)
        self.label_index: dict[str, str] = {}
        for rel in relations:
            self.outgoing[rel.source].append(rel)
            self.incoming[rel.target].append(rel)
        for node in nodes:
            for label in node.labels():
                self.label_index[normalize(label)] = node.id

    @classmethod
    def load(cls, path: str | Path) -> "Ontology":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        nodes = [
            OntologyNode(
                id=str(row["id"]),
                label=str(row["label"]),
                aliases=tuple(row.get("aliases", [])),
                level=row.get("level"),
                metadata=dict(row.get("metadata", {})),
            )
            for row in payload.get("nodes", [])
        ]
        relations = [
            OntologyRelation(
                source=str(row["source"]),
                target=str(row["target"]),
                type=str(row["type"]),
                weight=float(row.get("weight", 1.0)),
                metadata=dict(row.get("metadata", {})),
            )
            for row in payload.get("relations", [])
        ]
        return cls(nodes, relations)

    def resolve(self, concept: str | None) -> OntologyNode | None:
        if not concept:
            return None
        if concept in self.nodes:
            return self.nodes[concept]
        return self.nodes.get(self.label_index.get(normalize(concept), ""))

    def find_in_text(self, text: str) -> OntologyNode | None:
        lowered = normalize(text)
        candidates = sorted(self.label_index.items(), key=lambda item: len(item[0]), reverse=True)
        for label, node_id in candidates:
            if label and label in lowered:
                return self.nodes[node_id]
        return None

    def without_relations(self) -> "Ontology":
        """Return a label-only placebo ontology with the same nodes and no graph edges."""
        return Ontology(list(self.nodes.values()), [])

    def shuffled_relations(self, random_seed: int = 13) -> "Ontology":
        """Return a deterministic graph placebo preserving per-type source and target marginals."""
        rng = random.Random(random_seed)
        by_type: dict[str, list[OntologyRelation]] = defaultdict(list)
        for relation in self.relations:
            by_type[relation.type].append(relation)

        shuffled: list[OntologyRelation] = []
        for relation_type in sorted(by_type):
            relations = sorted(
                by_type[relation_type],
                key=lambda item: (item.source, item.target, item.weight),
            )
            targets = [relation.target for relation in relations]
            rng.shuffle(targets)
            if len(targets) > 1:
                candidate_offsets = list(range(len(targets)))
                rng.shuffle(candidate_offsets)
                candidate_offsets = candidate_offsets[: min(64, len(candidate_offsets))]
                offset = min(
                    candidate_offsets,
                    key=lambda value: sum(
                        relation.source == targets[(index + value) % len(targets)]
                        for index, relation in enumerate(relations)
                    ),
                )
                targets = targets[offset:] + targets[:offset]

            for relation, target in zip(relations, targets):
                shuffled.append(
                    OntologyRelation(
                        source=relation.source,
                        target=target,
                        type=relation.type,
                        weight=relation.weight,
                        metadata={**relation.metadata, "placebo": "shuffled_target"},
                    )
                )
        return Ontology(list(self.nodes.values()), shuffled)

    def context(self, concept: str | None, max_depth: int = 2) -> OntologyContext:
        target = self.resolve(concept)
        if target is None:
            return OntologyContext(target=None)

        broader_ids = self._collect(
            target.id,
            max_depth=max_depth,
            outgoing_types=HIERARCHY_UP_TYPES,
            incoming_types=HIERARCHY_DOWN_TYPES,
        )
        narrower_ids = self._collect(
            target.id,
            max_depth=max_depth,
            outgoing_types=HIERARCHY_DOWN_TYPES,
            incoming_types=HIERARCHY_UP_TYPES,
        )
        prerequisite_ids = self._collect(
            target.id,
            max_depth=max_depth,
            outgoing_types=PREREQUISITE_TYPES,
            incoming_types=POSTREQUISITE_TYPES,
        )
        postrequisite_ids = self._collect(
            target.id,
            max_depth=max_depth,
            outgoing_types=POSTREQUISITE_TYPES,
            incoming_types=PREREQUISITE_TYPES,
        )
        related_ids = self._collect(
            target.id,
            max_depth=1,
            outgoing_types=RELATED_TYPES,
            incoming_types=RELATED_TYPES,
        )

        sibling_ids: set[str] = set()
        direct_parent_ids = self._collect(
            target.id,
            max_depth=1,
            outgoing_types=HIERARCHY_UP_TYPES,
            incoming_types=HIERARCHY_DOWN_TYPES,
        )
        for parent_id in direct_parent_ids:
            for rel in self.incoming[parent_id]:
                if rel.type in HIERARCHY_UP_TYPES and rel.source != target.id:
                    sibling_ids.add(rel.source)
            for rel in self.outgoing[parent_id]:
                if rel.type in HIERARCHY_DOWN_TYPES and rel.target != target.id:
                    sibling_ids.add(rel.target)

        return OntologyContext(
            target=target,
            broader=tuple(self.nodes[x] for x in sorted(broader_ids) if x in self.nodes),
            narrower=tuple(self.nodes[x] for x in sorted(narrower_ids) if x in self.nodes),
            siblings=tuple(self.nodes[x] for x in sorted(sibling_ids) if x in self.nodes),
            prerequisites=tuple(self.nodes[x] for x in sorted(prerequisite_ids) if x in self.nodes),
            postrequisites=tuple(self.nodes[x] for x in sorted(postrequisite_ids) if x in self.nodes),
            related=tuple(self.nodes[x] for x in sorted(related_ids) if x in self.nodes),
        )

    def _collect(
        self,
        start_id: str,
        max_depth: int,
        outgoing_types: set[str],
        incoming_types: set[str],
    ) -> set[str]:
        visited: set[str] = set()
        frontier = {start_id}
        for _ in range(max(0, max_depth)):
            next_frontier: set[str] = set()
            for node_id in frontier:
                for rel in self.outgoing[node_id]:
                    if rel.type in outgoing_types and rel.target not in visited and rel.target != start_id:
                        next_frontier.add(rel.target)
                for rel in self.incoming[node_id]:
                    if rel.type in incoming_types and rel.source not in visited and rel.source != start_id:
                        next_frontier.add(rel.source)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return visited


def make_sample_ontology() -> Ontology:
    nodes = [
        OntologyNode("sql_querying", "SQL querying", aliases=("SQL queries",), level="beginner"),
        OntologyNode("sql_joins", "SQL joins", aliases=("INNER JOIN", "LEFT JOIN", "joins"), level="beginner"),
        OntologyNode("sql_filtering", "SQL filtering", aliases=("WHERE clause",), level="beginner"),
        OntologyNode("sql_aggregation", "SQL aggregation", aliases=("GROUP BY",), level="intermediate"),
        OntologyNode("sql_subqueries", "SQL subqueries", aliases=("subquery",), level="intermediate"),
        OntologyNode("query_optimization", "SQL query optimization", aliases=("execution plans",), level="advanced"),
        OntologyNode("relational_keys", "relational keys", aliases=("primary keys", "foreign keys"), level="beginner"),
        OntologyNode("python_programming", "Python programming", aliases=("Python",), level="beginner"),
        OntologyNode("python_loops", "Python loops", aliases=("for loops", "while loops"), level="beginner"),
        OntologyNode("python_functions", "Python functions", aliases=("def function",), level="beginner"),
    ]
    relations = [
        OntologyRelation("sql_joins", "sql_querying", "is_a"),
        OntologyRelation("sql_filtering", "sql_querying", "is_a"),
        OntologyRelation("sql_aggregation", "sql_querying", "is_a"),
        OntologyRelation("sql_subqueries", "sql_querying", "is_a"),
        OntologyRelation("query_optimization", "sql_querying", "is_a"),
        OntologyRelation("sql_joins", "relational_keys", "requires"),
        OntologyRelation("query_optimization", "sql_joins", "requires"),
        OntologyRelation("python_loops", "python_programming", "is_a"),
        OntologyRelation("python_functions", "python_programming", "is_a"),
    ]
    return Ontology(nodes, relations)
