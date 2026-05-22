from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

from .judges import ConstraintJudge, HeuristicConstraintJudge
from .ontology import Ontology
from .retrievers import BM25Retriever, CandidatePool, DenseRetriever
from .schemas import Document, NegativeRecord, Qrel, Query


@dataclass(frozen=True)
class MiningConfig:
    top_k_bm25: int = 50
    top_k_dense: int = 50
    max_negatives_per_query: int = 8
    confidence_threshold: float = 0.75
    balance_by_violation: bool = True
    random_seed: int = 13
    dense_model: str | None = None
    dense_backend: str = "auto"
    dense_batch_size: int = 16
    dense_device: str | None = None
    dense_max_seq_length: int | None = None
    source: str = "CA-HNM"
    context_depth: int = 2
    exclude_positives: bool = True
    allowed_violation_types: tuple[str, ...] = ()
    candidate_fusion: str = "max_score"
    selection_policy: str = "balanced"
    retrieval_weight: float = 0.55
    constraint_weight: float = 0.30
    ontology_weight: float = 0.15
    diversity_penalty: float = 0.06


def positives_by_query(qrels: list[Qrel]) -> dict[str, set[str]]:
    positives: dict[str, set[str]] = defaultdict(set)
    for qrel in qrels:
        if qrel.relevance > 0:
            positives[qrel.query_id].add(qrel.doc_id)
    return positives


class CAHNMiner:
    def __init__(
        self,
        documents: list[Document],
        ontology: Ontology,
        judge: ConstraintJudge | None = None,
        config: MiningConfig | None = None,
    ):
        self.documents = documents
        self.doc_by_id = {doc.id: doc for doc in documents}
        self.ontology = ontology
        self.judge = judge or HeuristicConstraintJudge()
        self.config = config or MiningConfig()
        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever(
            documents,
            model_name=self.config.dense_model,
            backend=self.config.dense_backend,
            batch_size=self.config.dense_batch_size,
            device=self.config.dense_device,
            max_seq_length=self.config.dense_max_seq_length,
        )

    def mine(self, queries: list[Query], qrels: list[Qrel]) -> list[NegativeRecord]:
        positives = positives_by_query(qrels)
        pools = CandidatePool.union(
            queries,
            self.bm25,
            self.dense,
            top_k_bm25=self.config.top_k_bm25,
            top_k_dense=self.config.top_k_dense,
            fusion=self.config.candidate_fusion,
        )
        results: list[NegativeRecord] = []
        rng = random.Random(self.config.random_seed)

        for query in queries:
            concept = query.target_concept
            if concept is None:
                inferred = self.ontology.find_in_text(query.text)
                concept = inferred.id if inferred else None
            context = self.ontology.context(concept, max_depth=self.config.context_depth)

            candidates: list[NegativeRecord] = []
            for rank, (doc_id, score, source) in enumerate(pools.rankings.get(query.id, []), start=1):
                if self.config.exclude_positives and doc_id in positives.get(query.id, set()):
                    continue
                document = self.doc_by_id[doc_id]
                decision = self.judge.classify(query, document, context)
                violation_types = decision.violation_types
                if self.config.allowed_violation_types:
                    allowed = set(self.config.allowed_violation_types)
                    violation_types = tuple(item for item in violation_types if item in allowed)
                if (
                    decision.label == "HardNeg"
                    and violation_types
                    and decision.confidence >= self.config.confidence_threshold
                ):
                    candidates.append(
                        NegativeRecord(
                            query_id=query.id,
                            doc_id=doc_id,
                            source=self.config.source,
                            label=decision.label,
                            violation_types=violation_types,
                            evidence=decision.evidence,
                            confidence=decision.confidence,
                            rank=rank,
                            score=score,
                            metadata={
                                "retrieval_source": source,
                                "target_concept": context.target.id if context.target else None,
                                "candidate_fusion": self.config.candidate_fusion,
                                "selection_policy": self.config.selection_policy,
                            },
                        )
                    )

            results.extend(self._select_candidates(candidates, rng))

        return results

    def _select_candidates(self, candidates: list[NegativeRecord], rng: random.Random) -> list[NegativeRecord]:
        if self.config.selection_policy == "retrieval_aware":
            return self._select_retrieval_aware(candidates)
        if self.config.selection_policy == "balanced":
            return self._select_balanced(candidates, rng)
        if self.config.selection_policy == "top_ranked":
            return candidates[: self.config.max_negatives_per_query]
        raise ValueError(f"Unknown selection policy: {self.config.selection_policy}")

    def _select_balanced(self, candidates: list[NegativeRecord], rng: random.Random) -> list[NegativeRecord]:
        if len(candidates) <= self.config.max_negatives_per_query:
            return candidates
        if not self.config.balance_by_violation:
            return candidates[: self.config.max_negatives_per_query]

        buckets: dict[str, list[NegativeRecord]] = defaultdict(list)
        for item in candidates:
            key = item.violation_types[0] if item.violation_types else "unknown"
            buckets[key].append(item)

        selected: list[NegativeRecord] = []
        keys = sorted(buckets)
        while len(selected) < self.config.max_negatives_per_query and keys:
            next_keys: list[str] = []
            for key in keys:
                bucket = buckets[key]
                if bucket and len(selected) < self.config.max_negatives_per_query:
                    selected.append(bucket.pop(0))
                if bucket:
                    next_keys.append(key)
            keys = next_keys

        if len(selected) < self.config.max_negatives_per_query:
            remaining = [item for bucket in buckets.values() for item in bucket]
            rng.shuffle(remaining)
            selected.extend(remaining[: self.config.max_negatives_per_query - len(selected)])

        return selected

    def _select_retrieval_aware(self, candidates: list[NegativeRecord]) -> list[NegativeRecord]:
        if len(candidates) <= self.config.max_negatives_per_query:
            return [
                self._with_v2_score(item, item.score or 0.0, selection_rank=rank)
                for rank, item in enumerate(candidates, start=1)
            ]

        raw_scores = [float(item.score or 0.0) for item in candidates]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        score_range = max(max_score - min_score, 1e-9)

        def base_score(item: NegativeRecord) -> float:
            retrieval = (float(item.score or 0.0) - min_score) / score_range
            constraint = min(1.0, max(0.0, item.confidence))
            ontology = _ontology_hardness(item.violation_types)
            return (
                self.config.retrieval_weight * retrieval
                + self.config.constraint_weight * constraint
                + self.config.ontology_weight * ontology
            )

        scored = [(base_score(item), item) for item in candidates]
        selected: list[NegativeRecord] = []
        violation_counts: dict[str, int] = defaultdict(int)
        used_docs: set[str] = set()

        while len(selected) < self.config.max_negatives_per_query and scored:
            best_idx = 0
            best_score = float("-inf")
            for idx, (score, item) in enumerate(scored):
                primary_violation = item.violation_types[0] if item.violation_types else "unknown"
                adjusted = score - self.config.diversity_penalty * violation_counts[primary_violation]
                if adjusted > best_score:
                    best_idx = idx
                    best_score = adjusted
            base, item = scored.pop(best_idx)
            if item.doc_id in used_docs:
                continue
            used_docs.add(item.doc_id)
            primary = item.violation_types[0] if item.violation_types else "unknown"
            violation_counts[primary] += 1
            selected.append(self._with_v2_score(item, base, selection_rank=len(selected) + 1))

        return selected

    def _with_v2_score(self, item: NegativeRecord, final_score: float, selection_rank: int) -> NegativeRecord:
        metadata = dict(item.metadata)
        metadata["retrieval_aware_score"] = final_score
        metadata["original_retrieval_score"] = item.score
        return NegativeRecord(
            query_id=item.query_id,
            doc_id=item.doc_id,
            source=item.source,
            label=item.label,
            violation_types=item.violation_types,
            evidence=item.evidence,
            confidence=item.confidence,
            rank=selection_rank,
            score=final_score,
            metadata=metadata,
        )


def _ontology_hardness(violation_types: tuple[str, ...]) -> float:
    weights = {
        "target_concept_mismatch": 1.00,
        "sibling_concept_confusion": 0.95,
        "wrong_granularity": 0.90,
        "postrequisite_mismatch": 0.85,
        "prerequisite_mismatch": 0.80,
        "level_mismatch": 0.75,
        "context_mismatch": 0.70,
        "semantic_similarity_only": 0.55,
    }
    if not violation_types:
        return 0.0
    return max(weights.get(item, 0.60) for item in violation_types)
