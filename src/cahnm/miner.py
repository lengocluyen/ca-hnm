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
    top_k_bm25: int = 100
    top_k_dense: int = 100
    max_negatives_per_query: int = 4
    confidence_threshold: float = 0.75
    balance_by_violation: bool = True
    random_seed: int = 13
    dense_model: str | None = None
    dense_backend: str = "auto"
    dense_batch_size: int = 8
    dense_device: str | None = None
    dense_max_seq_length: int | None = 128
    source: str = "CA-HNM-full"
    context_depth: int = 2
    candidate_fusion: str = "max_score"


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
                if doc_id in positives.get(query.id, set()):
                    continue
                document = self.doc_by_id[doc_id]
                decision = self.judge.classify(query, document, context)
                violation_types = decision.violation_types
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
                                "selection_policy": "balanced",
                            },
                        )
                    )

            results.extend(self._select_balanced(candidates, rng))

        return results

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
