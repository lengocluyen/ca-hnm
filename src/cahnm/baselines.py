from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from .judges import ConstraintJudge, HeuristicConstraintJudge
from .miner import CAHNMiner, MiningConfig, positives_by_query
from .ontology import Ontology
from .retrievers import BM25Retriever, CandidatePool, DenseRetriever
from .schemas import Document, NegativeRecord, Qrel, Query
from .text import contains_any, term_overlap


CORE_STRATEGIES = [
    "DPR-Random",
    "DenseNeg",
    "CA-HNM-full",
    "CA-HNM-rank-matched",
]
MIXED_STRATEGIES = [
    "DPR-Random",
    "DenseNeg",
    "CA-HNM-mixed",
    "CA-HNM-matched-mixed",
]
STRUCTURAL_STRATEGIES = [
    "CA-HNM-full",
    "CA-HNM-label-only",
    "CA-HNM-shuffled-graph",
    "CA-HNM-no-ontology",
]
DEFAULT_STRATEGIES = CORE_STRATEGIES


@dataclass(frozen=True)
class BaselineConfig:
    max_negatives_per_query: int = 4
    top_k: int = 100
    random_seed: int = 13
    dense_model: str | None = None
    dense_backend: str = "auto"
    dense_batch_size: int = 8
    dense_device: str | None = None
    dense_max_seq_length: int | None = 128
    confidence_threshold: float = 0.75
    candidate_fusion: str = "max_score"


class NegativeStrategyRunner:
    def __init__(
        self,
        documents: list[Document],
        queries: list[Query],
        qrels: list[Qrel],
        ontology: Ontology,
        judge: ConstraintJudge | None = None,
        config: BaselineConfig | None = None,
    ):
        self.documents = documents
        self.doc_by_id = {doc.id: doc for doc in documents}
        self.queries = queries
        self.qrels = qrels
        self.ontology = ontology
        self.judge = judge or HeuristicConstraintJudge()
        self.config = config or BaselineConfig()
        self.positives = positives_by_query(qrels)
        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever(
            documents,
            model_name=self.config.dense_model,
            backend=self.config.dense_backend,
            batch_size=self.config.dense_batch_size,
            device=self.config.dense_device,
            max_seq_length=self.config.dense_max_seq_length,
        )
    def run(self, strategy: str) -> list[NegativeRecord]:
        if strategy == "DPR-Random":
            return self.random_negatives(source="DPR-Random", label="InBatchNeg", confidence=0.5)
        if strategy == "DenseNeg":
            return self.ranked_negatives("DenseNeg", self.dense)
        if strategy == "CA-HNM-full":
            return self.cahnm_negatives()
        if strategy == "CA-HNM-rank-matched":
            return self.cahnm_rank_matched_negatives()
        if strategy == "CA-HNM-mixed":
            return self.cahnm_mixed_negatives()
        if strategy == "CA-HNM-matched-mixed":
            return self.cahnm_matched_mixed_negatives()
        if strategy == "CA-HNM-label-only":
            return self.cahnm_negatives(source=strategy, ontology=self.ontology.without_relations())
        if strategy == "CA-HNM-shuffled-graph":
            return self.cahnm_negatives(
                source=strategy,
                ontology=self.ontology.shuffled_relations(self.config.random_seed),
            )
        if strategy == "CA-HNM-no-ontology":
            return self.cahnm_no_ontology_negatives()
        raise ValueError(f"Unknown strategy: {strategy}")

    def cahnm_negatives(
        self,
        source: str = "CA-HNM-full",
        ontology: Ontology | None = None,
    ) -> list[NegativeRecord]:
        miner = CAHNMiner(
            self.documents,
            ontology or self.ontology,
            judge=self.judge,
            config=MiningConfig(
                top_k_bm25=self.config.top_k,
                top_k_dense=self.config.top_k,
                max_negatives_per_query=self.config.max_negatives_per_query,
                confidence_threshold=self.config.confidence_threshold,
                random_seed=self.config.random_seed,
                dense_model=self.config.dense_model,
                dense_backend=self.config.dense_backend,
                dense_batch_size=self.config.dense_batch_size,
                dense_device=self.config.dense_device,
                dense_max_seq_length=self.config.dense_max_seq_length,
                source=source,
                context_depth=2,
                candidate_fusion=self.config.candidate_fusion,
            ),
        )
        return miner.mine(self.queries, self.qrels)

    def cahnm_mixed_negatives(self) -> list[NegativeRecord]:
        components = [
            self.random_negatives(source="random_component", label="EasyNeg", confidence=0.5),
            self.ranked_negatives("bm25_component", self.bm25, violation_type="lexical_hard_negative", confidence=0.58),
            self.ranked_negatives("dense_component", self.dense, violation_type="dense_hard_negative", confidence=0.6),
            self.cahnm_negatives(source="constraint_component"),
        ]
        return self._mix_negative_components(components, source="CA-HNM-mixed")

    def cahnm_matched_mixed_negatives(self) -> list[NegativeRecord]:
        constraint_negatives = self.cahnm_negatives(source="constraint_component_for_matching")
        matched_component = self._retrieval_rank_matched_placebos(constraint_negatives)
        components = [
            self.random_negatives(source="random_component", label="EasyNeg", confidence=0.5),
            self.ranked_negatives("bm25_component", self.bm25, violation_type="lexical_hard_negative", confidence=0.58),
            self.ranked_negatives("dense_component", self.dense, violation_type="dense_hard_negative", confidence=0.6),
            matched_component,
        ]
        return self._mix_negative_components(components, source="CA-HNM-matched-mixed")

    def cahnm_rank_matched_negatives(self) -> list[NegativeRecord]:
        """Build a one-for-one retrieval-hardness placebo for CA-HNM.

        Each constraint-selected negative defines a target retrieval source and
        rank. The control selects the closest unused non-positive from that source
        when possible and excludes every document selected by the treatment.
        """
        constraint_negatives = self.cahnm_negatives(source="constraint_component_for_matching")
        matched = self._retrieval_rank_matched_placebos(constraint_negatives)
        return [
            NegativeRecord(
                query_id=item.query_id,
                doc_id=item.doc_id,
                source="CA-HNM-rank-matched",
                label=item.label,
                violation_types=item.violation_types,
                evidence=item.evidence,
                confidence=item.confidence,
                rank=item.rank,
                score=item.score,
                metadata={**item.metadata, "causal_control": "retrieval-rank-and-source-matched"},
            )
            for item in matched
        ]

    def _retrieval_rank_matched_placebos(
        self,
        constraint_negatives: list[NegativeRecord],
    ) -> list[NegativeRecord]:
        pools = CandidatePool.union(
            self.queries,
            self.bm25,
            self.dense,
            top_k_bm25=self.config.top_k,
            top_k_dense=self.config.top_k,
            fusion=self.config.candidate_fusion,
        )
        targets_by_query: dict[str, list[NegativeRecord]] = defaultdict(list)
        excluded_by_query: dict[str, set[str]] = defaultdict(set)
        for negative in constraint_negatives:
            targets_by_query[negative.query_id].append(negative)
            excluded_by_query[negative.query_id].add(negative.doc_id)

        matched: list[NegativeRecord] = []
        for query in self.queries:
            used_docs = set(excluded_by_query.get(query.id, set()))
            used_docs.update(self.positives.get(query.id, set()))
            ranked_pool = [
                (rank, doc_id, score, source)
                for rank, (doc_id, score, source) in enumerate(
                    pools.rankings.get(query.id, []), start=1
                )
                if doc_id not in used_docs
            ]
            for target in targets_by_query.get(query.id, []):
                if not ranked_pool:
                    break
                target_rank = int(target.rank or 1)
                target_source = str(target.metadata.get("retrieval_source", ""))
                best_index = min(
                    range(len(ranked_pool)),
                    key=lambda index: (
                        ranked_pool[index][3] != target_source,
                        abs(ranked_pool[index][0] - target_rank),
                        ranked_pool[index][0],
                        ranked_pool[index][1],
                    ),
                )
                rank, doc_id, score, retrieval_source = ranked_pool.pop(best_index)
                used_docs.add(doc_id)
                matched.append(
                    NegativeRecord(
                        query_id=query.id,
                        doc_id=doc_id,
                        source="matched_constraint_component",
                        label="HardNeg",
                        violation_types=("matched_retrieval_placebo",),
                        evidence="constraint-unaware candidate matched to a constraint negative's retrieval rank",
                        confidence=target.confidence,
                        rank=rank,
                        score=score,
                        metadata={
                            "retrieval_source": retrieval_source,
                            "matched_target_rank": target_rank,
                            "matched_target_source": target_source,
                            "rank_distance": abs(rank - target_rank),
                            "matching_policy": "source_then_absolute_rank",
                        },
                    )
                )
        return matched

    def _mix_negative_components(
        self,
        components: list[list[NegativeRecord]],
        source: str,
    ) -> list[NegativeRecord]:
        by_query: dict[str, dict[str, list[NegativeRecord]]] = defaultdict(lambda: defaultdict(list))
        for negatives in components:
            for negative in negatives:
                by_query[negative.query_id][negative.source].append(negative)

        mixed: list[NegativeRecord] = []
        for query in self.queries:
            buckets = by_query.get(query.id, {})
            keys = sorted(buckets)
            selected = 0
            seen_docs: set[str] = set()
            while selected < self.config.max_negatives_per_query and keys:
                next_keys = []
                for key in keys:
                    bucket = buckets[key]
                    while bucket and bucket[0].doc_id in seen_docs:
                        bucket.pop(0)
                    if bucket and selected < self.config.max_negatives_per_query:
                        negative = bucket.pop(0)
                        seen_docs.add(negative.doc_id)
                        mixed.append(
                            NegativeRecord(
                                query_id=negative.query_id,
                                doc_id=negative.doc_id,
                                source=source,
                                label=negative.label,
                                violation_types=negative.violation_types,
                                evidence=negative.evidence,
                                confidence=negative.confidence,
                                rank=negative.rank,
                                score=negative.score,
                                metadata={**negative.metadata, "component": key},
                            )
                        )
                        selected += 1
                    if bucket:
                        next_keys.append(key)
                keys = next_keys
        return mixed

    def cahnm_no_ontology_negatives(self) -> list[NegativeRecord]:
        negatives: list[NegativeRecord] = []
        pools = CandidatePool.union(
            self.queries,
            self.bm25,
            self.dense,
            top_k_bm25=self.config.top_k,
            top_k_dense=self.config.top_k,
        )
        for query in self.queries:
            selected = 0
            for rank, (doc_id, score, retrieval_source) in enumerate(pools.rankings.get(query.id, []), start=1):
                if doc_id in self.positives.get(query.id, set()):
                    continue
                negatives.append(
                    NegativeRecord(
                        query_id=query.id,
                        doc_id=doc_id,
                        source="CA-HNM-no-ontology",
                        label="HardNeg",
                        violation_types=("semantic_similarity_only",),
                        evidence="selected from retrieval pool without ontology constraints",
                        confidence=0.55,
                        rank=rank,
                        score=score,
                        metadata={"retrieval_source": retrieval_source},
                    )
                )
                selected += 1
                if selected >= self.config.max_negatives_per_query:
                    break
        return negatives

    def random_negatives(self, source: str = "RandomNeg", label: str = "EasyNeg", confidence: float = 0.5) -> list[NegativeRecord]:
        negatives: list[NegativeRecord] = []
        doc_ids = [doc.id for doc in self.documents]
        for query in self.queries:
            available = [doc_id for doc_id in doc_ids if doc_id not in self.positives.get(query.id, set())]
            # Query-local RNG makes sampling invariant to strategy call order.
            seed_payload = f"{self.config.random_seed}|{source}|{query.id}".encode("utf-8")
            local_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
            random.Random(local_seed).shuffle(available)
            for rank, doc_id in enumerate(available[: self.config.max_negatives_per_query], start=1):
                negatives.append(
                    NegativeRecord(
                        query_id=query.id,
                        doc_id=doc_id,
                        source=source,
                        label=label,
                        rank=rank,
                        confidence=confidence,
                        metadata={"related_work": source == "DPR-Random"},
                    )
                )
        return negatives

    def ranked_negatives(
        self,
        source: str,
        retriever,
        violation_type: str = "similarity_only",
        evidence: str | None = None,
        confidence: float = 0.6,
    ) -> list[NegativeRecord]:
        negatives: list[NegativeRecord] = []
        for query in self.queries:
            selected = 0
            for rank, (doc_id, score) in enumerate(retriever.rank(query.text, self.config.top_k), start=1):
                if doc_id in self.positives.get(query.id, set()):
                    continue
                negatives.append(
                    NegativeRecord(
                        query_id=query.id,
                        doc_id=doc_id,
                        source=source,
                        label="HardNeg",
                        violation_types=(violation_type,),
                        evidence=evidence or f"selected by {source} rank",
                        rank=rank,
                        score=score,
                        confidence=confidence,
                    )
                )
                selected += 1
                if selected >= self.config.max_negatives_per_query:
                    break
        return negatives

def summarize_negative_quality(
    negatives_by_strategy: dict[str, list[NegativeRecord]],
    documents: list[Document],
    queries: list[Query],
    qrels: list[Qrel],
    ontology: Ontology,
    judge: ConstraintJudge | None = None,
) -> list[dict]:
    judge = judge or HeuristicConstraintJudge()
    doc_by_id = {doc.id: doc for doc in documents}
    query_by_id = {query.id: query for query in queries}
    positives = positives_by_query(qrels)
    rows: list[dict] = []

    for strategy, negatives in negatives_by_strategy.items():
        hard_valid = 0
        false_negative = 0
        target_leakage = 0
        confidence_total = 0.0
        overlap_total = 0.0
        violations: Counter[str] = Counter()
        for neg in negatives:
            query = query_by_id[neg.query_id]
            doc = doc_by_id[neg.doc_id]
            if neg.doc_id in positives.get(neg.query_id, set()):
                false_negative += 1
            concept = query.target_concept or (ontology.find_in_text(query.text).id if ontology.find_in_text(query.text) else None)
            context = ontology.context(concept)
            overlap_total += term_overlap(query.text, doc.searchable_text)
            confidence_total += neg.confidence
            if context.target and contains_any(doc.searchable_text, context.target.labels()):
                target_leakage += 1
            decision = judge.classify(query, doc, context)
            if decision.is_hard_negative:
                hard_valid += 1
                violations.update(decision.violation_types)
        count = len(negatives)
        rows.append(
            {
                "strategy": strategy,
                "negatives": count,
                "valid_hard_rate": hard_valid / count if count else 0.0,
                "false_negative_rate": false_negative / count if count else 0.0,
                "target_leakage_rate": target_leakage / count if count else 0.0,
                "avg_query_doc_overlap": overlap_total / count if count else 0.0,
                "avg_confidence": confidence_total / count if count else 0.0,
                "violation_distribution": dict(violations),
            }
        )
    return rows
