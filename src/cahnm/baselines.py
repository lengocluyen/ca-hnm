from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from .judges import ConstraintJudge, HeuristicConstraintJudge
from .miner import CAHNMiner, MiningConfig, positives_by_query
from .ontology import Ontology
from .retrievers import BM25Retriever, CandidatePool, DenseRetriever
from .schemas import Document, NegativeRecord, Qrel, Query
from .text import contains_any, term_overlap


CORE_STRATEGIES = ["RandomNeg", "BM25Neg", "DenseNeg", "OntoNeg", "LLMNeg", "CA-HNM"]
RELATED_WORK_STRATEGIES = [
    "DPR-Random",
    "ANCE",
    "ADORE",
    "RocketQA-Denoised",
    "TAS-Balanced",
    "GPL-Pseudo",
    "SyNeg",
]
ABLATION_STRATEGIES = [
    "CA-HNM-v2",
    "CA-HNM-v2-mixed",
    "CA-HNM-full",
    "CA-HNM-mixed",
    "CA-HNM-no-ontology",
    "CA-HNM-no-reasoning",
    "CA-HNM-no-fn-filter",
    "CA-HNM-prereq-only",
    "CA-HNM-sibling-only",
]
DEFAULT_STRATEGIES = [
    "RandomNeg",
    "BM25Neg",
    "DenseNeg",
    *RELATED_WORK_STRATEGIES,
    "OntoNeg",
    "LLMNeg",
    "CA-HNM",
]


@dataclass(frozen=True)
class BaselineConfig:
    max_negatives_per_query: int = 8
    top_k: int = 50
    random_seed: int = 13
    dense_model: str | None = None
    dense_backend: str = "auto"
    dense_batch_size: int = 16
    dense_device: str | None = None
    dense_max_seq_length: int | None = None
    confidence_threshold: float = 0.75
    candidate_fusion: str = "max_score"
    selection_policy: str = "balanced"
    retrieval_weight: float = 0.55
    constraint_weight: float = 0.30
    ontology_weight: float = 0.15
    diversity_penalty: float = 0.06


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
        self.rng = random.Random(self.config.random_seed)

    def run(self, strategy: str) -> list[NegativeRecord]:
        if strategy == "RandomNeg":
            return self.random_negatives()
        if strategy == "DPR-Random":
            return self.random_negatives(source="DPR-Random", label="InBatchNeg", confidence=0.5)
        if strategy == "BM25Neg":
            return self.ranked_negatives("BM25Neg", self.bm25)
        if strategy == "DenseNeg":
            return self.ranked_negatives("DenseNeg", self.dense)
        if strategy == "ANCE":
            return self.ranked_negatives(
                "ANCE",
                self.dense,
                violation_type="ann_dense_hard_negative",
                evidence="selected as an approximate-nearest-neighbor dense hard negative",
                confidence=0.62,
            )
        if strategy == "ADORE":
            return self.adore_negatives()
        if strategy == "RocketQA-Denoised":
            return self.rocketqa_denoised_negatives()
        if strategy == "TAS-Balanced":
            return self.tas_balanced_negatives()
        if strategy == "GPL-Pseudo":
            return self.gpl_pseudo_negatives()
        if strategy == "SyNeg":
            return self.llm_only_negatives(source="SyNeg", violation_type="llm_synthetic_style_negative")
        if strategy == "OntoNeg":
            return self.ontology_negatives()
        if strategy == "LLMNeg":
            return self.llm_only_negatives()
        if strategy in {"CA-HNM", "CA-HNM-full"}:
            return self.cahnm_negatives(source=strategy)
        if strategy in {"CA-HNM-v2", "CA-HNM-retrieval-aware"}:
            return self.cahnm_v2_negatives(source=strategy)
        if strategy in {"CA-HNM-v2-mixed", "CA-HNM-retrieval-aware-mixed"}:
            return self.cahnm_v2_mixed_negatives(source=strategy)
        if strategy == "CA-HNM-mixed":
            return self.cahnm_mixed_negatives()
        if strategy == "CA-HNM-no-ontology":
            return self.cahnm_no_ontology_negatives()
        if strategy == "CA-HNM-no-reasoning":
            return self.cahnm_negatives(source=strategy, context_depth=1)
        if strategy == "CA-HNM-no-fn-filter":
            return self.cahnm_negatives(source=strategy, exclude_positives=False)
        if strategy == "CA-HNM-prereq-only":
            return self.cahnm_negatives(
                source=strategy,
                allowed_violation_types=("prerequisite_mismatch", "postrequisite_mismatch"),
            )
        if strategy == "CA-HNM-sibling-only":
            return self.cahnm_negatives(source=strategy, allowed_violation_types=("sibling_concept_confusion",))
        raise ValueError(f"Unknown strategy: {strategy}")

    def cahnm_negatives(
        self,
        source: str = "CA-HNM",
        context_depth: int = 2,
        exclude_positives: bool = True,
        allowed_violation_types: tuple[str, ...] = (),
        candidate_fusion: str | None = None,
        selection_policy: str | None = None,
        confidence_threshold: float | None = None,
    ) -> list[NegativeRecord]:
        miner = CAHNMiner(
            self.documents,
            self.ontology,
            judge=self.judge,
            config=MiningConfig(
                top_k_bm25=self.config.top_k,
                top_k_dense=self.config.top_k,
                max_negatives_per_query=self.config.max_negatives_per_query,
                confidence_threshold=confidence_threshold
                if confidence_threshold is not None
                else self.config.confidence_threshold,
                random_seed=self.config.random_seed,
                dense_model=self.config.dense_model,
                dense_backend=self.config.dense_backend,
                dense_batch_size=self.config.dense_batch_size,
                dense_device=self.config.dense_device,
                dense_max_seq_length=self.config.dense_max_seq_length,
                source=source,
                context_depth=context_depth,
                exclude_positives=exclude_positives,
                allowed_violation_types=allowed_violation_types,
                candidate_fusion=candidate_fusion or self.config.candidate_fusion,
                selection_policy=selection_policy or self.config.selection_policy,
                retrieval_weight=self.config.retrieval_weight,
                constraint_weight=self.config.constraint_weight,
                ontology_weight=self.config.ontology_weight,
                diversity_penalty=self.config.diversity_penalty,
            ),
        )
        return miner.mine(self.queries, self.qrels)

    def cahnm_v2_negatives(self, source: str = "CA-HNM-v2") -> list[NegativeRecord]:
        return self.cahnm_negatives(
            source=source,
            candidate_fusion="rrf",
            selection_policy="retrieval_aware",
            confidence_threshold=min(self.config.confidence_threshold, 0.72),
        )

    def cahnm_v2_mixed_negatives(self, source: str = "CA-HNM-v2-mixed") -> list[NegativeRecord]:
        components = [
            self.ranked_negatives("dense_component", self.dense, violation_type="dense_hard_negative", confidence=0.62),
            self.adore_negatives(),
            self.cahnm_v2_negatives(source="constraint_retrieval_aware_component"),
        ]
        by_query: dict[str, list[NegativeRecord]] = defaultdict(list)
        for negatives in components:
            for negative in negatives:
                by_query[negative.query_id].append(negative)

        mixed: list[NegativeRecord] = []
        for query in self.queries:
            candidates = sorted(
                by_query.get(query.id, []),
                key=lambda item: (float(item.score or 0.0), item.confidence),
                reverse=True,
            )
            selected = 0
            seen_docs: set[str] = set()
            for candidate in candidates:
                if candidate.doc_id in seen_docs:
                    continue
                seen_docs.add(candidate.doc_id)
                mixed.append(
                    NegativeRecord(
                        query_id=candidate.query_id,
                        doc_id=candidate.doc_id,
                        source=source,
                        label=candidate.label,
                        violation_types=candidate.violation_types,
                        evidence=candidate.evidence,
                        confidence=candidate.confidence,
                        rank=selected + 1,
                        score=candidate.score,
                        metadata={**candidate.metadata, "component": candidate.source},
                    )
                )
                selected += 1
                if selected >= self.config.max_negatives_per_query:
                    break
        return mixed

    def cahnm_mixed_negatives(self) -> list[NegativeRecord]:
        components = [
            self.random_negatives(source="random_component", label="EasyNeg", confidence=0.5),
            self.ranked_negatives("bm25_component", self.bm25, violation_type="lexical_hard_negative", confidence=0.58),
            self.ranked_negatives("dense_component", self.dense, violation_type="dense_hard_negative", confidence=0.6),
            self.cahnm_negatives(source="constraint_component"),
        ]
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
                                source="CA-HNM-mixed",
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
            self.rng.shuffle(available)
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

    def ontology_negatives(self) -> list[NegativeRecord]:
        negatives: list[NegativeRecord] = []
        for query in self.queries:
            concept = query.target_concept or (self.ontology.find_in_text(query.text).id if self.ontology.find_in_text(query.text) else None)
            context = self.ontology.context(concept)
            labels = []
            labels.extend(label for node in context.broader for label in node.labels())
            labels.extend(label for node in context.narrower for label in node.labels())
            labels.extend(label for node in context.siblings for label in node.labels())
            labels.extend(label for node in context.prerequisites for label in node.labels())
            labels.extend(label for node in context.postrequisites for label in node.labels())
            labels.extend(label for node in context.related for label in node.labels())
            candidates: list[tuple[str, str, float]] = []
            for doc in self.documents:
                if doc.id in self.positives.get(query.id, set()):
                    continue
                hit = contains_any(doc.searchable_text, labels)
                if hit:
                    candidates.append((doc.id, hit, term_overlap(query.text, doc.searchable_text)))
            candidates.sort(key=lambda item: item[2], reverse=True)
            for rank, (doc_id, hit, score) in enumerate(candidates[: self.config.max_negatives_per_query], start=1):
                negatives.append(
                    NegativeRecord(
                        query_id=query.id,
                        doc_id=doc_id,
                        source="OntoNeg",
                        label="HardNeg",
                        violation_types=("ontology_neighbor",),
                        evidence=f"document mentions ontology neighbor '{hit}'",
                        confidence=0.68,
                        rank=rank,
                        score=score,
                    )
                )
        return negatives

    def adore_negatives(self) -> list[NegativeRecord]:
        negatives: list[NegativeRecord] = []
        for query in self.queries:
            ranked = self.dense.rank(query.text, self.config.top_k)
            if not ranked:
                continue
            top_score = ranked[0][1]
            selected = 0
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                if doc_id in self.positives.get(query.id, set()):
                    continue
                hardness = score / top_score if top_score else score
                negatives.append(
                    NegativeRecord(
                        query_id=query.id,
                        doc_id=doc_id,
                        source="ADORE",
                        label="HardNeg",
                        violation_types=("top_ranked_dense_negative",),
                        evidence="selected from top-ranked dense retrieval results",
                        confidence=max(0.55, min(0.9, hardness)),
                        rank=rank,
                        score=score,
                        metadata={"score_ratio_to_top": hardness},
                    )
                )
                selected += 1
                if selected >= self.config.max_negatives_per_query:
                    break
        return negatives

    def rocketqa_denoised_negatives(self) -> list[NegativeRecord]:
        negatives: list[NegativeRecord] = []
        empty_context = self.ontology.context(None)
        for query in self.queries:
            selected = 0
            for rank, (doc_id, score) in enumerate(self.dense.rank(query.text, self.config.top_k), start=1):
                if doc_id in self.positives.get(query.id, set()):
                    continue
                decision = self.judge.classify(query, self.doc_by_id[doc_id], empty_context)
                if decision.label in {"Positive", "Ambiguous"}:
                    continue
                negatives.append(
                    NegativeRecord(
                        query_id=query.id,
                        doc_id=doc_id,
                        source="RocketQA-Denoised",
                        label="HardNeg",
                        violation_types=("denoised_dense_negative",),
                        evidence=decision.evidence or "dense hard negative retained after denoising",
                        confidence=max(0.58, decision.confidence),
                        rank=rank,
                        score=score,
                    )
                )
                selected += 1
                if selected >= self.config.max_negatives_per_query:
                    break
        return negatives

    def tas_balanced_negatives(self) -> list[NegativeRecord]:
        negatives: list[NegativeRecord] = []
        pools = CandidatePool.union(
            self.queries,
            self.bm25,
            self.dense,
            top_k_bm25=self.config.top_k,
            top_k_dense=self.config.top_k,
        )
        for query in self.queries:
            buckets: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
            for doc_id, score, source in pools.rankings.get(query.id, []):
                if doc_id in self.positives.get(query.id, set()):
                    continue
                topic = self._topic_key(self.doc_by_id[doc_id])
                buckets[topic].append((doc_id, score, source))
            selected = self._round_robin_buckets(buckets, self.config.max_negatives_per_query)
            for rank, (doc_id, score, retrieval_source) in enumerate(selected, start=1):
                negatives.append(
                    NegativeRecord(
                        query_id=query.id,
                        doc_id=doc_id,
                        source="TAS-Balanced",
                        label="HardNeg",
                        violation_types=("topic_aware_balanced_negative",),
                        evidence=f"balanced sample from topic '{self._topic_key(self.doc_by_id[doc_id])}'",
                        confidence=0.62,
                        rank=rank,
                        score=score,
                        metadata={"retrieval_source": retrieval_source, "topic": self._topic_key(self.doc_by_id[doc_id])},
                    )
                )
        return negatives

    def gpl_pseudo_negatives(self) -> list[NegativeRecord]:
        negatives: list[NegativeRecord] = []
        pools = CandidatePool.union(
            self.queries,
            self.bm25,
            self.dense,
            top_k_bm25=self.config.top_k,
            top_k_dense=self.config.top_k,
        )
        empty_context = self.ontology.context(None)
        for query in self.queries:
            selected = 0
            for rank, (doc_id, score, retrieval_source) in enumerate(pools.rankings.get(query.id, []), start=1):
                if doc_id in self.positives.get(query.id, set()):
                    continue
                doc = self.doc_by_id[doc_id]
                pseudo_score = 0.5 * term_overlap(query.text, doc.searchable_text) + 0.5 * max(0.0, float(score))
                decision = self.judge.classify(query, doc, empty_context)
                if decision.label == "Positive":
                    continue
                negatives.append(
                    NegativeRecord(
                        query_id=query.id,
                        doc_id=doc_id,
                        source="GPL-Pseudo",
                        label="HardNeg",
                        violation_types=("generative_pseudo_labeled_negative",),
                        evidence=decision.evidence or "pseudo-labeled as non-relevant from retrieved candidate pool",
                        confidence=max(0.55, min(0.9, pseudo_score)),
                        rank=rank,
                        score=pseudo_score,
                        metadata={"retrieval_source": retrieval_source},
                    )
                )
                selected += 1
                if selected >= self.config.max_negatives_per_query:
                    break
        return negatives

    def llm_only_negatives(
        self,
        source: str = "LLMNeg",
        violation_type: str = "llm_similarity_only",
    ) -> list[NegativeRecord]:
        negatives: list[NegativeRecord] = []
        empty_context = self.ontology.context(None)
        for query in self.queries:
            selected = 0
            for rank, (doc_id, score) in enumerate(self.dense.rank(query.text, self.config.top_k), start=1):
                if doc_id in self.positives.get(query.id, set()):
                    continue
                decision = self.judge.classify(query, self.doc_by_id[doc_id], empty_context)
                if decision.label == "HardNeg":
                    negatives.append(
                        NegativeRecord(
                            query_id=query.id,
                            doc_id=doc_id,
                            source=source,
                            label="HardNeg",
                            violation_types=(violation_type,),
                            evidence=decision.evidence,
                            confidence=max(0.55, decision.confidence),
                            rank=rank,
                            score=score,
                        )
                    )
                    selected += 1
                if selected >= self.config.max_negatives_per_query:
                    break
        return negatives

    def _topic_key(self, document: Document) -> str:
        for key in ("topic", "subject", "category", "industry", "type", "source"):
            value = document.metadata.get(key)
            if isinstance(value, list) and value:
                return str(value[0])
            if value:
                return str(value)
        inferred = self.ontology.find_in_text(document.searchable_text)
        return inferred.id if inferred else "unknown"

    def _round_robin_buckets(
        self,
        buckets: dict[str, list[tuple[str, float, str]]],
        limit: int,
    ) -> list[tuple[str, float, str]]:
        selected: list[tuple[str, float, str]] = []
        keys = sorted(buckets)
        while len(selected) < limit and keys:
            next_keys: list[str] = []
            for key in keys:
                bucket = buckets[key]
                if bucket and len(selected) < limit:
                    selected.append(bucket.pop(0))
                if bucket:
                    next_keys.append(key)
            keys = next_keys
        return selected


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
