from __future__ import annotations

import math
from collections import defaultdict

from .judges import ConstraintJudge, HeuristicConstraintJudge
from .ontology import Ontology
from .schemas import Document, Qrel, Query, RetrievalRun


def qrels_to_dict(qrels: list[Qrel]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in qrels:
        grouped[qrel.query_id][qrel.doc_id] = qrel.relevance
    return grouped


def dcg(relevances: list[int]) -> float:
    return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def evaluate_run(run: RetrievalRun, qrels: list[Qrel], ks: tuple[int, ...] = (10, 100)) -> dict[str, float]:
    per_query = evaluate_run_per_query(run, qrels, ks=ks)
    metric_names = sorted({name for row in per_query.values() for name in row})
    return {
        name: sum(row.get(name, 0.0) for row in per_query.values()) / len(per_query) if per_query else 0.0
        for name in metric_names
    }


def evaluate_run_per_query(run: RetrievalRun, qrels: list[Qrel], ks: tuple[int, ...] = (10, 100)) -> dict[str, dict[str, float]]:
    qrel_map = qrels_to_dict(qrels)
    per_query: dict[str, dict[str, float]] = {}

    for query_id, rels in qrel_map.items():
        ranking = [doc_id for doc_id, _ in run.rankings.get(query_id, [])]
        relevant_docs = {doc_id for doc_id, rel in rels.items() if rel > 0}
        if not relevant_docs:
            continue
        row: dict[str, float] = {}

        for k in ks:
            top = ranking[:k]
            gains = [rels.get(doc_id, 0) for doc_id in top]
            ideal = sorted([rel for rel in rels.values() if rel > 0], reverse=True)[:k]
            row[f"NDCG@{k}"] = dcg(gains) / dcg(ideal) if ideal and dcg(ideal) else 0.0
            hits = sum(1 for doc_id in top if doc_id in relevant_docs)
            row[f"Recall@{k}"] = hits / len(relevant_docs)
            row[f"Precision@{k}"] = hits / k
            rr_at_k = 0.0
            ap_sum_at_k = 0.0
            hits_at_k = 0
            for idx, doc_id in enumerate(top, start=1):
                if doc_id in relevant_docs:
                    hits_at_k += 1
                    ap_sum_at_k += hits_at_k / idx
                    if rr_at_k == 0.0:
                        rr_at_k = 1.0 / idx
            row[f"MRR@{k}"] = rr_at_k
            row[f"MAP@{k}"] = ap_sum_at_k / min(len(relevant_docs), k)

        rr = 0.0
        for idx, doc_id in enumerate(ranking[:10], start=1):
            if doc_id in relevant_docs:
                rr = 1.0 / idx
                break
        row["MRR@10"] = rr

        ap_sum = 0.0
        hits = 0
        for idx, doc_id in enumerate(ranking, start=1):
            if doc_id in relevant_docs:
                hits += 1
                ap_sum += hits / idx
        row["MAP"] = ap_sum / len(relevant_docs)
        per_query[query_id] = row

    return per_query


def constraint_violation_at_k(
    run: RetrievalRun,
    queries: list[Query],
    documents: list[Document],
    qrels: list[Qrel],
    ontology: Ontology,
    k: int = 10,
    judge: ConstraintJudge | None = None,
) -> dict[str, float]:
    judge = judge or HeuristicConstraintJudge()
    doc_by_id = {doc.id: doc for doc in documents}
    positives = qrels_to_dict(qrels)
    rows: list[float] = []
    type_counts: defaultdict[str, int] = defaultdict(int)
    total = 0

    for query in queries:
        concept = query.target_concept or (ontology.find_in_text(query.text).id if ontology.find_in_text(query.text) else None)
        context = ontology.context(concept)
        violated = 0
        considered = 0
        for doc_id, _ in run.rankings.get(query.id, [])[:k]:
            if positives.get(query.id, {}).get(doc_id, 0) > 0:
                continue
            considered += 1
            decision = judge.classify(query, doc_by_id[doc_id], context)
            if decision.is_hard_negative:
                violated += 1
                for violation in decision.violation_types:
                    type_counts[violation] += 1
        if considered:
            rows.append(violated / considered)
            total += considered

    summary = {f"ConstraintViolation@{k}": sum(rows) / len(rows) if rows else 0.0}
    for violation, count in sorted(type_counts.items()):
        summary[f"{violation}@{k}"] = count / total if total else 0.0
    return summary
