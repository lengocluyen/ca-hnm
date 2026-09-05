import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

from cahnm.baselines import (
    CORE_STRATEGIES,
    DEFAULT_STRATEGIES,
    MIXED_STRATEGIES,
    STRUCTURAL_STRATEGIES,
    BaselineConfig,
    NegativeStrategyRunner,
)
from cahnm.cli import _summarize_training_data
from cahnm.data.download import (
    build_course_skill_atlas_ontology,
    build_mooccubex_ontology,
)
from cahnm.data.prepare import (
    _safe_query_id,
    prepare_course_skill_atlas_benchmark,
    prepare_mooccubex_benchmark,
)
from cahnm.evaluation import evaluate_run
from cahnm.io_utils import (
    load_corpus,
    load_qrels,
    load_queries,
    write_qrels,
    write_queries,
)
from cahnm.ontology import Ontology, make_sample_ontology
from cahnm.schemas import Qrel, Query, RetrievalRun
from cahnm.training import build_triplets
from scripts.split_benchmark import main as split_benchmark_main


REPO_ROOT = Path(__file__).resolve().parents[1]


def _runner(tmp_path: Path, *, max_negatives: int = 4) -> NegativeStrategyRunner:
    dataset = REPO_ROOT / "examples" / "sample_benchmark"
    return NegativeStrategyRunner(
        load_corpus(dataset / "corpus.jsonl"),
        load_queries(dataset / "queries.jsonl"),
        load_qrels(dataset / "qrels.tsv"),
        Ontology.load(dataset / "ontology.json"),
        config=BaselineConfig(
            max_negatives_per_query=max_negatives,
            top_k=5,
            dense_backend="hash",
            random_seed=13,
        ),
    )


def test_public_strategy_surface_matches_paper(tmp_path: Path) -> None:
    assert DEFAULT_STRATEGIES == CORE_STRATEGIES
    assert CORE_STRATEGIES == [
        "DPR-Random",
        "DenseNeg",
        "CA-HNM-full",
        "CA-HNM-rank-matched",
    ]
    assert MIXED_STRATEGIES == [
        "DPR-Random",
        "DenseNeg",
        "CA-HNM-mixed",
        "CA-HNM-matched-mixed",
    ]
    assert STRUCTURAL_STRATEGIES == [
        "CA-HNM-full",
        "CA-HNM-label-only",
        "CA-HNM-shuffled-graph",
        "CA-HNM-no-ontology",
    ]

    runner = _runner(tmp_path)
    for strategy in sorted(set(CORE_STRATEGIES + MIXED_STRATEGIES + STRUCTURAL_STRATEGIES)):
        negatives = runner.run(strategy)
        assert all(row.source == strategy for row in negatives)
        assert all(row.doc_id not in runner.positives.get(row.query_id, set()) for row in negatives)


def test_rank_matched_control_is_one_for_one_and_excludes_treatment(tmp_path: Path) -> None:
    runner = _runner(tmp_path, max_negatives=2)
    treatment = runner.run("CA-HNM-full")
    control = runner.run("CA-HNM-rank-matched")

    assert len(control) == len(treatment)
    assert not {(row.query_id, row.doc_id) for row in treatment} & {
        (row.query_id, row.doc_id) for row in control
    }
    assert all(
        row.metadata["causal_control"] == "retrieval-rank-and-source-matched"
        for row in control
    )
    assert all(row.metadata.get("rank_distance", -1) >= 0 for row in control)


def test_matched_mixture_replaces_constraint_component(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    treatment = runner.run("CA-HNM-mixed")
    control = runner.run("CA-HNM-matched-mixed")

    assert treatment and control
    assert len(treatment) == len(control)
    matched = [
        row
        for row in control
        if row.metadata.get("component") == "matched_constraint_component"
    ]
    assert matched
    assert all(
        row.metadata.get("matching_policy") == "source_then_absolute_rank"
        for row in matched
    )


def test_random_component_is_invariant_to_call_order(tmp_path: Path) -> None:
    runner = _runner(tmp_path, max_negatives=2)
    first = runner.random_negatives(source="random_component")
    runner.run("DPR-Random")
    second = runner.random_negatives(source="random_component")

    assert [(row.query_id, row.doc_id) for row in first] == [
        (row.query_id, row.doc_id) for row in second
    ]


def test_training_data_summary_reports_realized_counts(tmp_path: Path) -> None:
    runner = _runner(tmp_path, max_negatives=2)
    negatives = runner.run("DenseNeg")
    triplets = build_triplets(
        runner.queries,
        runner.documents,
        runner.qrels,
        negatives,
        random_seed=17,
    )

    summary = _summarize_training_data(
        "DenseNeg", len(runner.queries), negatives, triplets
    )
    assert summary == {
        "strategy": "DenseNeg",
        "input_queries": len(runner.queries),
        "retained_negative_queries": len({row.query_id for row in negatives}),
        "negative_count": len(negatives),
        "retained_triplet_queries": len({row["query_id"] for row in triplets}),
        "triplet_count": len(triplets),
    }


def test_shuffled_ontology_preserves_typed_degree_marginals() -> None:
    ontology = make_sample_ontology()
    shuffled = ontology.shuffled_relations(random_seed=13)

    assert Counter((row.type, row.source) for row in ontology.relations) == Counter(
        (row.type, row.source) for row in shuffled.relations
    )
    assert Counter((row.type, row.target) for row in ontology.relations) == Counter(
        (row.type, row.target) for row in shuffled.relations
    )
    assert all(
        row.metadata.get("placebo") == "shuffled_target"
        for row in shuffled.relations
    )


def test_binary_retrieval_metrics() -> None:
    qrels = [Qrel("q1", "d1", 1), Qrel("q2", "d5", 1)]
    run = RetrievalRun(
        "toy",
        {"q1": [("d1", 1.0), ("d2", 0.5)], "q2": [("d4", 1.0), ("d5", 0.5)]},
    )
    metrics = evaluate_run(run, qrels, ks=(1, 2))

    assert "NDCG@1" in metrics
    assert "MRR@2" in metrics
    assert "MAP@2" in metrics
    assert "Recall@2" in metrics


def test_ontology_reasoning_closure() -> None:
    ontology = make_sample_ontology()
    prerequisites = {
        node.id for node in ontology.context("query_optimization").prerequisites
    }
    postrequisites = {
        node.id for node in ontology.context("relational_keys").postrequisites
    }
    siblings = {node.id for node in ontology.context("sql_joins").siblings}

    assert {"sql_joins", "relational_keys"} <= prerequisites
    assert {"sql_joins", "query_optimization"} <= postrequisites
    assert "sql_filtering" in siblings


def test_safe_query_ids_preserve_punctuation_distinctions() -> None:
    assert _safe_query_id("K_itemset ") != _safe_query_id("K_itemset.")


def test_target_disjoint_split(tmp_path: Path, monkeypatch) -> None:
    queries = [
        Query("q0a", "first rendering", "shared-target"),
        Query("q0b", "second rendering", "shared-target"),
        *(Query(f"q{i}", f"query {i}", f"target-{i}") for i in range(1, 12)),
    ]
    qrels = [Qrel(query.id, f"d-{query.id}", 1) for query in queries]
    query_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.tsv"
    out_dir = tmp_path / "split"
    write_queries(query_path, queries)
    write_qrels(qrels_path, qrels)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_benchmark.py",
            "--queries",
            str(query_path),
            "--qrels",
            str(qrels_path),
            "--out-dir",
            str(out_dir),
            "--seed",
            "19",
        ],
    )

    split_benchmark_main()
    split_ids = {
        name: {query.id for query in load_queries(out_dir / f"{name}.queries.jsonl")}
        for name in ("train", "dev", "test")
    }

    assert not split_ids["train"] & split_ids["dev"]
    assert not split_ids["train"] & split_ids["test"]
    assert not split_ids["dev"] & split_ids["test"]
    assert any({"q0a", "q0b"} <= ids for ids in split_ids.values())
    manifest = json.loads((out_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["overlap_audit"] == {
        "dev_test": 0,
        "train_dev": 0,
        "train_test": 0,
    }


def test_mooccubex_preparation(tmp_path: Path) -> None:
    raw = tmp_path / "mooccubex"
    (raw / "entities").mkdir(parents=True)
    (raw / "relations").mkdir()
    (raw / "prerequisites").mkdir()
    (raw / "entities" / "concept.json").write_text(
        '{"id":"K_array_cs","name":"array","context":"array data structure"}\n'
        '{"id":"K_tree_cs","name":"tree","context":"tree data structure"}\n',
        encoding="utf-8",
    )
    (raw / "entities" / "course.json").write_text(
        '{"id":"C1","name":"Data Structures","about":"Arrays and trees.",'
        '"field":["Computer Science"],"prerequisites":"Programming basics",'
        '"resource":[]}\n',
        encoding="utf-8",
    )
    (raw / "relations" / "concept-course.txt").write_text(
        "K_array_cs\tC1\nK_tree_cs\tC1\n", encoding="utf-8"
    )
    (raw / "prerequisites" / "cs.json").write_text(
        '{"c1":"array","c2":"tree","ground_truth":1}\n', encoding="utf-8"
    )

    ontology_path = build_mooccubex_ontology(
        raw, tmp_path / "ontology.json", max_prerequisites=10
    )
    prepared = prepare_mooccubex_benchmark(
        raw, tmp_path / "prepared", ontology_path, max_queries=10
    )

    assert load_corpus(prepared / "corpus.jsonl")
    assert load_queries(prepared / "queries.jsonl")
    assert load_qrels(prepared / "qrels.tsv")


def test_course_skill_atlas_preparation(tmp_path: Path) -> None:
    raw = tmp_path / "course_skill_atlas"
    raw.mkdir()
    (raw / "field_name_and_code.csv").write_text(
        "field_name,field_code\nComputer Science,11\nAccounting,52.03\n",
        encoding="utf-8",
    )
    with gzip.open(
        raw / "detailed_work_activities_scores.gzip", "wt", encoding="utf-8", newline=""
    ) as handle:
        handle.write("id,Write computer programs.,Analyze financial information.\n")
        handle.write("1,0.9,0.1\n")
    with gzip.open(raw / "abilities_scores.gzip", "wt", encoding="utf-8", newline="") as handle:
        handle.write("id,Deductive Reasoning\n1,0.7\n")
    with gzip.open(raw / "tasks_scores.gzip", "wt", encoding="utf-8", newline="") as handle:
        handle.write("id,Develop software applications.\n1,0.8\n")
    with gzip.open(
        raw / "institution_fos_year.gzip", "wt", encoding="utf-8", newline=""
    ) as handle:
        handle.write(
            "id,year,field_name,field_code,syllabi_cnt,UnitID,institution_name,city,state_code,sector\n"
        )
        handle.write("1,2020,Computer Science,11,4,100,Example University,Boston,MA,Public\n")
        handle.write("2,2020,Accounting,52.03,3,101,Example College,Austin,TX,Private\n")
    (raw / "top10_DWA_per_FOS.csv").write_text(
        "Detailed Work Activity (DWA),Rank\n"
        "Computer Science,\nWrite computer programs.,1\n"
        "Accounting,\nAnalyze financial information.,1\n",
        encoding="utf-8",
    )

    ontology_path = build_course_skill_atlas_ontology(raw, tmp_path / "ontology.json")
    prepared = prepare_course_skill_atlas_benchmark(
        raw, tmp_path / "prepared", ontology_path, max_queries=10
    )

    assert len(load_corpus(prepared / "corpus.jsonl")) == 2
    assert load_queries(prepared / "queries.jsonl")
    assert load_qrels(prepared / "qrels.tsv")


def test_included_forest_source_matches_paper_claim() -> None:
    with (REPO_ROOT / "results" / "effect_forest_source.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 8
    assert sum(float(row["mean_diff"]) > 0 for row in rows) == 6
    assert sum(float(row["hierarchical_ci95_low"]) > 0 for row in rows) == 4
    for row in rows:
        observed = float(row["strategy_mean"]) - float(row["baseline_mean"])
        assert abs(observed - float(row["mean_diff"])) < 1e-12
