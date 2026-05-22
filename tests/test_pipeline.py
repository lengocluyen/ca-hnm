import gzip
from pathlib import Path

from cahnm.baselines import BaselineConfig, NegativeStrategyRunner, summarize_negative_quality
from cahnm.data.download import (
    build_course_skill_atlas_ontology,
    build_mooccubex_ontology,
    download_mooccubex,
    write_sample_dataset,
)
from cahnm.data.prepare import prepare_course_skill_atlas_benchmark, prepare_mooccubex_benchmark
from cahnm.evaluation import evaluate_run
from cahnm.external import load_external_negatives, load_run
from cahnm.io_utils import load_corpus, load_qrels, load_queries
from cahnm.miner import CAHNMiner, MiningConfig
from cahnm.ontology import Ontology, make_sample_ontology
from cahnm.retrievers import BM25Retriever, CandidatePool, DenseRetriever
from cahnm.schemas import RetrievalRun


def test_sample_cahnm_pipeline(tmp_path: Path) -> None:
    dataset = write_sample_dataset(tmp_path)
    docs = load_corpus(dataset / "corpus.jsonl")
    queries = load_queries(dataset / "queries.jsonl")
    qrels = load_qrels(dataset / "qrels.tsv")
    ontology = Ontology.load(dataset / "ontology.json")

    miner = CAHNMiner(
        docs,
        ontology,
        config=MiningConfig(top_k_bm25=5, top_k_dense=5, max_negatives_per_query=3, dense_backend="hash"),
    )
    negatives = miner.mine(queries, qrels)

    assert negatives
    assert any("level_mismatch" in n.violation_types or "postrequisite_mismatch" in n.violation_types for n in negatives)
    assert all(n.source == "CA-HNM" for n in negatives)


def test_baseline_quality_summary(tmp_path: Path) -> None:
    dataset = write_sample_dataset(tmp_path)
    docs = load_corpus(dataset / "corpus.jsonl")
    queries = load_queries(dataset / "queries.jsonl")
    qrels = load_qrels(dataset / "qrels.tsv")
    ontology = Ontology.load(dataset / "ontology.json")

    runner = NegativeStrategyRunner(
        docs,
        queries,
        qrels,
        ontology,
        config=BaselineConfig(max_negatives_per_query=2, top_k=5, dense_backend="hash"),
    )
    negatives_by_strategy = {name: runner.run(name) for name in ["RandomNeg", "BM25Neg", "DenseNeg", "OntoNeg", "LLMNeg", "CA-HNM"]}
    rows = summarize_negative_quality(negatives_by_strategy, docs, queries, qrels, ontology)

    assert {row["strategy"] for row in rows} == set(negatives_by_strategy)
    assert any(row["strategy"] == "CA-HNM" and row["valid_hard_rate"] > 0 for row in rows)


def test_related_work_strategy_names(tmp_path: Path) -> None:
    dataset = write_sample_dataset(tmp_path)
    docs = load_corpus(dataset / "corpus.jsonl")
    queries = load_queries(dataset / "queries.jsonl")
    qrels = load_qrels(dataset / "qrels.tsv")
    ontology = Ontology.load(dataset / "ontology.json")

    runner = NegativeStrategyRunner(
        docs,
        queries,
        qrels,
        ontology,
        config=BaselineConfig(max_negatives_per_query=2, top_k=5, dense_backend="hash"),
    )
    strategies = ["DPR-Random", "ANCE", "ADORE", "RocketQA-Denoised", "TAS-Balanced", "GPL-Pseudo", "SyNeg"]

    for strategy in strategies:
        negatives = runner.run(strategy)
        assert negatives
        assert all(negative.source == strategy for negative in negatives)


def test_cahnm_ablation_strategy_names(tmp_path: Path) -> None:
    dataset = write_sample_dataset(tmp_path)
    docs = load_corpus(dataset / "corpus.jsonl")
    queries = load_queries(dataset / "queries.jsonl")
    qrels = load_qrels(dataset / "qrels.tsv")
    ontology = Ontology.load(dataset / "ontology.json")

    runner = NegativeStrategyRunner(
        docs,
        queries,
        qrels,
        ontology,
        config=BaselineConfig(max_negatives_per_query=2, top_k=5, dense_backend="hash"),
    )
    strategies = [
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

    for strategy in strategies:
        negatives = runner.run(strategy)
        assert all(negative.source == strategy for negative in negatives)


def test_retrieval_aware_cahnm_uses_rrf_metadata(tmp_path: Path) -> None:
    dataset = write_sample_dataset(tmp_path)
    docs = load_corpus(dataset / "corpus.jsonl")
    queries = load_queries(dataset / "queries.jsonl")
    qrels = load_qrels(dataset / "qrels.tsv")
    ontology = Ontology.load(dataset / "ontology.json")

    runner = NegativeStrategyRunner(
        docs,
        queries,
        qrels,
        ontology,
        config=BaselineConfig(max_negatives_per_query=2, top_k=5, dense_backend="hash"),
    )
    negatives = runner.run("CA-HNM-v2")

    assert negatives
    assert all(negative.source == "CA-HNM-v2" for negative in negatives)
    assert all(negative.metadata["candidate_fusion"] == "rrf" for negative in negatives)
    assert all(negative.metadata["selection_policy"] == "retrieval_aware" for negative in negatives)
    assert all("retrieval_aware_score" in negative.metadata for negative in negatives)


def test_candidate_pool_rrf_avoids_raw_score_fusion(tmp_path: Path) -> None:
    dataset = write_sample_dataset(tmp_path)
    docs = load_corpus(dataset / "corpus.jsonl")
    queries = load_queries(dataset / "queries.jsonl")
    bm25 = BM25Retriever(docs)
    dense = DenseRetriever(docs, backend="hash")

    pool = CandidatePool.union(queries[:1], bm25, dense, top_k_bm25=5, top_k_dense=5, fusion="rrf")

    assert pool.rankings[queries[0].id]
    assert all(0.0 < score < 1.0 for _, score, _ in pool.rankings[queries[0].id])
    assert any("bm25:" in source or "dense:" in source for _, _, source in pool.rankings[queries[0].id])


def test_evaluate_run_reports_rank_metrics(tmp_path: Path) -> None:
    dataset = write_sample_dataset(tmp_path)
    qrels = load_qrels(dataset / "qrels.tsv")
    run = RetrievalRun("toy", {"q1": [("d1", 1.0), ("d2", 0.5)], "q2": [("d4", 1.0), ("d5", 0.5)]})

    metrics = evaluate_run(run, qrels, ks=(1, 2))

    assert "NDCG@1" in metrics
    assert "MAP@2" in metrics
    assert "MRR@2" in metrics


def test_ontology_reasoning_closure() -> None:
    ontology = make_sample_ontology()

    optimization_context = ontology.context("query_optimization")
    prerequisite_ids = {node.id for node in optimization_context.prerequisites}
    assert {"sql_joins", "relational_keys"} <= prerequisite_ids

    keys_context = ontology.context("relational_keys")
    postrequisite_ids = {node.id for node in keys_context.postrequisites}
    assert {"sql_joins", "query_optimization"} <= postrequisite_ids

    joins_context = ontology.context("sql_joins")
    sibling_ids = {node.id for node in joins_context.siblings}
    assert "sql_filtering" in sibling_ids


def test_external_trec_run_loader_and_negative_normalizer(tmp_path: Path) -> None:
    run_path = tmp_path / "ance.run"
    run_path.write_text(
        "q1 Q0 d2 1 12.5 ANCE\n"
        "q1 Q0 d3 2 10.0 ANCE\n"
        "q2 Q0 d4 1 9.5 ANCE\n",
        encoding="utf-8",
    )

    run = load_run(run_path, run_format="trec", name="ANCE-official")
    negatives = load_external_negatives(run_path, source="ANCE", input_format="trec")

    assert run.name == "ANCE-official"
    assert run.rankings["q1"] == [("d2", 12.5), ("d3", 10.0)]
    assert negatives[0].query_id == "q1"
    assert negatives[0].doc_id == "d2"
    assert negatives[0].source == "ANCE"


def test_external_dpr_json_maps_question_text_to_query_id(tmp_path: Path) -> None:
    dataset = write_sample_dataset(tmp_path / "data")
    queries = load_queries(dataset / "queries.jsonl")
    dpr_path = tmp_path / "dpr_results.json"
    dpr_path.write_text(
        """
        [
          {
            "question": "beginner SQL joins",
            "ctxs": [
              {"id": "doc-x", "score": "4.2"},
              {"id": "doc-y", "score": "3.1"}
            ]
          }
        ]
        """,
        encoding="utf-8",
    )

    run = load_run(dpr_path, run_format="dpr", name="DPR-official", queries=queries)

    assert "q1" in run.rankings
    assert run.rankings["q1"] == [("doc-x", 4.2), ("doc-y", 3.1)]


def test_mooccubex_manifest_and_benchmark_preparation(tmp_path: Path) -> None:
    manifest_path = download_mooccubex(tmp_path / "manifest_only")
    assert manifest_path.name == "mooccubex_manifest.json"

    raw = tmp_path / "mooccubex"
    (raw / "entities").mkdir(parents=True)
    (raw / "relations").mkdir()
    (raw / "prerequisites").mkdir()
    (raw / "entities" / "concept.json").write_text(
        '{"id":"K_array_cs","name":"array","context":"array\n'
        'data structure"}\n'
        '{"id":"K_tree_cs","name":"tree","context":"tree data structure"}\n',
        encoding="utf-8",
    )
    (raw / "entities" / "course.json").write_text(
        '{"id":"C1","name":"Data Structures","about":"Arrays\nand trees.","field":["Computer Science"],"prerequisites":"Programming basics","resource":[]}\n',
        encoding="utf-8",
    )
    (raw / "relations" / "concept-course.txt").write_text("K_array_cs\tC1\nK_tree_cs\tC1\n", encoding="utf-8")
    (raw / "prerequisites" / "cs.json").write_text('{"c1":"array","c2":"tree","ground_truth":1}\n', encoding="utf-8")

    ontology_path = build_mooccubex_ontology(raw, tmp_path / "mooccubex_ontology.json", max_prerequisites=10)
    prepared = prepare_mooccubex_benchmark(raw, tmp_path / "prepared", ontology_path, max_queries=10)
    ontology = Ontology.load(ontology_path)
    docs = load_corpus(prepared / "corpus.jsonl")
    queries = load_queries(prepared / "queries.jsonl")
    qrels = load_qrels(prepared / "qrels.tsv")

    assert "K_array_cs" in ontology.nodes
    assert any(rel.source == "K_tree_cs" and rel.target == "K_array_cs" and rel.type == "requires" for rel in ontology.relations)
    assert docs and docs[0].id == "C1"
    assert queries
    assert qrels


def test_course_skill_atlas_ontology_and_benchmark_preparation(tmp_path: Path) -> None:
    raw = tmp_path / "course_skill_atlas"
    raw.mkdir()
    (raw / "field_name_and_code.csv").write_text(
        "field_name,field_code\nComputer Science,11\nAccounting,52.03\n",
        encoding="utf-8",
    )
    with gzip.open(raw / "detailed_work_activities_scores.gzip", "wt", encoding="utf-8", newline="") as handle:
        handle.write("id,Write computer programs.,Analyze financial information.\n")
        handle.write("1,0.9,0.1\n")
    with gzip.open(raw / "abilities_scores.gzip", "wt", encoding="utf-8", newline="") as handle:
        handle.write("id,Deductive Reasoning\n1,0.7\n")
    with gzip.open(raw / "tasks_scores.gzip", "wt", encoding="utf-8", newline="") as handle:
        handle.write("id,Develop software applications.\n1,0.8\n")
    with gzip.open(raw / "institution_fos_year.gzip", "wt", encoding="utf-8", newline="") as handle:
        handle.write("id,year,field_name,field_code,syllabi_cnt,UnitID,institution_name,city,state_code,sector\n")
        handle.write("1,2020,Computer Science,11,4,100,Example University,Boston,MA,Public\n")
        handle.write("2,2020,Accounting,52.03,3,101,Example College,Austin,TX,Private\n")
    (raw / "top10_DWA_per_FOS.csv").write_text(
        "Detailed Work Activity (DWA),Rank\n"
        "Computer Science,\n"
        "Write computer programs.,1\n"
        "Accounting,\n"
        "Analyze financial information.,1\n",
        encoding="utf-8",
    )

    ontology_path = build_course_skill_atlas_ontology(raw, tmp_path / "course_skill_atlas_ontology.json")
    prepared = prepare_course_skill_atlas_benchmark(raw, tmp_path / "prepared_csa", ontology_path, max_queries=10)
    ontology = Ontology.load(ontology_path)
    docs = load_corpus(prepared / "corpus.jsonl")
    queries = load_queries(prepared / "queries.jsonl")
    qrels = load_qrels(prepared / "qrels.tsv")

    assert "csa_field" in ontology.nodes
    assert "csa_dwa" in ontology.nodes
    assert any(rel.type == "related" for rel in ontology.relations)
    assert len(docs) == 2
    assert queries
    assert qrels
