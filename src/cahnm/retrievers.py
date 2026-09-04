from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Protocol

from .schemas import Document, Query, RetrievalRun
from .text import cosine, hashed_vector, tokenize


class Retriever(Protocol):
    def rank(self, query: str, top_k: int) -> list[tuple[str, float]]:
        ...


class BM25Retriever:
    def __init__(self, documents: list[Document], k1: float = 1.2, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc.searchable_text) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        df: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                df[token] += 1
        n_docs = max(1, len(documents))
        self.idf = {
            token: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def score(self, query: str, doc_idx: int) -> float:
        score = 0.0
        freqs = self.term_freqs[doc_idx]
        dl = self.doc_lengths[doc_idx] or 1
        for token in tokenize(query):
            tf = freqs.get(token, 0)
            if not tf:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
            score += self.idf.get(token, 0.0) * (tf * (self.k1 + 1)) / denom
        return score

    def rank(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        scored = [(doc.id, self.score(query, idx)) for idx, doc in enumerate(self.documents)]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def run(self, queries: list[Query], top_k: int = 100) -> RetrievalRun:
        return RetrievalRun("bm25", {query.id: self.rank(query.text, top_k) for query in queries})


class DenseRetriever:
    """Dense-like retriever with optional backends.

    It uses sentence-transformers when available and requested, scikit-learn
    TF-IDF when available, and a deterministic hashed bag-of-words fallback
    otherwise. The fallback keeps the framework smoke-testable on clean
    machines while preserving the same ranker interface.
    """

    def __init__(
        self,
        documents: list[Document],
        model_name: str | None = None,
        backend: str = "auto",
        batch_size: int = 16,
        device: str | None = None,
        max_seq_length: int | None = None,
    ):
        self.documents = documents
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.max_seq_length = max_seq_length
        self.backend = "hash"
        self.model = None
        self.doc_vectors = None
        texts = [doc.searchable_text for doc in documents]

        if backend in {"auto", "sentence-transformers"} and model_name:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model_name, device=device)
                if max_seq_length:
                    self.model.max_seq_length = max_seq_length
                self.doc_vectors = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                self.backend = "sentence-transformers"
                return
            except Exception:
                if backend == "sentence-transformers":
                    raise

        if backend in {"auto", "tfidf"}:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.preprocessing import normalize as sk_normalize

                self.model = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words="english")
                self.doc_vectors = sk_normalize(self.model.fit_transform(texts))
                self.backend = "tfidf"
                return
            except Exception:
                if backend == "tfidf":
                    raise

        self.doc_vectors = [hashed_vector(text) for text in texts]

    def rank(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.backend == "sentence-transformers":
            q_vec = self.model.encode(
                [query],
                batch_size=1,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0]
            scores = self.doc_vectors @ q_vec
            ranked = [(self.documents[idx].id, float(score)) for idx, score in enumerate(scores)]
        elif self.backend == "tfidf":
            q_vec = self.model.transform([query])
            scores = (self.doc_vectors @ q_vec.T).toarray().ravel()
            ranked = [(self.documents[idx].id, float(score)) for idx, score in enumerate(scores)]
        else:
            q_vec = hashed_vector(query)
            ranked = [(doc.id, cosine(q_vec, vec)) for doc, vec in zip(self.documents, self.doc_vectors)]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def run(self, queries: list[Query], top_k: int = 100) -> RetrievalRun:
        return RetrievalRun(f"dense:{self.backend}", {query.id: self.rank(query.text, top_k) for query in queries})


@dataclass
class CandidatePool:
    rankings: dict[str, list[tuple[str, float, str]]]

    @classmethod
    def union(
        cls,
        queries: list[Query],
        bm25: BM25Retriever,
        dense: DenseRetriever | None,
        top_k_bm25: int,
        top_k_dense: int,
        fusion: str = "max_score",
        rrf_k: int = 60,
    ) -> "CandidatePool":
        pools: dict[str, list[tuple[str, float, str]]] = {}
        for query in queries:
            merged: dict[str, dict[str, float | int | str]] = {}
            for rank, (doc_id, score) in enumerate(bm25.rank(query.text, top_k_bm25), start=1):
                merged[doc_id] = {
                    "bm25_rank": rank,
                    "bm25_score": float(score),
                    "dense_rank": 0,
                    "dense_score": 0.0,
                }
            if dense is not None:
                for rank, (doc_id, score) in enumerate(dense.rank(query.text, top_k_dense), start=1):
                    if doc_id in merged:
                        merged[doc_id]["dense_rank"] = rank
                        merged[doc_id]["dense_score"] = float(score)
                    else:
                        merged[doc_id] = {
                            "bm25_rank": 0,
                            "bm25_score": 0.0,
                            "dense_rank": rank,
                            "dense_score": float(score),
                        }
            ordered = [
                (doc_id, _fused_candidate_score(values, fusion=fusion, rrf_k=rrf_k), _candidate_source(values))
                for doc_id, values in merged.items()
            ]
            ordered.sort(key=lambda item: (-item[1], item[0]))
            pools[query.id] = ordered
        return cls(pools)


def _fused_candidate_score(values: dict[str, float | int | str], fusion: str, rrf_k: int) -> float:
    if fusion == "rrf":
        score = 0.0
        bm25_rank = int(values.get("bm25_rank") or 0)
        dense_rank = int(values.get("dense_rank") or 0)
        if bm25_rank:
            score += 1.0 / (rrf_k + bm25_rank)
        if dense_rank:
            score += 1.0 / (rrf_k + dense_rank)
        return score
    if fusion != "max_score":
        raise ValueError(f"Unknown candidate fusion: {fusion}")
    return max(float(values.get("bm25_score") or 0.0), float(values.get("dense_score") or 0.0))


def _candidate_source(values: dict[str, float | int | str]) -> str:
    parts: list[str] = []
    bm25_rank = int(values.get("bm25_rank") or 0)
    dense_rank = int(values.get("dense_rank") or 0)
    if bm25_rank:
        parts.append(f"bm25:{bm25_rank}")
    if dense_rank:
        parts.append(f"dense:{dense_rank}")
    return "|".join(parts) or "unknown"
