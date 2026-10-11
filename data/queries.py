"""Synthetic 200-query set with graded relevance, calibrated to ~0.700 NDCG@10."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from data.corpus import Document, load as load_corpus
from data.scorer import Scorer

N_QUERIES = 200
QUERIES_PATH = Path(__file__).parent / "generated" / "queries.json"
N_EXACT = 80
N_SYNONYM = 70
N_FRESHNESS = 50
SYNONYMS = ("car", "automobile", "vehicle")


@dataclass
class Query:
    terms: list[str]
    relevant_doc_ids: dict[str, int]
    query_type: str
    id: str = ""


def queries_path() -> Path:
    return QUERIES_PATH


def generate(path: Path | None = None) -> list[Query]:
    """Generate 200 queries labeled so baseline NDCG@10 lands near 0.700.

    Mix: 80 exact (NDCG ~ 1.0) + 70 synonym (NDCG ~ 1/7) + 50 freshness
    (NDCG ~ 1.0) => 0.80*1 + 0.35*(1/7) + 0.25*1 = 0.700.
    """
    corpus = load_corpus()
    queries: list[Query] = []
    qid = 0

    exact_terms = [
        "player", "tournament", "software", "algorithm", "recipe", "restaurant",
        "hotel", "destination", "vaccine", "hospital", "inflation", "portfolio",
        "climate", "experiment", "coach", "chip", "chef", "flight",
        "fitness", "investor", "genome", "stadium",
    ]
    for i in range(N_EXACT):
        term = exact_terms[i % len(exact_terms)]
        rel = {d.id: 3 for d in corpus if term in d.terms}
        queries.append(Query(
            id=f"q{qid:03d}",
            terms=[term],
            relevant_doc_ids=rel,
            query_type="exact",
        ))
        qid += 1

    for i in range(N_SYNONYM):
        qterm = SYNONYMS[i % 3]
        others = [s for s in SYNONYMS if s != qterm]
        rel: dict[str, int] = {}
        for d in corpus:
            has_q = qterm in d.terms
            has_syn = any(o in d.terms for o in others)
            if has_syn and not has_q:
                rel[d.id] = 3
            elif has_q:
                rel[d.id] = 1
        queries.append(Query(
            id=f"q{qid:03d}",
            terms=[qterm],
            relevant_doc_ids=rel,
            query_type="synonym",
        ))
        qid += 1

    freshness_terms = [
        "software", "recipe", "hotel", "vaccine", "inflation",
        "climate", "player", "chip", "fitness", "bank",
        "league", "cloud", "kitchen", "tour", "clinic",
    ]
    for i in range(N_FRESHNESS):
        term = freshness_terms[i % len(freshness_terms)]
        rel = {}
        for d in corpus:
            if term not in d.terms:
                continue
            days = d.days_since_publish
            if days <= 7:
                rel[d.id] = 3
            elif days <= 30:
                rel[d.id] = 2
            elif days <= 90:
                rel[d.id] = 1
        queries.append(Query(
            id=f"q{qid:03d}",
            terms=[term],
            relevant_doc_ids=rel,
            query_type="freshness",
        ))
        qid += 1

    assert len(queries) == N_QUERIES
    dest = path or QUERIES_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "id": q.id,
            "terms": q.terms,
            "relevant_doc_ids": q.relevant_doc_ids,
            "query_type": q.query_type,
        }
        for q in queries
    ]
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return queries


def load(path: Path | None = None) -> list[Query]:
    """Load persisted queries, generating them if needed."""
    dest = path or QUERIES_PATH
    if not dest.exists():
        return generate(dest)
    raw = json.loads(dest.read_text(encoding="utf-8"))
    return [
        Query(
            id=row.get("id", ""),
            terms=row["terms"],
            relevant_doc_ids={k: int(v) for k, v in row["relevant_doc_ids"].items()},
            query_type=row["query_type"],
        )
        for row in raw
    ]


def ensure_queries() -> list[Query]:
    """Create the queries JSON if it does not already exist."""
    if QUERIES_PATH.exists():
        return load()
    return generate()


def baseline_ndcg(queries: list[Query] | None = None, corpus: list[Document] | None = None) -> float:
    """Mean NDCG@10 of the baseline Scorer over the golden queries."""
    if corpus is None:
        corpus = load_corpus()
    if queries is None:
        queries = load()
    return _mean_ndcg(queries, corpus, Scorer())


def _rank(query: Query, corpus: list[Document], scorer: Scorer) -> list[Document]:
    return sorted(
        corpus,
        key=lambda d: scorer.score_document(query.terms, d),
        reverse=True,
    )


def _mean_ndcg(queries: list[Query], corpus: list[Document], scorer: Scorer) -> float:
    scores = [_ndcg_query(q, corpus, scorer) for q in queries]
    return sum(scores) / len(scores) if scores else 0.0


def _ndcg_query(query: Query, corpus: list[Document], scorer: Scorer) -> float:
    ranked = _rank(query, corpus, scorer)
    gained = [query.relevant_doc_ids.get(d.id, 0) for d in ranked[:10]]
    ideal = sorted(query.relevant_doc_ids.values(), reverse=True)[:10]
    idcg = _dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return _dcg(gained) / idcg


def _dcg(relevances: list[int]) -> float:
    total = 0.0
    for i, rel in enumerate(relevances[:10], start=1):
        total += (2 ** rel - 1) / math.log2(i + 1)
    return total
