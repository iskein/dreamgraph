"""NDCG@10 evaluation of a scorer source against the golden set."""

from __future__ import annotations

import math
from typing import Any

from data.corpus import Document, load as load_corpus


def run(scorer_source: str, golden: list) -> float:
    """Exec `scorer_source`, score every golden query, return mean NDCG@10."""
    scorer = _instantiate_scorer(scorer_source)
    corpus = load_corpus()
    if not golden:
        return 0.0
    return sum(_ndcg_query(scorer, query, corpus) for query in golden) / len(golden)


def dcg_at_10(relevances: list[int]) -> float:
    """DCG@10 = sum((2^rel_i - 1) / log2(i + 1) for i in 1..10)."""
    total = 0.0
    for i, rel in enumerate(relevances[:10], start=1):
        total += (2 ** rel - 1) / math.log2(i + 1)
    return total


def ndcg_at_10(ranked_relevances: list[int], all_relevances: list[int]) -> float:
    """NDCG@10 = DCG@10 / IDCG@10. Zero if there are no relevant documents."""
    idcg = dcg_at_10(sorted(all_relevances, reverse=True))
    if idcg == 0.0:
        return 0.0
    return dcg_at_10(ranked_relevances) / idcg


def _ndcg_query(scorer: Any, query: Any, corpus: list[Document]) -> float:
    ranked = sorted(
        corpus,
        key=lambda d: scorer.score_document(query.terms, d),
        reverse=True,
    )
    gained = [query.relevant_doc_ids.get(d.id, 0) for d in ranked[:10]]
    return ndcg_at_10(gained, list(query.relevant_doc_ids.values()))


def _instantiate_scorer(scorer_source: str) -> Any:
    namespace: dict[str, Any] = {}
    exec(compile(scorer_source, "<scorer>", "exec"), namespace)
    return namespace["Scorer"]()
