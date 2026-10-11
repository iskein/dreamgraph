"""p95 query latency for a scorer source, in milliseconds."""

from __future__ import annotations

import statistics
import time
from typing import Any

import numpy as np

from data.corpus import load as load_corpus
from data.golden import load as load_golden
from eval.ndcg import _instantiate_scorer

N_REPEATS = 10


def run(scorer_source: str) -> float:
    """Time each query 10 times, take the median, then p95 across queries.

    Each timed call scores every corpus document via `score_document`.
    Result is milliseconds.
    """
    scorer = _instantiate_scorer(scorer_source)
    corpus = load_corpus()
    queries = load_golden()
    medians: list[float] = []
    for query in queries:
        samples = [_score_query_ms(scorer, query.terms, corpus) for _ in range(N_REPEATS)]
        medians.append(statistics.median(samples))
    return float(np.percentile(medians, 95))


def _score_query_ms(scorer: Any, query_terms: list[str], corpus: list) -> float:
    start = time.perf_counter()
    for doc in corpus:
        scorer.score_document(query_terms, doc)
    return (time.perf_counter() - start) * 1000.0
