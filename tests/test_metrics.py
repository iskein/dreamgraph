"""Tests for eval.ndcg, eval.latency, and eval.metrics."""

import math
from pathlib import Path
from types import SimpleNamespace

from eval.metrics import aggregate
from eval.ndcg import dcg_at_10, ndcg_at_10, run as run_ndcg


def test_ndcg_hand_computed_imperfect_ranking():
    """Hand-computed NDCG@10 for ranking [3, 0, 2] with labels {3, 2}.

    DCG@10 = (2^3-1)/log2(1+1) + (2^0-1)/log2(2+1) + (2^2-1)/log2(3+1)
           = 7/1 + 0 + 3/2
           = 8.5
    IDCG@10 (ideal order [3, 2]) = 7/log2(2) + 3/log2(3)
                                 = 7 + 3/log2(3)
    NDCG@10 = 8.5 / (7 + 3/log2(3))
    """
    ranked = [3, 0, 2]
    all_rels = [3, 2]
    expected_dcg = 8.5
    expected_idcg = 7.0 + 3.0 / math.log2(3)
    expected_ndcg = expected_dcg / expected_idcg
    assert abs(dcg_at_10(ranked) - expected_dcg) < 1e-12
    assert abs(ndcg_at_10(ranked, all_rels) - expected_ndcg) < 1e-12
    assert ndcg_at_10([3, 2], [3, 2]) == 1.0
    assert ndcg_at_10([0, 0, 0], []) == 0.0


def test_ndcg_run_baseline_near_700():
    src = Path("data/scorer.py").read_text(encoding="utf-8")
    from data.golden import load as load_golden

    score = run_ndcg(src, load_golden())
    assert abs(score - 0.700) < 0.01


def test_aggregate_counts_verdicts_and_skips():
    result = SimpleNamespace(
        current_ndcg=0.741,
        p95_latency=94.0,
        sessions_used=6,
        memory=SimpleNamespace(
            experiments=[
                {"session_id": 1, "verdict": "keep"},
                {"session_id": 2, "verdict": "revert"},
                {"session_id": 3, "verdict": "keep"},
                {"session_id": 4, "verdict": "keep"},
                {"session_id": 5, "verdict": "keep"},
                {"session_id": 6, "verdict": "revert"},
            ],
            skips=[{"id": "n_gram_matching", "reason": "cost profile"}],
        ),
    )
    metrics = aggregate(result)
    assert metrics["ndcg@10"] == 0.741
    assert metrics["p95_latency"] == 94.0
    assert metrics["sessions_used"] == 6
    assert metrics["experiments_run"] == 6
    assert metrics["experiments_kept"] == 4
    assert metrics["experiments_reverted"] == 2
    assert metrics["experiments_skipped"] == 1


def test_latency_run_returns_positive_ms():
    from eval.latency import run as run_latency

    src = Path("data/scorer.py").read_text(encoding="utf-8")
    p95 = run_latency(src)
    assert p95 > 0.0
