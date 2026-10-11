"""Aggregate metrics from a completed agent run."""

from __future__ import annotations

from typing import Any


def aggregate(run_result: Any) -> dict:
    """Collect ndcg@10, p95 latency, and experiment/session counters.

    `run_result` is duck-typed. Expected attributes (all optional):
    `current_ndcg` / `ndcg`, `p95_latency`, `sessions_used`, `history`,
    `memory.experiments`, `memory.skips`.
    """
    memory = getattr(run_result, "memory", run_result)
    experiments = list(
        getattr(memory, "experiments", None)
        or getattr(run_result, "experiments", None)
        or []
    )
    skips = getattr(memory, "skips", None)
    if skips is None:
        skips = getattr(run_result, "skips", [])

    kept = sum(1 for e in experiments if e.get("verdict") == "keep")
    reverted = sum(1 for e in experiments if e.get("verdict") == "revert")
    skipped_in_ledger = sum(1 for e in experiments if e.get("verdict") == "skip")
    if isinstance(skips, int):
        skipped = skips
    else:
        skipped = len(list(skips)) + skipped_in_ledger

    sessions_used = getattr(run_result, "sessions_used", None)
    if sessions_used is None:
        history = getattr(run_result, "history", None)
        if history is not None:
            sessions_used = len(history)
        else:
            session_ids = {e.get("session_id") for e in experiments if "session_id" in e}
            sessions_used = len(session_ids)

    ndcg = getattr(run_result, "current_ndcg", None)
    if ndcg is None:
        ndcg = getattr(run_result, "ndcg", 0.0)

    latency = getattr(run_result, "p95_latency", None)
    if latency is None:
        latency = getattr(run_result, "baseline_latency", 0.0)

    return {
        "ndcg@10": ndcg,
        "p95_latency": latency,
        "sessions_used": sessions_used,
        "experiments_run": len(experiments),
        "experiments_kept": kept,
        "experiments_reverted": reverted,
        "experiments_skipped": skipped,
    }
