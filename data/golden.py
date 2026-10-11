"""Fixed golden query set and ranking-diff helper."""

from __future__ import annotations

from data.queries import Query, load as load_queries


def load() -> list[Query]:
    """Return the 200 golden queries with relevance labels.

    The set is created once (see data.ensure_generated) and never modified
    during a run.
    """
    from data import ensure_generated

    ensure_generated()
    return load_queries()


def ranking_diff(
    baseline_rankings: list[list[str]],
    current_rankings: list[list[str]],
) -> int:
    """Sum of top-10 position changes across queries.

    For each query, count how many of the 10 ranked slots differ between
    the baseline ranking and the current ranking, then sum over queries.
    """
    total = 0
    for base, curr in zip(baseline_rankings, current_rankings):
        base_top = list(base[:10])
        curr_top = list(curr[:10])
        for i in range(10):
            b = base_top[i] if i < len(base_top) else None
            c = curr_top[i] if i < len(curr_top) else None
            if b != c:
                total += 1
    return total
