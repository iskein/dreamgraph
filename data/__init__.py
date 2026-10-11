"""Synthetic corpus, queries, scorer, and golden set."""


def ensure_generated() -> None:
    """Idempotently create corpus and queries JSON if they are missing."""
    from data.corpus import CORPUS_PATH, generate as generate_corpus
    from data.queries import QUERIES_PATH, generate as generate_queries

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CORPUS_PATH.exists():
        generate_corpus()
    if not QUERIES_PATH.exists():
        generate_queries()
