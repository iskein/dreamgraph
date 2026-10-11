from data import ensure_generated
from data.corpus import load as load_corpus
from data.golden import load as load_golden, ranking_diff
from data.queries import load as load_queries, baseline_ndcg
from data.scorer import Scorer


def test_ensure_generated_sizes():
    ensure_generated()
    docs = load_corpus()
    queries = load_queries()
    assert len(docs) == 500
    assert len(queries) == 200
    assert all(hasattr(d, "terms") and hasattr(d, "days_since_publish") for d in docs)
    types = {q.query_type for q in queries}
    assert types == {"exact", "synonym", "freshness"}


def test_synonym_overlap():
    docs = load_corpus()
    counts = {s: sum(1 for d in docs if s in d.terms) for s in ("car", "automobile", "vehicle")}
    assert all(n >= 30 for n in counts.values())
    both = sum(1 for d in docs if "car" in d.terms and "automobile" in d.terms)
    assert both >= 10


def test_baseline_ndcg_near_700():
    ndcg = baseline_ndcg()
    assert abs(ndcg - 0.700) < 0.01


def test_ranking_diff_counts_position_changes():
    a = [["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9"]]
    b = [["d0", "d2", "d1", "d3", "d4", "d5", "d6", "d7", "d8", "d9"]]
    assert ranking_diff(a, b) == 2
    assert ranking_diff(a, a) == 0


def test_scorer_baseline_weights():
    class _Doc:
        terms = ["alpha"]
        days_since_publish = 0

    s = Scorer()
    assert s.score_keyword(["alpha", "beta"], _Doc) == 1.0
    assert s.score_freshness(_Doc) == 1.0
    assert abs(s.score_document(["alpha"], _Doc) - 1.0) < 1e-9


def test_golden_load():
    golden = load_golden()
    assert len(golden) == 200
    assert all(1 <= v <= 3 for q in golden for v in q.relevant_doc_ids.values())
