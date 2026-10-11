import math


class Scorer:
    def score_keyword(self, query_terms, doc):
        """BM25 term overlap. Baseline: plain term frequency."""
        score = 0.0
        for term in query_terms:
            if term in doc.terms:
                score += 1.0
        return score

    def score_freshness(self, doc):
        """Exponential decay. Baseline: 30-day constant."""
        return math.exp(-doc.days_since_publish / 30.0)

    def score_document(self, query_terms, doc):
        """Combination. Baseline: 0.7 * keyword + 0.3 * freshness."""
        return 0.7 * self.score_keyword(query_terms, doc) \
             + 0.3 * self.score_freshness(doc)
