"""Retrieval + trust-aware re-ranking.

Stage 1 (retrieve): TF-IDF cosine similarity between the query and each resource.
Stage 2 (re-rank):  final = ALPHA * relevance + (1 - ALPHA) * trust.

Explanations are currently template-based and grounded in the retrieved data
(rating components, flagged reviews). They will be replaced by LLM-generated
RAG explanations in a later phase.
"""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .trust import trust_score

DATA = Path(__file__).parent / "data" / "resources.json"
ALPHA = 0.6  # weight on relevance vs trust


class Recommender:
    def __init__(self, path: Path = DATA, alpha: float = ALPHA):
        self.alpha = alpha
        self.resources = json.loads(Path(path).read_text())
        for r in self.resources:
            r["trust"] = trust_score(r["reviews"])
        docs = [f'{r["title"]} {r["topic"]} {r["description"]}' for r in self.resources]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(docs)

    def explain(self, r: dict, relevance: float) -> str:
        t = r["trust"]
        c = t["components"]
        parts = [f'Matches your query (relevance {relevance:.2f}).']
        parts.append(
            f'Trust {t["score"]:.2f}: {c["verified"]*100:.0f}% verified-review weight, '
            f'rating component {c["rating"]:.2f}, from {t["n_reviews"]} reviews.'
        )
        if t["n_flagged"]:
            parts.append(f'{t["n_flagged"]} suspicious review(s) were down-weighted.')
        return " ".join(parts)

    def search(self, query: str, k: int = 5, kind: str | None = None) -> list:
        if not query.strip():
            return []
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        out = []
        for r, rel in zip(self.resources, sims):
            if rel <= 0:
                continue
            if kind and r["type"] != kind:
                continue
            final = self.alpha * rel + (1 - self.alpha) * r["trust"]["score"]
            out.append({
                "id": r["id"], "title": r["title"], "type": r["type"],
                "provider": r["provider"], "relevance": round(float(rel), 3),
                "trust": r["trust"], "score": round(float(final), 3),
                "explanation": self.explain(r, float(rel)),
            })
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:k]
