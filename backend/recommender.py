"""Retrieval + trust-aware re-ranking.

Stage 1 (retrieve): cosine similarity between the query embedding and each
    resource embedding (sentence-transformers), so matching is by meaning,
    not just shared words.
Stage 2 (re-rank):  final = ALPHA * relevance + (1 - ALPHA) * trust.

Explanations are currently template-based and grounded in the retrieved data
(rating components, flagged reviews). They will be replaced by LLM-generated
RAG explanations in a later phase.
"""
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from .trust import trust_score

DATA = Path(__file__).parent / "data" / "resources.json"
ALPHA = 0.6  # weight on relevance vs trust
EMBED_MODEL = "all-MiniLM-L6-v2"
MIN_RELEVANCE = 0.25  # untuned guess: embedding similarity is rarely exactly 0


class Recommender:
    def __init__(self, path: Path = DATA, alpha: float = ALPHA):
        self.alpha = alpha
        self.resources = json.loads(Path(path).read_text())
        for r in self.resources:
            r["trust"] = trust_score(r["reviews"])
        docs = [f'{r["title"]} {r["topic"]} {r["description"]}' for r in self.resources]
        self.model = SentenceTransformer(EMBED_MODEL)
        self.embeddings = self.model.encode(docs, normalize_embeddings=True)

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
        q = self.model.encode([query], normalize_embeddings=True)
        sims = (self.embeddings @ q.T).ravel()  # unit vectors, so dot product = cosine
        out = []
        for r, rel in zip(self.resources, sims):
            if rel <= MIN_RELEVANCE:
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