"""Retrieval + trust-aware re-ranking using the ChromaDB vector store.

relevance = 1 - cosine distance; final = ALPHA * relevance + (1 - ALPHA) * trust.
Explanations are still template-based (RAG comes later).
"""
import json

from . import store
from .trust import trust_score

ALPHA = 0.6
MIN_RELEVANCE = 0.25  # untuned guess


class Recommender:
    def __init__(self, alpha: float = ALPHA, collection=None, seed_path=store.DATA):
        self.alpha = alpha
        self.col = collection if collection is not None else store.get_collection()
        if self.col.count() == 0:
            store.load_seed(self.col, seed_path)
        got = self.col.get(include=["metadatas"])
        self.resources = []
        for m in got["metadatas"]:
            r = {
                "id": m["rid"], "title": m["title"], "type": m["type"],
                "provider": m["provider"], "reviews": json.loads(m["reviews_json"]),
            }
            r["trust"] = trust_score(r["reviews"])
            self.resources.append(r)
        self.by_id = {r["id"]: r for r in self.resources}

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
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(
            query_texts=[query],
            n_results=n,
            where={"type": kind} if kind else None,
            include=["metadatas", "distances"],
        )
        out = []
        for m, dist in zip(res["metadatas"][0], res["distances"][0]):
            rel = 1.0 - dist
            if rel <= MIN_RELEVANCE:
                continue
            r = self.by_id[m["rid"]]
            final = self.alpha * rel + (1 - self.alpha) * r["trust"]["score"]
            out.append({
                "id": r["id"], "title": r["title"], "type": r["type"],
                "provider": r["provider"], "relevance": round(float(rel), 3),
                "trust": r["trust"], "score": round(float(final), 3),
                "explanation": self.explain(r, float(rel)),
            })
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:k]