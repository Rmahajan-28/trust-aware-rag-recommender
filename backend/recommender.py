"""Retrieval + trust-aware re-ranking using the ChromaDB vector store.

relevance = 1 - cosine distance; final = ALPHA * relevance + (1 - ALPHA) * trust.

Each result also carries `evidence`: the reviews of that resource most relevant to the
query, retrieved from the per-review collection with suspicious (spam-flagged) reviews
excluded. This is the retrieval half of RAG; the explanation is still template-based
and the LLM step will be written on top of `evidence`.
"""
import json

from . import store
from .trust import SPAM_THRESHOLD, trust_score

ALPHA = 0.6
MIN_RELEVANCE = 0.25  # untuned guess
EVIDENCE_K = 3


class Recommender:
    def __init__(self, alpha: float = ALPHA, collection=None, seed_path=store.DATA,
                 reviews_collection=None):
        self.alpha = alpha
        self.col = collection if collection is not None else store.get_collection()
        if reviews_collection is not None:
            self.rcol = reviews_collection
        elif collection is None:
            self.rcol = store.get_reviews_collection()
        else:
            self.rcol = None  # custom resource collection without reviews -> no evidence
        # idempotent: only adds items missing from an existing (older) store
        store.load_seed(self.col, seed_path)
        if self.rcol is not None:
            store.load_reviews(self.rcol, seed_path)
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

    def evidence(self, query: str, rid: int, n: int = EVIDENCE_K) -> list:
        """Top-n reviews of resource `rid` most relevant to `query`, spam excluded."""
        if self.rcol is None or not query.strip():
            return []
        total = len(self.by_id[rid]["reviews"])
        if total == 0:
            return []
        res = self.rcol.query(
            query_texts=[query],
            n_results=total,
            where={"rid": rid},
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if m["spam"] >= SPAM_THRESHOLD:
                continue
            out.append({
                "text": doc, "rating": m["rating"], "verified": m["verified"],
                "helpful_votes": m["helpful_votes"],
                "relevance": round(float(1.0 - dist), 3),
            })
            if len(out) == n:
                break
        return out

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
        out = out[:k]
        # evidence only for the results actually returned (one small query each)
        for item in out:
            item["evidence"] = self.evidence(query, item["id"])
        return out
