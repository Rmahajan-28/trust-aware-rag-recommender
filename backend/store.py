"""ChromaDB vector store: one collection of resources, one of individual reviews."""
import json
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from .trust import review_spam_score

DATA = Path(__file__).parent / "data" / "resources.json"
STORE_DIR = Path(__file__).parent / "chroma_store"
COLLECTION = "resources"
REVIEWS_COLLECTION = "reviews"
EMBED_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def default_embedding_fn():
    # cached so the model loads once even though two collections use it
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def get_collection(embedding_fn=None, path: Path = STORE_DIR):
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embedding_fn or default_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def get_reviews_collection(embedding_fn=None, path: Path = STORE_DIR):
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(
        name=REVIEWS_COLLECTION,
        embedding_function=embedding_fn or default_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def load_seed(col, seed_path: Path = DATA, reset: bool = False) -> int:
    """Add any resources not yet in the collection (idempotent)."""
    resources = json.loads(Path(seed_path).read_text())
    if reset and col.count():
        col.delete(ids=col.get()["ids"])
    existing = set(col.get()["ids"])
    new = [r for r in resources if str(r["id"]) not in existing]
    if not new:
        return 0
    col.add(
        ids=[str(r["id"]) for r in new],
        documents=[f'{r["title"]} {r["topic"]} {r["description"]}' for r in new],
        metadatas=[{
            "rid": int(r["id"]), "title": r["title"], "type": r["type"],
            "provider": r["provider"], "topic": r["topic"],
            "description": r["description"],
            "reviews_json": json.dumps(r["reviews"]),
        } for r in new],
    )
    return len(new)


def load_reviews(rcol, seed_path: Path = DATA, reset: bool = False) -> int:
    """Embed every review as its own entry (idempotent). Id = "<resource id>-<review index>".

    The spam score is stored in metadata so evidence retrieval can skip suspicious
    reviews without re-scoring.
    """
    resources = json.loads(Path(seed_path).read_text())
    if reset and rcol.count():
        rcol.delete(ids=rcol.get()["ids"])
    existing = set(rcol.get()["ids"])
    ids, docs, metas = [], [], []
    for r in resources:
        for i, rev in enumerate(r["reviews"]):
            rid = f'{r["id"]}-{i}'
            if rid in existing:
                continue
            ids.append(rid)
            docs.append(rev["text"])
            metas.append({
                "rid": int(r["id"]), "idx": i, "rating": int(rev["rating"]),
                "verified": bool(rev.get("verified", False)),
                "helpful_votes": int(rev.get("helpful_votes", 0)),
                "spam": float(review_spam_score(rev)),
            })
    if not ids:
        return 0
    rcol.add(ids=ids, documents=docs, metadatas=metas)
    return len(ids)
