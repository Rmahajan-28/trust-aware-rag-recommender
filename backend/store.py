"""ChromaDB vector store for resources."""
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DATA = Path(__file__).parent / "data" / "resources.json"
STORE_DIR = Path(__file__).parent / "chroma_store"
COLLECTION = "resources"
EMBED_MODEL = "all-MiniLM-L6-v2"


def default_embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def get_collection(embedding_fn=None, path: Path = STORE_DIR):
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embedding_fn or default_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def load_seed(col, seed_path: Path = DATA, reset: bool = False) -> int:
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