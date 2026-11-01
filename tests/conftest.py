"""Shared test fixtures.

`rec` builds a Recommender on in-memory ChromaDB collections with a tiny hashed
bag-of-words embedding, so these tests are fast and need no model download. The real
MiniLM embeddings are exercised by tests/test_api.py.
"""
import re
import uuid
import zlib

import chromadb
import numpy as np
import pytest
from chromadb.api.types import EmbeddingFunction

DIM = 512


class HashEmbedding(EmbeddingFunction):
    """Deterministic bag-of-words embedding: shared words -> higher cosine similarity."""

    def __init__(self):
        pass

    def __call__(self, input):
        out = []
        for text in input:
            v = np.zeros(DIM, dtype=np.float32)
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                v[zlib.crc32(tok.encode()) % DIM] += 1.0
            v[0] += 0.01  # keep the vector non-zero
            out.append(v / np.linalg.norm(v))
        return out

    @staticmethod
    def name():
        return "hash-bow-test"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return HashEmbedding()


@pytest.fixture(autouse=True)
def no_real_llm(monkeypatch):
    """Tests never call Groq, even if a key is set in the environment or a .env file."""
    from backend import explainer

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(explainer, "ENV_FILE", explainer.ENV_FILE.with_name(".env.does-not-exist"))
    explainer._cache.clear()


@pytest.fixture
def rec():
    from backend.recommender import Recommender

    client = chromadb.EphemeralClient()
    ef = HashEmbedding()
    tag = uuid.uuid4().hex[:8]
    meta = {"hnsw:space": "cosine"}
    resources = client.get_or_create_collection(f"resources_{tag}", embedding_function=ef, metadata=meta)
    reviews = client.get_or_create_collection(f"reviews_{tag}", embedding_function=ef, metadata=meta)
    return Recommender(collection=resources, reviews_collection=reviews)