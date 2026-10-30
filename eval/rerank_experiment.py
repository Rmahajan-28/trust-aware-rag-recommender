"""Compare pure-relevance ranking (alpha=1.0) with trust-aware ranking (alpha=0.6).

Measures where the low-trust resource (id=7, whose reviews are mostly spam in the
seed data) lands in each ranking. Seed data and labels are hand-made, so this is a
sanity check of the mechanism, not a claim about real-world accuracy.
"""
from backend.recommender import Recommender

QUERIES = ["data structures", "machine learning", "python programming", "neural networks",
           "algorithms", "rag retrieval", "sql databases", "web development"]
LOW_TRUST_ID = 7

base, aware = Recommender(alpha=1.0), Recommender(alpha=0.6)


def rank(rec, q):
    ids = [x["id"] for x in rec.search(q, k=12)]
    return ids.index(LOW_TRUST_ID) + 1 if LOW_TRUST_ID in ids else "-"


print(f"{'query':22} {'rank@pure':>10} {'rank@trust':>11}  (rank of low-trust item id=7; '-' = not retrieved)")
for q in QUERIES:
    print(f"{q:22} {str(rank(base, q)):>10} {str(rank(aware, q)):>11}")
