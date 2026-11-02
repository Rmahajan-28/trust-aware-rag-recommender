"""Compare pure-relevance ranking (alpha=1.0) with trust-aware ranking (alpha=0.6).

Tracks where the three low-trust clickbait videos (ids 7, 19, 26, whose reviews are mostly
spam in the seed data) land in each ranking, shown as "rank under pure relevance ->
rank under trust-aware". Seed data and labels are hand-made, so this is a sanity check
of the mechanism, not a claim about real-world accuracy.

Explanations are switched off here: this script only needs the ranking, so it must not
spend Groq quota (one LLM call per search otherwise).
"""
from backend.recommender import Recommender

QUERIES = ["data structures", "machine learning", "python programming", "neural networks",
           "algorithms", "rag retrieval", "sql databases", "web development"]
LOW_TRUST_IDS = (7, 19, 26)


def no_explain(query, items):
    return {it["id"]: {"text": "", "source": "template"} for it in items}


base = Recommender(alpha=1.0, explain_fn=no_explain)
aware = Recommender(alpha=0.6, explain_fn=no_explain)
K = len(base.resources)  # rank over the whole catalogue


def ranks(rec, q):
    ids = [x["id"] for x in rec.search(q, k=K)]
    return {i: (ids.index(i) + 1 if i in ids else None) for i in LOW_TRUST_IDS}


def cell(a, b):
    if a is None and b is None:
        return "-"
    return f"{a if a else '-'}->{b if b else '-'}"


header = "".join(f"{'id=' + str(i):>10}" for i in LOW_TRUST_IDS)
print(f"{'query':22}{header}   (pure -> trust-aware rank; '-' = not retrieved)")
for q in QUERIES:
    p, t = ranks(base, q), ranks(aware, q)
    print(f"{q:22}" + "".join(f"{cell(p[i], t[i]):>10}" for i in LOW_TRUST_IDS))