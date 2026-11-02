# TrustRec: Trust-Aware Educational Resource Recommendation (Mid-Term Prototype)

B.Tech Major Project 2026-27, IILM University. Project ID BT267.

A recommender for courses, books and videos that ranks results by **relevance and trust**, where trust is computed from review quality (verified learners, helpful votes, spam-like reviews down-weighted) instead of a raw star average. Each result comes with an explanation written from the reviews that support it (retrieval-augmented generation), and the evidence is shown next to it.

## Status (honest)

| Component | Status |
|---|---|
| FastAPI backend (`/api/search`, `/api/health`) | Working |
| Embedding retrieval (ChromaDB, `all-MiniLM-L6-v2`, cosine) over a 27-resource seed dataset (18 videos, 5 courses, 4 books) | Working |
| Every review embedded as its own entry | Working |
| Review trust scoring (heuristic spam detection + weighted components) | Working |
| Trust-aware re-ranking (`0.6 * relevance + 0.4 * trust`) | Working |
| Evidence retrieval (top 3 non-spam reviews most relevant to the query, per result) | Working |
| LLM explanations via Groq: one batched call per search, citations checked, template fallback, cache | Working (needs `GROQ_API_KEY`; without it every result uses the template) |
| Frontend (results, trust bar, explanation with source label, expandable evidence, flagged-review warning) | Working |
| Real review data (currently hand-written seed data) | **Not started** |
| Personalization (learner profile, skill-gap analysis, collaborative filtering) | **Not started** |
| Evaluation (Precision@k, Recall@k, NDCG, explanation faithfulness) and user study | **Not started** (only the re-ranking experiment below exists) |

## Architecture

```
frontend (HTML/CSS/JS)  --fetch-->  FastAPI (backend/main.py)
                                       |
                                       v
                         Recommender (backend/recommender.py)
                          1. embed query, retrieve resources      (Chroma "resources")
                          2. re-rank: alpha*relevance + (1-alpha)*trust
                          3. evidence: top reviews per result,    (Chroma "reviews")
                             spam-flagged reviews excluded
                          4. explain: one Groq call for all results,
                             template fallback per item
                                |                     |
                                v                     v
                     Trust scorer              Explainer (backend/explainer.py)
                     (backend/trust.py)        prompt, citation check, cache
                     rating, verified ratio,
                     helpfulness, volume,
                     per-review spam down-weighting
```

An explanation from the model is accepted only if it is short, is valid JSON for that resource, and every `[n]` citation points at a review that was actually supplied. Otherwise that result falls back to the template. The UI labels each explanation as "AI-written from reviews" or "template".

## Run

```bash
pip install -r requirements.txt
# optional: LLM explanations (free key from console.groq.com)
export GROQ_API_KEY=...        # or put GROQ_API_KEY=... in a .env file in the repo root (gitignored)
uvicorn backend.main:app --reload
# open http://127.0.0.1:8000
```

The first start downloads the MiniLM model and builds `backend/chroma_store/`. On later starts, resources and reviews that are missing from the store are added automatically. If you **edit an existing resource or review** in `resources.json`, delete `backend/chroma_store/` so it is rebuilt.

`GROQ_MODEL` overrides the model (default `llama-3.1-8b-instant`).

## Tests and experiment

```bash
python -m pytest -q                 # 36 tests
python -m eval.rerank_experiment    # pure-relevance vs trust-aware ranking of the 3 clickbait videos
```

Most tests run offline: they use a small hashed bag-of-words embedding and a fake LLM, so they need no model download and never call Groq. `tests/test_api.py` uses the real MiniLM model.

## Known limitations

- Seed data is hand-written (27 resources, 2-3 reviews each), so results are a mechanism check, not evidence of real-world accuracy.
- Spam detection is rule-based (length, exclamation marks, promo words, unverified with zero votes) and will miss subtle fake reviews.
- Weights in `trust.py` are hand-set and not yet tuned or validated against labelled data. `MIN_RELEVANCE` in `recommender.py` is an untuned guess.
- The citation check only proves a cited review exists, not that the sentence is faithful to it. Faithfulness still has to be measured.
- Groq's free tier is rate limited. When a call fails, results fall back to template explanations.
- Results depend only on the query: there is no learner profile or skill-gap input yet.