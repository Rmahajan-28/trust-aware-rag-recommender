# TrustRec: Trust-Aware Educational Resource Recommendation (Mid-Term Prototype)

B.Tech Major Project 2026-27, IILM University. Project ID BT267.

A recommender for courses, books and videos that ranks results by **relevance and trust**, where trust is computed from review quality (verified learners, helpful votes, spam-like reviews down-weighted) instead of a raw star average.

## Status (honest)

| Component | Status |
|---|---|
| FastAPI backend (`/api/search`, `/api/health`) | Working |
| TF-IDF retrieval over a 12-resource seed dataset | Working |
| Review trust scoring (heuristic spam detection + weighted components) | Working |
| Trust-aware re-ranking (`0.6 * relevance + 0.4 * trust`) | Working |
| Frontend search screen (results, trust bar, explanation, flagged-review warning) | Working |
| Template-based explanations | Working (placeholder for RAG) |
| LLM-based RAG explanations | **Not started** |
| Embedding-based retrieval / vector store | **Not started** |
| Real review data (currently hand-written seed data) | **Not started** |
| User personalization | **Not started** |

## Architecture

```
frontend (HTML/CSS/JS)  --fetch-->  FastAPI (backend/main.py)
                                       |
                                       v
                         Recommender (backend/recommender.py)
                          1. TF-IDF retrieve (scikit-learn)
                          2. re-rank: alpha*relevance + (1-alpha)*trust
                                       |
                                       v
                           Trust scorer (backend/trust.py)
                           rating, verified ratio, helpfulness, volume,
                           with per-review spam down-weighting
```

## Run

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
# open http://127.0.0.1:8000
```

## Tests and experiment

```bash
python -m pytest -q                 # 12 tests
python -m eval.rerank_experiment    # pure-relevance vs trust-aware ranking
```

## Known limitations

- Seed data is hand-written (12 resources), so results are a mechanism check, not evidence of real-world accuracy.
- Spam detection is rule-based (length, exclamation marks, promo words, unverified with zero votes) and will miss subtle fake reviews.
- Weights in `trust.py` are hand-set and not yet tuned or validated against labelled data.
- Retrieval is keyword-based (TF-IDF), so it cannot match on meaning yet.
