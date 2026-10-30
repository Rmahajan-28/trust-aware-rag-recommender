"""Trust scoring for educational resources.

A resource's trust score (0-1) combines review quality, not just average rating:
  - verified_ratio:   share of reviews from verified learners
  - helpfulness:      log-scaled helpful votes per review (saturating)
  - rating:           mean rating normalised to 0-1
  - volume:           more reviews -> more confidence (saturating)
  - spam_penalty:     reviews that are very short, all-caps / exclamation heavy,
                      or promotional are down-weighted

Weights are deliberately simple and transparent so they can be tuned/validated later.
"""
import math
import re

WEIGHTS = {"rating": 0.35, "verified": 0.25, "helpful": 0.20, "volume": 0.20}
PROMO_WORDS = ("buy now", "subscribe", "guaranteed", "scam", "click", "free money")


def review_spam_score(review: dict) -> float:
    """Return 0 (clean) .. 1 (very likely spam) using cheap heuristics."""
    text = review.get("text", "")
    score = 0.0
    if len(text.split()) < 4:
        score += 0.3
    if text.count("!") >= 3:
        score += 0.3
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        score += 0.2
    if any(w in text.lower() for w in PROMO_WORDS):
        score += 0.4
    if not review.get("verified", False) and review.get("helpful_votes", 0) == 0:
        score += 0.2
    return min(score, 1.0)


def trust_score(reviews: list) -> dict:
    if not reviews:
        return {"score": 0.0, "components": {}, "n_reviews": 0, "n_flagged": 0}

    # weight each review by (1 - spam) so suspicious reviews count less
    weights = [1.0 - review_spam_score(r) for r in reviews]
    total_w = sum(weights) or 1e-9
    mean_rating = sum(r["rating"] * w for r, w in zip(reviews, weights)) / total_w
    rating = (mean_rating - 1) / 4  # 1..5 -> 0..1

    verified = sum(w for r, w in zip(reviews, weights) if r.get("verified")) / total_w
    avg_votes = sum(r.get("helpful_votes", 0) for r in reviews) / len(reviews)
    helpful = min(math.log1p(avg_votes) / math.log1p(50), 1.0)
    effective_n = sum(weights)
    volume = min(math.log1p(effective_n) / math.log1p(6), 1.0)

    comps = {"rating": rating, "verified": verified, "helpful": helpful, "volume": volume}
    score = sum(WEIGHTS[k] * v for k, v in comps.items())
    flagged = sum(1 for r in reviews if review_spam_score(r) >= 0.5)
    return {
        "score": round(score, 3),
        "components": {k: round(v, 3) for k, v in comps.items()},
        "n_reviews": len(reviews),
        "n_flagged": flagged,
    }
