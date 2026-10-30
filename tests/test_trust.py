from backend.trust import review_spam_score, trust_score


def good(rating=5, votes=20):
    return {"rating": rating, "text": "Clear explanations and well structured exercises.", "verified": True, "helpful_votes": votes}


SPAM = {"rating": 5, "text": "Best course ever!!! Buy now!!!", "verified": False, "helpful_votes": 0}


def test_spam_review_is_flagged():
    assert review_spam_score(SPAM) >= 0.5


def test_clean_review_not_flagged():
    assert review_spam_score(good()) < 0.3


def test_empty_reviews_score_zero():
    assert trust_score([])["score"] == 0.0


def test_spam_lowers_trust_vs_clean_only():
    clean = trust_score([good(), good(4), good()])
    mixed = trust_score([good(), good(4), good(), SPAM, SPAM])
    assert mixed["n_flagged"] == 2
    # spam reviews are down-weighted, so score must not exceed the clean score
    assert mixed["score"] <= clean["score"] + 1e-9


def test_all_spam_scores_below_all_clean():
    spam_only = trust_score([SPAM, SPAM, SPAM])
    clean_only = trust_score([good(), good(), good()])
    assert spam_only["score"] < clean_only["score"]


def test_score_bounded():
    s = trust_score([good(5, 1000)] * 30)["score"]
    assert 0.0 <= s <= 1.0
