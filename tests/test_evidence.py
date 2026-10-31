import json

from backend import store
from backend.trust import SPAM_THRESHOLD, review_spam_score

SEED = json.loads(store.DATA.read_text())
TOTAL_REVIEWS = sum(len(r["reviews"]) for r in SEED)
QUERIES = ["machine learning", "data structures", "neural networks", "sql databases", "web development"]


def test_every_review_is_embedded_once(rec):
    assert rec.rcol.count() == TOTAL_REVIEWS
    # loading again must not duplicate anything
    assert store.load_reviews(rec.rcol) == 0
    assert rec.rcol.count() == TOTAL_REVIEWS


def test_seed_has_enough_videos():
    videos = [r for r in SEED if r["type"] == "video"]
    assert len(videos) >= 15


def test_video_filter_returns_several_results(rec):
    # the offline embedding only matches shared words, so pick a query several videos share
    results = rec.search("neural networks", k=5, kind="video")
    assert len(results) >= 3
    assert all(r["type"] == "video" for r in results)


def test_evidence_comes_only_from_requested_resource(rec):
    for r in SEED:
        own_texts = {rev["text"] for rev in r["reviews"]}
        for ev in rec.evidence("neural networks", r["id"]):
            assert ev["text"] in own_texts


def test_spam_never_returned_as_evidence(rec):
    spam_texts = {rev["text"] for r in SEED for rev in r["reviews"]
                  if review_spam_score(rev) >= SPAM_THRESHOLD}
    assert spam_texts, "seed data should contain spam reviews for this test to mean something"
    for r in SEED:
        for q in QUERIES:
            for ev in rec.evidence(q, r["id"], n=10):
                assert ev["text"] not in spam_texts


def test_evidence_limited_and_sorted_by_relevance(rec):
    ev = rec.evidence("machine learning", 3, n=2)
    assert len(ev) <= 2
    rels = [e["relevance"] for e in ev]
    assert rels == sorted(rels, reverse=True)


def test_search_attaches_evidence_to_every_result(rec):
    results = rec.search("machine learning", k=4)
    assert results
    for r in results:
        assert isinstance(r["evidence"], list) and 1 <= len(r["evidence"]) <= 3


def test_evidence_empty_for_blank_query(rec):
    assert rec.evidence("   ", 1) == []


def test_spam_heavy_videos_rank_below_clean_ones_in_trust(rec):
    # ids 19 and 26 are the clickbait/spam videos added to the seed data
    clean = rec.by_id[17]["trust"]["score"]
    for rid in (19, 26):
        assert rec.by_id[rid]["trust"]["n_flagged"] >= 2
        assert rec.by_id[rid]["trust"]["score"] < clean
