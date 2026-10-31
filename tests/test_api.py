import json

from fastapi.testclient import TestClient

from backend import store
from backend.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["resources"] == len(json.loads(store.DATA.read_text()))


def test_search_returns_relevant_top_result():
    r = client.get("/api/search", params={"q": "machine learning", "k": 3})
    assert r.status_code == 200
    titles = [x["title"] for x in r.json()["results"]]
    assert any("Machine Learning" in t for t in titles)


def test_kind_filter():
    r = client.get("/api/search", params={"q": "machine learning", "kind": "book"})
    assert all(x["type"] == "book" for x in r.json()["results"])


def test_no_match_returns_empty():
    r = client.get("/api/search", params={"q": "zzzzqqq"})
    assert r.json()["results"] == []


def test_missing_query_rejected():
    assert client.get("/api/search").status_code == 422


def test_frontend_served():
    r = client.get("/")
    assert r.status_code == 200 and "TrustRec" in r.text


def test_results_include_evidence():
    r = client.get("/api/search", params={"q": "machine learning", "k": 3})
    for x in r.json()["results"]:
        assert 1 <= len(x["evidence"]) <= 3
