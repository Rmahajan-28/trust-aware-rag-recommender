import json

from backend import explainer


def item(rid=1, n_ev=2, flagged=0):
    return {
        "id": rid, "title": f"Resource {rid}", "type": "course", "provider": "Test",
        "relevance": 0.71,
        "trust": {"score": 0.83, "n_reviews": 3, "n_flagged": flagged,
                  "components": {"rating": 0.9, "verified": 1.0, "helpful": 0.5, "volume": 0.6}},
        "evidence": [{"text": f"review text {i}", "rating": 5, "verified": True, "helpful_votes": 10}
                     for i in range(n_ev)],
    }


def fake(reply):
    """A stand-in for the Groq call that records how often it was used."""
    calls = []

    def complete(messages):
        calls.append(messages)
        return reply if isinstance(reply, str) or reply is None else json.dumps(reply)

    complete.calls = calls
    return complete


def test_valid_llm_output_is_used():
    c = fake({"1": "Covers the basics clearly [1] and is well rated [2]."})
    out = explainer.explain_batch("python", [item(1)], complete=c)
    assert out[1]["source"] == "llm"
    assert "[1]" in out[1]["text"]


def test_citation_to_missing_review_falls_back():
    c = fake({"1": "Great course [5]."})  # only 2 reviews exist
    out = explainer.explain_batch("python", [item(1, n_ev=2)], complete=c)
    assert out[1]["source"] == "template"


def test_no_citation_falls_back():
    c = fake({"1": "A great course for beginners."})
    assert explainer.explain_batch("python", [item(1)], complete=c)[1]["source"] == "template"


def test_overlong_output_falls_back():
    c = fake({"1": "Good [1]. " + "x" * 500})
    assert explainer.explain_batch("python", [item(1)], complete=c)[1]["source"] == "template"


def test_one_bad_item_does_not_spoil_the_others():
    c = fake({"1": "Solid [1].", "2": "Nonsense [9]."})
    out = explainer.explain_batch("python", [item(1), item(2)], complete=c)
    assert out[1]["source"] == "llm" and out[2]["source"] == "template"


def test_api_failure_falls_back_to_template():
    out = explainer.explain_batch("python", [item(1), item(2)], complete=fake(None))
    assert all(v["source"] == "template" for v in out.values())
    assert "Trust 0.83" in out[1]["text"]


def test_malformed_json_falls_back():
    for bad in ("not json", "[1, 2]", "null"):
        out = explainer.explain_batch("python", [item(1)], complete=fake(bad))
        assert out[1]["source"] == "template"


def test_item_without_evidence_skips_the_llm():
    c = fake({"1": "Good [1]."})
    out = explainer.explain_batch("python", [item(1, n_ev=0)], complete=c)
    assert out[1]["source"] == "template" and c.calls == []


def test_prompt_contains_numbered_evidence_and_query():
    c = fake({"1": "Fine [1]."})
    explainer.explain_batch("learn python", [item(1)], complete=c)
    user = c.calls[0][1]["content"]
    assert "learn python" in user and "[1]" in user and "[2]" in user and "review text 0" in user


def test_successful_reply_is_cached_but_failure_is_not():
    c = fake({"1": "Fine [1]."})
    explainer.explain_batch("python", [item(1)], complete=c)
    explainer.explain_batch("Python ", [item(1)], complete=c)  # same query after normalising
    assert len(c.calls) == 1

    bad = fake(None)
    explainer.explain_batch("sql", [item(2)], complete=bad)
    explainer.explain_batch("sql", [item(2)], complete=bad)
    assert len(bad.calls) == 2  # a failed call must be retried next time


def test_no_key_means_no_network_call():
    assert explainer.groq_complete([{"role": "user", "content": "hi"}]) is None


def test_search_uses_explainer_and_reports_source(rec):
    rec.explain_fn = lambda q, items: {
        it["id"]: {"text": f"explained {it['id']} [1]", "source": "llm"} for it in items
    }
    results = rec.search("machine learning", k=3)
    assert results
    for r in results:
        assert r["explanation"] == f"explained {r['id']} [1]"
        assert r["explanation_source"] == "llm"


def test_search_without_key_still_returns_template_explanations(rec):
    results = rec.search("machine learning", k=3)
    assert results
    assert all(r["explanation_source"] == "template" and r["explanation"] for r in results)


class _OkResponse:
    status_code = 200

    def json(self):
        return {"choices": [{"message": {"content": "{}"}}]}


def _capture_post(monkeypatch):
    """Replace the HTTP call so we can see exactly what would be sent to Groq."""
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent["headers"], sent["payload"] = headers, json
        return _OkResponse()

    monkeypatch.setattr(explainer.httpx, "post", fake_post)
    return sent


def test_key_and_model_are_both_read_from_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('GROQ_API_KEY="gsk_test"\nGROQ_MODEL=some/other-model\n')
    monkeypatch.setattr(explainer, "ENV_FILE", env)
    sent = _capture_post(monkeypatch)
    explainer.groq_complete([{"role": "user", "content": "hi"}])
    assert sent["headers"]["Authorization"] == "Bearer gsk_test"
    assert sent["payload"]["model"] == "some/other-model"


def test_environment_variable_beats_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=from_file\nGROQ_MODEL=file-model\n")
    monkeypatch.setattr(explainer, "ENV_FILE", env)
    monkeypatch.setenv("GROQ_API_KEY", "from_env")
    monkeypatch.setenv("GROQ_MODEL", "env-model")
    sent = _capture_post(monkeypatch)
    explainer.groq_complete([{"role": "user", "content": "hi"}])
    assert sent["headers"]["Authorization"] == "Bearer from_env"
    assert sent["payload"]["model"] == "env-model"


def test_default_model_and_request_shape(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    sent = _capture_post(monkeypatch)
    explainer.groq_complete([{"role": "user", "content": "hi"}])
    p = sent["payload"]
    assert p["model"] == explainer.DEFAULT_MODEL == "openai/gpt-oss-20b"
    assert p["response_format"] == {"type": "json_object"}
    assert p["max_completion_tokens"] == explainer.MAX_OUTPUT_TOKENS
    assert p["reasoning_effort"] == "low"  # only sent for gpt-oss models


def test_reasoning_effort_not_sent_to_other_models(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    sent = _capture_post(monkeypatch)
    explainer.groq_complete([{"role": "user", "content": "hi"}])
    assert "reasoning_effort" not in sent["payload"]