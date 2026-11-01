"""LLM explanations (the generation half of RAG), served by Groq's OpenAI-compatible API.

One batched call per search: the model gets the query plus, for every returned resource,
its trust summary and the numbered evidence reviews retrieved for it, and must write a
short explanation that cites reviews as [1], [2], ...

Every explanation is validated before it is used. If the model cites a review that does
not exist, cites nothing, returns bad JSON, or the API is unavailable (no key, rate
limit, network), that item falls back to the deterministic template. So the app always
shows something, and `explanation_source` says which kind it is.

Config (environment variables, or a `.env` file in the repo root):
  GROQ_API_KEY   required for LLM explanations; without it everything uses the template
  GROQ_MODEL     default "llama-3.1-8b-instant" (largest free-tier daily quota)
"""
import json
import os
import re
from pathlib import Path

import httpx

API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"
TIMEOUT_S = 20.0
MAX_EXPLANATION_CHARS = 420
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

SYSTEM_PROMPT = """You write short explanations for an educational resource recommender.
For each resource you receive, explain in 1-2 sentences why it suits the learner's query.

Rules:
- Use ONLY the numbered reviews provided for that resource. Do not add facts from your own knowledge.
- Cite the reviews you rely on with their numbers in square brackets, like [1] or [2][3].
- If a review raises a weakness (too fast, outdated, needs prior knowledge), mention it honestly.
- Do not invent numbers. You may mention the trust score and verified-review share given to you.
- Plain text only, no markdown, at most 2 sentences.

Reply with a single JSON object mapping each resource id (as a string) to its explanation."""

_cache: dict = {}


def _api_key() -> str | None:
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key.strip()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            name, _, val = line.partition("=")
            if name.strip() == "GROQ_API_KEY" and val.strip():
                return val.strip().strip('"').strip("'")
    return None


def groq_complete(messages: list) -> str | None:
    """Call Groq; return the reply text, or None on any failure (never raises)."""
    key = _api_key()
    if not key:
        return None
    try:
        resp = httpx.post(
            API_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 700,
                "response_format": {"type": "json_object"},
            },
            timeout=TIMEOUT_S,
        )
        if resp.status_code != 200:  # 429 rate limit, 401 bad key, 5xx ...
            return None
        return resp.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return None


def template_explanation(item: dict) -> str:
    """Deterministic fallback, built only from the trust components."""
    t = item["trust"]
    c = t["components"]
    parts = [f'Matches your query (relevance {item["relevance"]:.2f}).']
    parts.append(
        f'Trust {t["score"]:.2f}: {c["verified"]*100:.0f}% verified-review weight, '
        f'rating component {c["rating"]:.2f}, from {t["n_reviews"]} reviews.'
    )
    if t["n_flagged"]:
        parts.append(f'{t["n_flagged"]} suspicious review(s) were down-weighted.')
    return " ".join(parts)


def build_messages(query: str, items: list) -> list:
    blocks = []
    for it in items:
        t = it["trust"]
        lines = [
            f'Resource id {it["id"]}: "{it["title"]}" ({it["type"]}, {it["provider"]})',
            f'Trust score {t["score"]:.2f}; verified-review share {t["components"]["verified"]*100:.0f}%; '
            f'{t["n_reviews"]} reviews, {t["n_flagged"]} flagged as suspicious and excluded below.',
            "Reviews:",
        ]
        for n, ev in enumerate(it["evidence"], start=1):
            tag = "verified" if ev["verified"] else "unverified"
            lines.append(f'[{n}] ({ev["rating"]}/5, {tag}, {ev["helpful_votes"]} helpful) {ev["text"]}')
        blocks.append("\n".join(lines))
    user = f'Learner query: "{query}"\n\n' + "\n\n".join(blocks)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def validate(text, n_evidence: int) -> bool:
    """Accept only a short string whose citations all point at real evidence."""
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_EXPLANATION_CHARS:
        return False
    cited = [int(x) for x in re.findall(r"\[(\d+)\]", text)]
    return bool(cited) and all(1 <= c <= n_evidence for c in cited)


def explain_batch(query: str, items: list, complete=groq_complete) -> dict:
    """Return {resource id: {"text": str, "source": "llm" | "template"}} for every item."""
    out = {it["id"]: {"text": template_explanation(it), "source": "template"} for it in items}
    askable = [it for it in items if it["evidence"]]
    if not askable:
        return out

    cache_key = (query.strip().lower(), tuple(it["id"] for it in askable))
    if cache_key in _cache:
        raw = _cache[cache_key]
    else:
        raw = complete(build_messages(query, askable))
        if raw is not None:
            _cache[cache_key] = raw  # only successful replies are cached
    if raw is None:
        return out

    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return out
    if not isinstance(parsed, dict):
        return out

    for it in askable:
        text = parsed.get(str(it["id"]))
        if validate(text, len(it["evidence"])):
            out[it["id"]] = {"text": text.strip(), "source": "llm"}
    return out