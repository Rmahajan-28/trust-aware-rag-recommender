const form = document.getElementById("search-form");
const qEl = document.getElementById("q");
const kindEl = document.getElementById("kind");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

function trustColor(s) {
  return s >= 0.75 ? "var(--good)" : s >= 0.55 ? "var(--mid)" : "var(--low)";
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text; // textContent: no HTML injection
  return e;
}

function card(r) {
  const c = el("article", "card");
  c.appendChild(el("h2", null, r.title));

  const meta = el("div", "meta");
  meta.appendChild(el("span", "badge", r.type));
  meta.appendChild(document.createTextNode(`${r.provider} · relevance ${r.relevance.toFixed(2)} · overall ${r.score.toFixed(2)}`));
  c.appendChild(meta);

  const row = el("div", "trust-row");
  row.appendChild(el("span", null, "Trust"));
  const bar = el("div", "bar");
  const fill = el("span");
  fill.style.width = `${Math.round(r.trust.score * 100)}%`;
  fill.style.background = trustColor(r.trust.score);
  bar.appendChild(fill);
  row.appendChild(bar);
  row.appendChild(el("span", "trust-val", r.trust.score.toFixed(2)));
  c.appendChild(row);

  const why = el("p", "why", r.explanation + " ");
  why.appendChild(el("span", "src", r.explanation_source === "llm" ? "AI-written from reviews" : "template"));
  c.appendChild(why);

  if (r.evidence && r.evidence.length) {
    const det = el("details", "evidence");
    det.appendChild(el("summary", null, `Evidence: ${r.evidence.length} reviews`));
    const ol = el("ol");
    r.evidence.forEach((ev) => {
      const tag = ev.verified ? "verified" : "unverified";
      ol.appendChild(el("li", null, `“${ev.text}” (${ev.rating}/5, ${tag}, ${ev.helpful_votes} helpful)`));
    });
    det.appendChild(ol);
    c.appendChild(det);
  }
  if (r.trust.n_flagged) {
    c.appendChild(el("p", "flag", `⚠ ${r.trust.n_flagged} of ${r.trust.n_reviews} reviews flagged as suspicious`));
  }
  return c;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = qEl.value.trim();
  if (!q) return;
  statusEl.textContent = "Searching…";
  resultsEl.replaceChildren();
  try {
    const params = new URLSearchParams({ q, k: 6 });
    if (kindEl.value) params.set("kind", kindEl.value);
    const res = await fetch(`/api/search?${params}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    statusEl.textContent = data.results.length
      ? `${data.results.length} result(s) for “${data.query}”`
      : `No matches for “${data.query}”. Try different keywords.`;
    data.results.forEach((r) => resultsEl.appendChild(card(r)));
  } catch (err) {
    statusEl.textContent = `Something went wrong: ${err.message}`;
  }
});