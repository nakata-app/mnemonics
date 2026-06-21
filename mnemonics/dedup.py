"""Suggest-dedup: surface high-similarity neighbours at ingest time, never merge silently.

The contract is intentionally observational. find_similar() answers the question
"is anything in the store within `threshold` cosine of this new text?" and
returns the matches. The actual decision (merge / save-new / cancel) lives at
the call site — the CLI, an MCP client, or a script. This keeps the library
honest: no row is ever rewritten or dropped without the caller asking for it.
"""
from __future__ import annotations

from typing import Any

from mnemonics.ingest import _get_encoder
from mnemonics.store import Store

# Cosine threshold above which two texts are "near-duplicates" worth surfacing.
# 0.92 picked from observation: same-topic / different-wording memories cluster
# around 0.85-0.91; true paraphrases or restatements typically sit at 0.93+.
DEFAULT_THRESHOLD = 0.92


def find_similar(
    text: str,
    store: Store,
    ns: str = "default",
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = 3,
    model: str = "all-MiniLM-L6-v2",
) -> list[dict[str, Any]]:
    """Return existing memories in `ns` with cosine similarity ≥ threshold.

    Results are sorted best-first and capped at `top_k`. Empty list means no
    near-duplicate was found and the caller can ingest without conflict.
    """
    enc = _get_encoder(model)
    qvec = enc.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
    # Pull a larger pool than top_k so the threshold filter still gives top_k
    # genuine matches when the namespace has many near-but-not-quite hits.
    pool = max(top_k * 5, 10)
    candidates = store.search(qvec, ns=ns, top_k=pool)
    matches: list[dict[str, Any]] = []
    for c in candidates:
        sim = float(c["score"])
        if sim >= threshold:
            matches.append({
                "id": c["id"],
                "text": c["text"],
                "similarity": round(sim, 4),
            })
            if len(matches) >= top_k:
                break
    return matches


def check_before_ingest(
    texts: list[str],
    store: Store,
    ns: str = "default",
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Per-text similarity check. Returns one entry per input, with `suggestions`.

    Result shape:
        [{"text": <input>, "suggestions": [{"id", "text", "similarity"}, ...]}, ...]

    An empty `suggestions` list means that text is safe to ingest as-is.
    """
    out: list[dict[str, Any]] = []
    for t in texts:
        matches = find_similar(t, store=store, ns=ns, threshold=threshold, top_k=top_k)
        out.append({"text": t, "suggestions": matches})
    return out
