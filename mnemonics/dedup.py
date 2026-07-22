"""Suggest-dedup: surface high-similarity neighbours at ingest time, never merge silently.

The contract is intentionally observational. find_similar() answers the question
"is anything in the store within `threshold` cosine of this new text?" and
returns the matches. The actual decision (merge / save-new / cancel) lives at
the call site — the CLI, an MCP client, or a script. This keeps the library
honest: no row is ever rewritten or dropped without the caller asking for it.
"""
from __future__ import annotations

from typing import Any

from mnemonics.ingest import _get_encoder, ingest
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


# Cosine at/above which two texts are treated as the SAME memory restated, so
# the incoming copy is dropped (NOOP) rather than stored as a near-duplicate.
# Deliberately higher than DEFAULT_THRESHOLD (0.92): 0.92 is "worth a look",
# 0.98 is "this is the same sentence in different words". Below it we never
# auto-skip: a false NOOP silently loses a memory, the one failure this module
# exists to avoid.
NOOP_THRESHOLD = 0.98


def reconcile_ingest(
    texts: list[str],
    store: Store,
    ns: str = "default",
    *,
    noop_threshold: float = NOOP_THRESHOLD,
    supersede_map: dict[int, list[int]] | None = None,
    tier: int = 1,
    model: str = "all-MiniLM-L6-v2",
    **ingest_kwargs: Any,
) -> dict[str, Any]:
    """Conflict-aware ingest: the mnemonics answer to Mem0's ADD/UPDATE/DELETE/NOOP.

    This is the *only* path that skips or archives on its own, and it does so
    conservatively, because the judgment Mem0 delegates to an LLM lives at the
    call site here (the agent invoking this), not in an embedding heuristic:

    - NOOP: if a text is ≥ ``noop_threshold`` cosine to an existing memory in
      ``ns``, it is a restatement; skip it (do not store a duplicate).
    - SUPERSEDE (explicit UPDATE/DELETE): ``supersede_map`` maps a text's index
      to the ids of existing memories it replaces. Those texts are always
      ingested (NOOP is bypassed for them: the caller asserted this is new,
      corrected information), and each named old id is archived via
      ``store.supersede`` (kept for audit, hidden from retrieval). Never a hard
      delete.
    - ADD: everything else is ingested normally.

    Returns::

        {"added": [ids...],
         "noop_skipped": [{"index", "text", "matched_id", "similarity"}...],
         "superseded": [{"old_id", "new_id"}...],
         "supersede_failed": [old_id...]}   # ids not found in the store
    """
    supersede_map = supersede_map or {}
    added: list[int] = []
    noop_skipped: list[dict[str, Any]] = []
    superseded: list[dict[str, Any]] = []
    supersede_failed: list[int] = []

    for i, text in enumerate(texts):
        old_ids = supersede_map.get(i)
        if not old_ids:
            # Only auto-NOOP when the caller did NOT flag this as a correction.
            top = find_similar(text, store=store, ns=ns,
                               threshold=noop_threshold, top_k=1, model=model)
            if top:
                noop_skipped.append({
                    "index": i, "text": text,
                    "matched_id": top[0]["id"], "similarity": top[0]["similarity"],
                })
                continue

        new_ids = ingest([text], store, ns=ns, tier=tier, model=model,
                         return_ids=True, **ingest_kwargs)
        if not new_ids:
            continue
        added.extend(new_ids)
        new_primary = new_ids[0]
        for old_id in (old_ids or []):
            if store.supersede(old_id, new_primary):
                superseded.append({"old_id": old_id, "new_id": new_primary})
            else:
                supersede_failed.append(old_id)

    return {
        "added": added,
        "noop_skipped": noop_skipped,
        "superseded": superseded,
        "supersede_failed": supersede_failed,
    }
