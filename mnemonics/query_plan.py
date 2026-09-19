"""Deterministic multi-query planning and fusion for agent-memory retrieval.

This module deliberately uses no LLM. It expands a user turn into a small,
bounded set of complementary queries, retrieves each independently, then fuses
rankings with weighted RRF. The original query always has the highest weight.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from mnemonics.ingest import _get_encoder, _resolve_model_for_store
from mnemonics.retrieve import _ce_rerank, project_scope_factor, retrieve
from mnemonics.store import Store

_WORD_RE = re.compile(r"[A-Za-z0-9_./:@+-]+")
_PATH_RE = re.compile(r"(?:^|\s)([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)")
_SYMBOL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
_STOP = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "what", "when", "where",
    "which", "why", "how", "can", "could", "would", "should", "please", "about",
    "into", "then", "than", "have", "has", "had", "was", "were", "are", "is",
    "bir", "ve", "bu", "şu", "icin", "için", "neden", "nasıl", "nasil", "olan",
})


@dataclass(frozen=True)
class PlannedQuery:
    text: str
    weight: float
    kind: str


def _dedupe(items: list[PlannedQuery], max_queries: int) -> list[PlannedQuery]:
    out: list[PlannedQuery] = []
    seen: set[str] = set()
    for item in items:
        key = " ".join(item.text.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max_queries:
            break
    return out


def plan_queries(query: str, max_queries: int = 4) -> list[PlannedQuery]:
    """Expand one turn into a bounded set of high-signal retrieval queries."""
    query = " ".join(query.split()).strip()
    if not query or max_queries <= 0:
        return []

    planned = [PlannedQuery(query, 1.0, "original")]

    paths = [m.group(1) for m in _PATH_RE.finditer(query)]
    if paths:
        planned.append(PlannedQuery(" ".join(paths[:6]), 0.95, "paths"))

    symbols = [
        s for s in _SYMBOL_RE.findall(query)
        if s.lower() not in _STOP and (("_" in s) or any(c.isupper() for c in s[1:]))
    ]
    if symbols:
        planned.append(PlannedQuery(" ".join(dict.fromkeys(symbols[:10])), 0.9, "symbols"))

    terms = [
        w for w in _WORD_RE.findall(query)
        if len(w) >= 4 and w.lower() not in _STOP
    ]
    if terms:
        compact = " ".join(dict.fromkeys(terms[:12]))
        planned.append(PlannedQuery(compact, 0.8, "keywords"))

    return _dedupe(planned, max_queries)

def _weighted_rrf(
    ranked_lists: list[tuple[PlannedQuery, list[dict[str, Any]]]],
    *,
    top_k: int,
    rrf_k: float = 60.0,
) -> list[dict[str, Any]]:
    """Fuse ranked result lists by row id using per-query weighted RRF."""
    scores: dict[int, float] = {}
    best: dict[int, dict[str, Any]] = {}
    matched: dict[int, list[str]] = {}

    for plan, rows in ranked_lists:
        for rank, row in enumerate(rows, start=1):
            row_id = int(row["id"])
            scores[row_id] = scores.get(row_id, 0.0) + plan.weight / (rrf_k + rank)
            matched.setdefault(row_id, []).append(plan.kind)
            prev = best.get(row_id)
            if prev is None or float(row.get("score", 0.0)) > float(prev.get("score", 0.0)):
                best[row_id] = dict(row)

    ordered = sorted(scores, key=lambda row_id: (-scores[row_id], row_id))
    out: list[dict[str, Any]] = []
    for row_id in ordered[:top_k]:
        item = dict(best[row_id])
        item["plan_score"] = round(scores[row_id], 8)
        item["matched_queries"] = matched[row_id]
        out.append(item)
    return out


# Compatibility alias for tests/callers that imported the early private helper.
_scope_factor = project_scope_factor


def retrieve_planned(
    query: str,
    store: Store,
    *,
    ns: str = "default",
    top_k: int = 5,
    candidate_k: int = 50,
    max_queries: int = 4,
    decay: bool = True,
    hybrid: bool = True,
    rerank: bool = False,
    min_tier: int | None = None,
    max_tier: int | None = None,
    project_hints: list[str] | None = None,
) -> dict[str, Any]:
    """Retrieve with deterministic query expansion and weighted rank fusion.

    Subqueries intentionally do not cross-encode independently. When rerank is
    requested, one cross-encoder pass runs only after fusion, against the
    original user query. This avoids N expensive reranker passes.
    """
    plans = plan_queries(query, max_queries=max_queries)
    if not plans:
        return {"results": [], "queries": []}

    per_query_k = max(top_k * 3, min(candidate_k, 20))
    resolved_model = _resolve_model_for_store("all-MiniLM-L6-v2", store)
    encoder = _get_encoder(resolved_model)
    vectors = encoder.encode(
        [plan.text for plan in plans],
        batch_size=max(1, len(plans)),
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    ranked: list[tuple[PlannedQuery, list[dict[str, Any]]]] = []
    for plan, query_vector in zip(plans, vectors):
        result = retrieve(
            query=plan.text,
            store=store,
            ns=ns,
            top_k=per_query_k,
            model=resolved_model,
            decay=decay,
            hybrid=hybrid,
            candidate_k=candidate_k,
            rerank=False,
            min_tier=min_tier,
            max_tier=max_tier,
            query_vector=query_vector,
            touch=False,
        )
        ranked.append((plan, result["results"]))

    fused = _weighted_rrf(ranked, top_k=max(top_k * 3, top_k))
    if project_hints:
        for row in fused:
            factor = _scope_factor(row, project_hints)
            row["scope_boost"] = factor
            row["scope_score"] = round(float(row["plan_score"]) * factor, 8)
        fused.sort(
            key=lambda row: (
                -float(row.get("scope_score", row["plan_score"])),
                int(row["id"]),
            )
        )
    if rerank:
        fused = _ce_rerank(query, fused, top_k=top_k)
    else:
        fused = fused[:top_k]
    store.touch_ids([int(row["id"]) for row in fused])

    return {
        "results": fused,
        "queries": [
            {"text": plan.text, "weight": plan.weight, "kind": plan.kind}
            for plan in plans
        ],
    }