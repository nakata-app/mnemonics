"""Retrieve memories — embed query, search, apply tier-aware decay + reinforcement scoring."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from mnemonics.store import Store
from mnemonics.ingest import _get_encoder


# Tier-based half-life (days). Tier 0 (pinned) skips decay entirely.
_HALF_LIFE_DAYS = {1: 90.0, 2: 14.0}

# Reinforcement boost cap. A memory accessed many times still tops out at 2x.
_BOOST_CAP = 2.0
_BOOST_RATE = 0.1

# RRF damping constant. 60 is the value used in the original Cormack/Clarke/Buettcher
# paper; small enough that rank-1 dominates, large enough to keep tail signal.
_RRF_K = 60.0


def _rrf_fuse(
    rankings: list[list[dict[str, Any]]],
    top_k: int,
    rrf_k: float = _RRF_K,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion across multiple ranked lists keyed by row id.

    Each list must already be sorted best-first. RRF score for a doc is
    sum over lists of 1 / (rrf_k + rank_in_list). Vector-side scores are
    discarded in favor of the fused rank; downstream decay/boost still
    multiply against the raw vector score, so we preserve the *first*
    occurrence of each row.
    """
    by_id: dict[int, dict[str, Any]] = {}
    rrf_score: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            row_id = item["id"]
            rrf_score[row_id] = rrf_score.get(row_id, 0.0) + 1.0 / (rrf_k + rank)
            if row_id not in by_id:
                by_id[row_id] = item
    fused = []
    for row_id, score in sorted(rrf_score.items(), key=lambda kv: kv[1], reverse=True):
        item = dict(by_id[row_id])
        item["rrf_score"] = score
        # Replace source-specific score (cosine OR -bm25) with the fused rank
        # score so the downstream decay/boost loop has a unified base.
        item["score"] = score
        fused.append(item)
    return fused[:top_k]


def _age_days(created_str: str) -> float:
    """`created` is SQLite datetime('now') format: 'YYYY-MM-DD HH:MM:SS' (UTC)."""
    try:
        dt = datetime.strptime(created_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return 0.0
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0


def _decay_factor(tier: int, age_days: float) -> float:
    if tier == 0:
        return 1.0
    half_life = _HALF_LIFE_DAYS.get(tier, 90.0)
    return math.exp(-math.log(2) * age_days / half_life)


def _reinforcement_boost(access_count: int) -> float:
    """Spaced-repetition style boost. log(1+n) so frequent rows climb but never run away."""
    if not access_count or access_count < 0:
        return 1.0
    return min(1.0 + math.log(1 + access_count) * _BOOST_RATE, _BOOST_CAP)


def retrieve(
    query: str,
    store: Store,
    ns: str = "default",
    top_k: int = 5,
    model: str = "all-MiniLM-L6-v2",
    decay: bool = True,
    hybrid: bool = True,
    candidate_k: int = 20,
) -> dict[str, Any]:
    """Search the store for query. Tier-aware decay + reinforcement applied unless decay=False.

    When `hybrid=True`, the vector top-`candidate_k` and the BM25 (SQLite FTS5)
    top-`candidate_k` are fused with Reciprocal Rank Fusion before the
    decay/boost pass runs on the truncated top-`top_k`.
    """
    enc = _get_encoder(model)
    qvec = enc.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    if hybrid:
        vec_results = store.search(qvec, ns=ns, top_k=candidate_k)
        bm25_results = store.search_bm25(query, ns=ns, top_k=candidate_k)
        results = _rrf_fuse([vec_results, bm25_results], top_k=top_k)
    else:
        results = store.search(qvec, ns=ns, top_k=top_k)

    for r in results:
        r["raw_score"] = round(r["score"], 4)
        age = _age_days(r["created"])
        r["age_days"] = round(age, 1)
        if decay:
            df = _decay_factor(r["tier"], age)
            boost = _reinforcement_boost(r.get("access_count", 0) or 0)
            r["decay_factor"] = round(df, 4)
            r["boost"] = round(boost, 4)
            r["score"] = round(r["raw_score"] * df * boost, 4)
        else:
            r["decay_factor"] = 1.0
            r["boost"] = 1.0
            r["score"] = r["raw_score"]

    if decay:
        results.sort(key=lambda r: r["score"], reverse=True)

    return {"results": results}
