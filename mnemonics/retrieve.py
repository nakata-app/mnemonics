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
) -> dict[str, Any]:
    """Search the store for query. Tier-aware decay + reinforcement applied unless decay=False."""
    enc = _get_encoder(model)
    qvec = enc.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
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

    if decay:
        results.sort(key=lambda r: r["score"], reverse=True)

    return {"results": results}
