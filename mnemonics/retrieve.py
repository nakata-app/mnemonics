"""Retrieve memories — embed query, search, apply tier-aware decay + reinforcement scoring."""
from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from typing import Any

from mnemonics.store import Store
from mnemonics.ingest import _get_encoder

# Question-signal extractors. Lifted from longmemeval analysis: quoted phrases
# and proper-noun person names that the bi-encoder under-weights are reliably
# recoverable via exact-match boost.
_QUOTED_RE = (re.compile(r"'([^']{3,60})'"), re.compile(r'"([^"]{3,60})"'))
_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,15}\b")
_NOT_NAMES = frozenset({
    "What","When","Where","Who","How","Which","Did","Do","Was","Were","Have","Has",
    "Had","Is","Are","The","My","Our","Their","Can","Could","Would","Should","Will",
    "Shall","May","Might","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday",
    "Sunday","January","February","March","April","May","June","July","August",
    "September","October","November","December","In","On","At","For","To","Of","With",
    "By","From","And","But","I","It","Its","This","That","These","Those","Previously",
    "Recently","Also","Just","Very","More",
})


def _extract_quoted_phrases(query: str) -> list[str]:
    phrases: list[str] = []
    for pat in _QUOTED_RE:
        phrases.extend(pat.findall(query))
    return [p.strip().lower() for p in phrases if len(p.strip()) >= 3]


def _extract_person_names(query: str) -> list[str]:
    return [w.lower() for w in set(_NAME_RE.findall(query)) if w not in _NOT_NAMES]


def _signal_boost(text: str, quoted: list[str], names: list[str]) -> float:
    """Multiplicative boost for rows whose text contains question-side signals.
    Returns 1.0 (no-op) when no signals are present or matched.
    Strong (1.6x) on exact quoted-phrase match; moderate (1.25x) on name match.
    """
    if not quoted and not names:
        return 1.0
    t = text.lower()
    q_hit = sum(1 for p in quoted if p in t) / max(len(quoted), 1) if quoted else 0.0
    n_hit = sum(1 for n in names if n in t) / max(len(names), 1) if names else 0.0
    return 1.0 + 0.60 * q_hit + 0.25 * n_hit

# Lazy-cached AdaptMem instance for CE rerank. We hold only the CrossEncoder
# state; no bi-encoder index is needed (mnemonics owns its own HNSW + BM25).
_rerank_am: Any = None
_rerank_model_name: str | None = None


def _get_rerank_am(model: str | None = None) -> Any:
    """Return a cached AdaptMem instance whose CE is ready to score pairs."""
    global _rerank_am, _rerank_model_name
    try:
        from adaptmem import AdaptMem
    except ImportError as e:
        raise RuntimeError(
            "rerank=True requires the 'adaptmem' package. Install with "
            "`pip install adaptmem` or set rerank=False."
        ) from e
    name = model or os.environ.get(
        "MNEMONICS_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2"
    )
    if _rerank_am is None or _rerank_model_name != name:
        _rerank_am = AdaptMem(rerank_model=name)
        _rerank_model_name = name
    return _rerank_am


def _ce_rerank(query: str, results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Cross-encoder rerank: attach ce_score, replace score, return top_k sorted desc."""
    if not results:
        return []
    am = _get_rerank_am()
    ranked = am.rerank(query, [r["text"] for r in results])
    out: list[dict[str, Any]] = []
    for idx, ce_score in ranked:
        item = dict(results[idx])
        item["ce_score"] = round(float(ce_score), 4)
        item["score"] = item["ce_score"]
        out.append(item)
        if len(out) >= top_k:
            break
    return out


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
    candidate_k: int = 50,
    rerank: bool = False,
    boost_signals: bool = True,
    use_doc_filter: bool = False,
) -> dict[str, Any]:
    """Search the store for query. Tier-aware decay + reinforcement applied unless decay=False.

    When `hybrid=True`, the vector top-`candidate_k` and the BM25 (SQLite FTS5)
    top-`candidate_k` are fused with Reciprocal Rank Fusion before the
    decay/boost pass runs on the truncated top-`top_k`.

    When `rerank=True`, the fusion stage keeps the full `candidate_k` band
    (instead of truncating to `top_k`), decay/boost still annotate each row,
    then the AdaptMem cross-encoder rescores (query, text) pairs and the
    final list is truncated to `top_k` sorted by ce_score desc. Requires the
    `adaptmem` package; env `MNEMONICS_RERANK_MODEL` overrides the CE model.

    When `boost_signals=True` (default), exact-match boost is applied for
    quoted phrases ('X' / "X") and proper-noun person names (capitalized
    mid-sentence) extracted from the query. No-op when no signals are found
    or no candidate text matches; never penalizes.
    """
    enc = _get_encoder(model)
    qvec = enc.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    fusion_top = candidate_k if rerank else top_k

    # Stage 1: session-level filter. Pull the top candidate_k *sessions* by
    # doc embedding, then keep only chunks whose meta.source_idx belongs to
    # one of those sessions. Falls back to no-op when the doc index is empty
    # (legacy ns ingested before doc support).
    candidate_source_idxs: set[int] | None = None
    if use_doc_filter:
        doc_hits = store.search_docs(qvec, ns=ns, top_k=candidate_k)
        if doc_hits:
            candidate_source_idxs = set(doc_hits)

    def _filter_by_session(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if candidate_source_idxs is None:
            return items
        return [it for it in items if it["meta"].get("source_idx") in candidate_source_idxs]

    if hybrid:
        vec_results = _filter_by_session(store.search(qvec, ns=ns, top_k=candidate_k))
        bm25_results = _filter_by_session(store.search_bm25(query, ns=ns, top_k=candidate_k))
        results = _rrf_fuse([vec_results, bm25_results], top_k=fusion_top)
    else:
        results = _filter_by_session(store.search(qvec, ns=ns, top_k=fusion_top))

    quoted = _extract_quoted_phrases(query) if boost_signals else []
    names = _extract_person_names(query) if boost_signals else []

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
        if boost_signals and (quoted or names):
            sig = _signal_boost(r["text"], quoted, names)
            r["signal_boost"] = round(sig, 4)
            r["score"] = round(r["score"] * sig, 4)
        else:
            r["signal_boost"] = 1.0

    if rerank:
        results = _ce_rerank(query, results, top_k=top_k)
    else:
        if decay:
            results.sort(key=lambda r: r["score"], reverse=True)

    return {"results": results}
