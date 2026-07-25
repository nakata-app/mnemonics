"""Retrieve memories — embed query, search, apply tier-aware decay + reinforcement scoring."""
from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from typing import Any

from mnemonics.ingest import _get_encoder, _resolve_model
from mnemonics.store import Store

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

# Lazy-cached CrossEncoder for rerank. Loaded once per process and reused.
# Tries AdaptMem.rerank first (Atakan's local repo has it); falls back to a
# bare sentence_transformers.CrossEncoder. The fallback keeps Kaggle/Colab
# runs working where the PyPI adaptmem release lacks the rerank method.
_rerank_ce: Any = None
_rerank_model_name: str | None = None


def _get_rerank_ce(model: str | None = None) -> Any:
    """Return a cached CrossEncoder instance (via AdaptMem if available, else bare ST)."""
    global _rerank_ce, _rerank_model_name
    name = model or os.environ.get(
        "MNEMONICS_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"
    )
    if _rerank_ce is not None and _rerank_model_name == name:
        return _rerank_ce
    # Prefer AdaptMem if it exposes rerank (lets future AdaptMem versions
    # inject FT'd CE heads). Otherwise use sentence-transformers directly.
    try:
        from adaptmem import AdaptMem
        am = AdaptMem(rerank_model=name)
        if hasattr(am, "rerank"):
            _rerank_ce = am
            _rerank_model_name = name
            return _rerank_ce
    except Exception:
        pass
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise RuntimeError(
            "rerank=True requires either 'adaptmem' (with rerank support) or "
            "'sentence-transformers'. Install with `pip install sentence-transformers`."
        ) from e
    # max_length: bge-reranker-v2-m3 advertises 8192, so a band of long rows
    # becomes a 50 x 8192 attention batch and OOMs a 16GB GPU (measured on a
    # T4: 5.68 GiB single allocation). Left unset the model config wins, which
    # keeps existing results byte-identical; MNEMONICS_RERANK_MAX_LENGTH caps
    # it for corpora with long rows.
    max_len = os.environ.get("MNEMONICS_RERANK_MAX_LENGTH")
    kwargs: dict[str, Any] = {}
    if max_len:
        kwargs["max_length"] = int(max_len)
    _rerank_ce = CrossEncoder(name, **kwargs)
    _rerank_model_name = name
    return _rerank_ce


def _ce_rerank(query: str, results: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Cross-encoder rerank: attach ce_score, replace score, return top_k sorted desc."""
    if not results:
        return []
    ce = _get_rerank_ce()
    texts = [r["text"] for r in results]
    if hasattr(ce, "rerank"):
        # AdaptMem-style: returns [(idx, score), ...] already sorted desc
        ranked = ce.rerank(query, texts)
    else:
        # Bare CrossEncoder: score pairs, sort ourselves
        pairs = [(query, t) for t in texts]
        # Default batch_size is 32; one batch of long rows is what blows up.
        # Unset means unchanged behaviour, so champion numbers stay reproducible.
        bs = os.environ.get("MNEMONICS_RERANK_BATCH_SIZE")
        predict_kwargs: dict[str, Any] = {"show_progress_bar": False}
        if bs:
            predict_kwargs["batch_size"] = int(bs)
        scores = ce.predict(pairs, **predict_kwargs)
        ranked = sorted(enumerate(scores), key=lambda x: -float(x[1]))
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
    min_tier: int | None = None,
    max_tier: int | None = None,
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
    # Encoder drift kontrolu: stored vektorler farkli encoder ile gomulduyse
    # sessiz kalite kaybini gorunur uyariya cevir. Bir kez (cached), hard fail
    # YOK (canli retrieval'i kirmaz). Eski (damgasiz) DB'leri ilk kullanimda damgalar.
    try:
        from mnemonics import embed_manifest as _em
        _resolved = _resolve_model(model)
        _key = (str(store.root), _resolved, store.dim)
        _seen = retrieve.__dict__.setdefault("_enc_checked", set())
        if _key not in _seen:
            _seen.add(_key)
            _live = _em.encoder_fingerprint(_resolved, store.dim)
            _st, _why = _em.verify(store.root, _live)
            if _st == "missing":
                _em.write(store.root, _live)
            elif _st == "mismatch":
                import logging
                logging.getLogger("mnemonics").warning(
                    "ENCODER DRIFT: %s. Re-embed gerekli (store.reindex_all()).", _why)
    except Exception:
        pass
    fusion_top = candidate_k if rerank else top_k
    if hybrid:
        vec_results = store.search(qvec, ns=ns, top_k=candidate_k, min_tier=min_tier, max_tier=max_tier)
        bm25_results = store.search_bm25(query, ns=ns, top_k=candidate_k, min_tier=min_tier, max_tier=max_tier)
        results = _rrf_fuse([vec_results, bm25_results], top_k=fusion_top)
    else:
        results = store.search(qvec, ns=ns, top_k=fusion_top, min_tier=min_tier, max_tier=max_tier)

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
