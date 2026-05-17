"""Tests for hybrid search: FTS5-backed BM25 + RRF fusion.

Covers Store-level BM25 path and the in-memory eval-side fusion utilities.
Full end-to-end retrieve() with sentence-transformers is exercised by the
real eval harness on the 30-query set, not here.
"""
from __future__ import annotations

import pytest

from mnemonics.eval import _build_bm25_index, _bm25_rank, rrf_fuse_ids
from mnemonics.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(path=tmp_path)
    return s


def _seed_store(s: Store) -> None:
    """Insert a small corpus with bypass of vector embedding."""
    import numpy as np
    texts = [
        "Zeus Faz 3 v0.5 drift test, planner regression bug fix",
        "Sienna packshot pipeline 1353 manifest hazır R2 upload",
        "Pistachio proxy URL unwrap fix",
        "AdaptMem encoder fine-tune Colab T4 GPU training",
        "Aegis CLI MCP boot wiring mouse capture config follow-up",
    ]
    # Cheap fake 384-dim vectors (we only test BM25 path here).
    vecs = np.random.RandomState(0).rand(len(texts), 384).astype("float32")
    vecs /= (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
    s.add(texts=texts, vectors=vecs, ns="testns")


def test_fts_sanitize_keeps_alphanumeric_tokens():
    out = Store._fts_sanitize("PR #1490 (granularity) fix!")
    assert '"PR"' in out and '"1490"' in out and '"granularity"' in out and '"fix"' in out


def test_fts_sanitize_empty_punctuation_only():
    assert Store._fts_sanitize("!?.,") == ""


def test_search_bm25_keyword_hit(store):
    _seed_store(store)
    out = store.search_bm25("AdaptMem encoder", ns="testns", top_k=3)
    assert len(out) >= 1
    assert "AdaptMem" in out[0]["text"]


def test_search_bm25_returns_empty_for_no_terms(store):
    _seed_store(store)
    assert store.search_bm25("!!!", ns="testns") == []


def test_search_bm25_respects_namespace(store):
    _seed_store(store)
    # Different ns → no hits even though tokens are present in another ns.
    assert store.search_bm25("AdaptMem", ns="otherns") == []


def test_search_bm25_score_higher_is_better(store):
    _seed_store(store)
    out = store.search_bm25("Sienna packshot", ns="testns", top_k=5)
    # All scores should be negative-of-bm25 → first result has the highest
    # (least-negative) score among returned rows.
    scores = [r["score"] for r in out]
    assert scores == sorted(scores, reverse=True)


def test_rrf_fuse_ids_prefers_doc_ranked_well_in_both_lists():
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    fused = rrf_fuse_ids([a, b], top_k=3)
    # y is rank 2 + rank 1 = 1/62 + 1/61 ≈ 0.0327
    # x is rank 1 + rank 2 = 1/61 + 1/62 ≈ 0.0327 (same to ~1e-5)
    # tie-broken by sort stability, both should come above z and w.
    assert set(fused[:2]) == {"x", "y"}


def test_rrf_fuse_ids_single_list_falls_back_to_ranks():
    fused = rrf_fuse_ids([["a", "b", "c"]], top_k=3)
    assert fused == ["a", "b", "c"]


def test_rrf_fuse_ids_top_k_truncates():
    fused = rrf_fuse_ids([["a", "b", "c", "d", "e"]], top_k=2)
    assert fused == ["a", "b"]


def test_eval_bm25_index_round_trip():
    texts = [
        "alpha beta gamma",
        "alpha delta",
        "epsilon zeta",
    ]
    chunk_ids = ["doc-0", "doc-1", "doc-2"]
    conn = _build_bm25_index(texts)
    out = _bm25_rank(conn, "alpha", chunk_ids, top_k=10)
    # Both alpha-bearing docs should come back, epsilon should not.
    assert set(out) == {"doc-0", "doc-1"}
    assert "doc-2" not in out


def test_eval_bm25_index_handles_punctuation_only_query():
    conn = _build_bm25_index(["foo bar"])
    assert _bm25_rank(conn, "!!", ["doc-0"], top_k=5) == []
