"""Property-based tests using Hypothesis.

Invariants that must hold regardless of input:
- add N items → count increases by exactly N
- delete(id) → id never appears in search results
- forget(ns) → count(ns) == 0
- search returns at most top_k results
- search scores are in [-0.01, 1.01]
- gc only removes tier-2 items
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mnemonics.store import DIM, Store


# ── helpers ──────────────────────────────────────────────────────────────────

def _rand_vecs(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random((n, DIM)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _fresh_store() -> tuple[Store, tempfile.TemporaryDirectory]:
    """Return a new Store in a throwaway temp dir. Caller owns the context."""
    d = tempfile.TemporaryDirectory()
    return Store(Path(d.name)), d


_SUPPRESS = [HealthCheck.too_slow]


# ── count invariants ─────────────────────────────────────────────────────────

@given(
    n=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=30, suppress_health_check=_SUPPRESS, deadline=None)
def test_add_increases_count_by_n(n, seed):
    s, d = _fresh_store()
    with d:
        vecs = _rand_vecs(n, seed)
        before = s.count()
        s.add([f"text-{i}" for i in range(n)], vecs)
        assert s.count() == before + n


@given(
    n=st.integers(min_value=2, max_value=15),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=20, suppress_health_check=_SUPPRESS, deadline=None)
def test_delete_reduces_count_by_one(n, seed):
    s, d = _fresh_store()
    with d:
        vecs = _rand_vecs(n, seed)
        ids = s.add([f"t{i}" for i in range(n)], vecs)
        before = s.count()
        deleted = s.delete(ids[0])
        assert deleted is True
        assert s.count() == before - 1


# ── search invariants ─────────────────────────────────────────────────────────

@given(
    n=st.integers(min_value=1, max_value=30),
    top_k=st.integers(min_value=1, max_value=10),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=25, suppress_health_check=_SUPPRESS, deadline=None)
def test_search_returns_at_most_top_k(n, top_k, seed):
    s, d = _fresh_store()
    with d:
        vecs = _rand_vecs(n, seed)
        s.add([f"d{i}" for i in range(n)], vecs)
        q = _rand_vecs(1, seed + 1)[0]
        results = s.search(q, top_k=top_k)
        assert len(results) <= top_k
        assert len(results) <= n


@given(
    n=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=25, suppress_health_check=_SUPPRESS, deadline=None)
def test_search_scores_in_valid_range(n, seed):
    s, d = _fresh_store()
    with d:
        vecs = _rand_vecs(n, seed)
        s.add([f"d{i}" for i in range(n)], vecs)
        q = _rand_vecs(1, seed + 7)[0]
        for r in s.search(q, top_k=n):
            assert -0.01 <= r["score"] <= 1.01, f"score out of range: {r['score']}"


@given(
    n=st.integers(min_value=2, max_value=20),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=20, suppress_health_check=_SUPPRESS, deadline=None)
def test_deleted_item_absent_from_search(n, seed):
    s, d = _fresh_store()
    with d:
        vecs = _rand_vecs(n, seed)
        ids = s.add([f"d{i}" for i in range(n)], vecs)
        target_id = ids[0]
        target_vec = vecs[0]
        s.delete(target_id)
        results = s.search(target_vec, top_k=n)
        assert all(r["id"] != target_id for r in results)


# ── namespace isolation ───────────────────────────────────────────────────────

@given(
    n1=st.integers(min_value=1, max_value=10),
    n2=st.integers(min_value=1, max_value=10),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=20, suppress_health_check=_SUPPRESS, deadline=None)
def test_namespace_count_isolation(n1, n2, seed):
    s, d = _fresh_store()
    with d:
        v1 = _rand_vecs(n1, seed)
        v2 = _rand_vecs(n2, seed + 1)
        s.add([f"a{i}" for i in range(n1)], v1, ns="alpha")
        s.add([f"b{i}" for i in range(n2)], v2, ns="beta")
        assert s.count("alpha") == n1
        assert s.count("beta") == n2
        assert s.count(ns=None) == n1 + n2


@given(
    n=st.integers(min_value=1, max_value=15),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=15, suppress_health_check=_SUPPRESS, deadline=None)
def test_forget_zeroes_ns_count(n, seed):
    s, d = _fresh_store()
    with d:
        vecs = _rand_vecs(n, seed)
        s.add([f"x{i}" for i in range(n)], vecs, ns="wipe-me")
        s.add(["keeper"], _rand_vecs(1, seed + 100), ns="keep")
        deleted = s.forget(ns="wipe-me")
        assert deleted == n
        assert s.count("wipe-me") == 0
        assert s.count("keep") == 1


# ── tier / gc invariants ──────────────────────────────────────────────────────

@given(
    n=st.integers(min_value=2, max_value=10),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=15, suppress_health_check=_SUPPRESS, deadline=None)
def test_gc_only_removes_tier2(n, seed):
    s, d = _fresh_store()
    with d:
        vecs = _rand_vecs(n, seed)
        ids = s.add([f"t{i}" for i in range(n)], vecs)
        s.pin(ids[0])
        for id_ in ids[2:]:
            s.set_tier(id_, 2)
        s.gc(age_days=0)
        assert s.get(ids[0]) is not None, "pinned item was gc'd"
        assert s.get(ids[1]) is not None, "tier-1 item was gc'd"
        for id_ in ids[2:]:
            assert s.get(id_) is None, f"tier-2 item {id_} survived gc"
