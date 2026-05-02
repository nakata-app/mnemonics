"""Tests for mnemonics.store."""
import numpy as np
import pytest
from mnemonics.store import Store, DIM


def make_vecs(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random((n, DIM)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ── add / count ──────────────────────────────────────────────────────────────

def test_add_returns_ids(tmp_store):
    vecs = make_vecs(3)
    ids = tmp_store.add(["a", "b", "c"], vecs)
    assert len(ids) == 3
    assert ids == sorted(ids)  # autoincrement


def test_count_empty(tmp_store):
    assert tmp_store.count() == 0


def test_count_after_add(tmp_store):
    tmp_store.add(["x", "y"], make_vecs(2))
    assert tmp_store.count() == 2


def test_count_namespace_isolation(tmp_store):
    tmp_store.add(["a"], make_vecs(1), ns="ns1")
    tmp_store.add(["b", "c"], make_vecs(2), ns="ns2")
    assert tmp_store.count("ns1") == 1
    assert tmp_store.count("ns2") == 2
    assert tmp_store.count("ns3") == 0


# ── search ───────────────────────────────────────────────────────────────────

def test_search_returns_closest(tmp_store):
    vecs = make_vecs(5)
    tmp_store.add(["doc0", "doc1", "doc2", "doc3", "doc4"], vecs)
    results = tmp_store.search(vecs[2], top_k=1)
    assert results[0]["text"] == "doc2"
    assert results[0]["score"] > 0.99


def test_search_top_k_limits(tmp_store):
    tmp_store.add([f"d{i}" for i in range(10)], make_vecs(10))
    assert len(tmp_store.search(make_vecs(1)[0], top_k=3)) == 3


def test_search_empty_store(tmp_store):
    assert tmp_store.search(make_vecs(1)[0]) == []


def test_search_top_k_larger_than_store(tmp_store):
    tmp_store.add(["only"], make_vecs(1))
    results = tmp_store.search(make_vecs(1)[0], top_k=10)
    assert len(results) == 1


def test_search_namespace_isolation(tmp_store):
    v = make_vecs(2)
    tmp_store.add(["ns1-doc"], v[:1], ns="ns1")
    tmp_store.add(["ns2-doc"], v[1:], ns="ns2")
    results = tmp_store.search(v[0], ns="ns1", top_k=5)
    assert all(r["text"] == "ns1-doc" for r in results)


def test_search_score_range(tmp_store):
    vecs = make_vecs(5)
    tmp_store.add([f"d{i}" for i in range(5)], vecs)
    results = tmp_store.search(make_vecs(1)[0], top_k=5)
    for r in results:
        assert -0.01 <= r["score"] <= 1.01


def test_search_result_has_required_keys(tmp_store):
    vecs = make_vecs(1)
    tmp_store.add(["hello"], vecs)
    results = tmp_store.search(vecs[0])
    assert set(results[0].keys()) >= {"id", "text", "meta", "created", "score"}


# ── metadata ─────────────────────────────────────────────────────────────────

def test_meta_roundtrip(tmp_store):
    vecs = make_vecs(1)
    meta = [{"source": "book", "page": 42}]
    tmp_store.add(["text"], vecs, meta=meta)
    results = tmp_store.search(vecs[0])
    assert results[0]["meta"] == {"source": "book", "page": 42}


def test_meta_default_empty(tmp_store):
    vecs = make_vecs(1)
    tmp_store.add(["text"], vecs)
    results = tmp_store.search(vecs[0])
    assert results[0]["meta"] == {}


# ── delete ───────────────────────────────────────────────────────────────────

def test_delete_existing(tmp_store):
    ids = tmp_store.add(["to-delete"], make_vecs(1))
    assert tmp_store.delete(ids[0]) is True
    assert tmp_store.count() == 0


def test_delete_nonexistent(tmp_store):
    assert tmp_store.delete(99999) is False


def test_delete_does_not_affect_others(tmp_store):
    ids = tmp_store.add(["keep", "drop"], make_vecs(2))
    tmp_store.delete(ids[1])
    assert tmp_store.count() == 1


# ── namespaces ───────────────────────────────────────────────────────────────

def test_list_namespaces_empty(tmp_store):
    assert tmp_store.list_namespaces() == []


def test_list_namespaces(tmp_store):
    tmp_store.add(["a"], make_vecs(1), ns="alpha")
    tmp_store.add(["b"], make_vecs(1), ns="beta")
    ns = tmp_store.list_namespaces()
    assert set(ns) == {"alpha", "beta"}


# ── persistence ──────────────────────────────────────────────────────────────

def test_persists_across_reload(tmp_path):
    vecs = make_vecs(2)
    s1 = Store(tmp_path)
    s1.add(["persisted-a", "persisted-b"], vecs)
    del s1

    s2 = Store(tmp_path)
    assert s2.count() == 2
    results = s2.search(vecs[0], top_k=1)
    assert results[0]["text"] == "persisted-a"


# ── edge cases ───────────────────────────────────────────────────────────────

def test_add_single_item(tmp_store):
    ids = tmp_store.add(["solo"], make_vecs(1))
    assert len(ids) == 1


def test_large_batch(tmp_store):
    n = 500
    tmp_store.add([f"doc{i}" for i in range(n)], make_vecs(n))
    assert tmp_store.count() == n


def test_multiple_namespaces_independent_indices(tmp_store):
    v1 = make_vecs(3, seed=1)
    v2 = make_vecs(3, seed=2)
    tmp_store.add(["a0", "a1", "a2"], v1, ns="A")
    tmp_store.add(["b0", "b1", "b2"], v2, ns="B")
    ra = tmp_store.search(v1[0], ns="A", top_k=3)
    rb = tmp_store.search(v2[0], ns="B", top_k=3)
    assert {r["text"] for r in ra} == {"a0", "a1", "a2"}
    assert {r["text"] for r in rb} == {"b0", "b1", "b2"}
