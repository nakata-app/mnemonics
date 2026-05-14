"""Peer-safety tests for Store: cross-process locking + stale-cache reload.

These simulate the "two peer agents write to the same ns" scenario that
caused the corrupt 14.6 GB sessions index on 2026-05-14. The key invariant:
a process that called add() AFTER a peer wrote new rows must see those rows
in its next search, without crashing or overwriting them.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np
import pytest

from mnemonics.store import Store


def _unit_vec(seed: int, dim: int = 384) -> np.ndarray:
    """Deterministic normalised vector for a given seed."""
    rng = np.random.RandomState(seed)
    v = rng.rand(dim).astype("float32")
    v /= float((v ** 2).sum() ** 0.5)
    return v


def test_wal_mode_active(tmp_path):
    s = Store(path=tmp_path)
    mode = s._db.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_lock_file_is_created_on_write(tmp_path):
    s = Store(path=tmp_path)
    vec = _unit_vec(1).reshape(1, -1)
    s.add(texts=["hello"], vectors=vec, ns="lockns")
    lock = tmp_path / "index_lockns.lock"
    assert lock.exists()


def test_mtime_watermark_updates_after_write(tmp_path):
    s = Store(path=tmp_path)
    vec = _unit_vec(1).reshape(1, -1)
    s.add(texts=["first"], vectors=vec, ns="mtimes")
    first_mtime = s._index_mtime["mtimes"]
    # Sleep past filesystem mtime granularity (HFS+/APFS = ~1µs, but be safe).
    time.sleep(0.01)
    s.add(texts=["second"], vectors=_unit_vec(2).reshape(1, -1), ns="mtimes")
    assert s._index_mtime["mtimes"] > first_mtime


def test_search_picks_up_peer_write_via_stale_reload(tmp_path):
    """Simulate a peer: first Store writes a row, second Store searches and
    must see it without restart (i.e. _reload_if_stale fires)."""
    writer = Store(path=tmp_path)
    reader = Store(path=tmp_path)

    # Reader warms up its in-memory index for the ns (empty so far).
    reader.search(_unit_vec(0), ns="peerns", top_k=1)

    # Writer adds a row from a separate Store instance.
    writer.add(texts=["peer-written row"], vectors=_unit_vec(7).reshape(1, -1), ns="peerns")

    # Reader's next search MUST include the peer's row, not its stale cache.
    out = reader.search(_unit_vec(7), ns="peerns", top_k=3)
    assert len(out) == 1
    assert out[0]["text"] == "peer-written row"


def _child_writer(path: str, ns: str, seed: int, texts: list[str]) -> None:
    """Worker for the multiprocessing test — opens its own Store handle."""
    s = Store(path=path)
    vecs = np.stack([_unit_vec(seed + i) for i in range(len(texts))])
    s.add(texts=texts, vectors=vecs, ns=ns)


def test_concurrent_multiprocess_writes_dont_corrupt_index(tmp_path):
    """Spawn two processes that each add 10 rows to the same ns. The file
    lock must serialise them so the index survives and contains all 20
    rows on the final reload."""
    ns = "racens"
    p1_texts = [f"p1-{i}" for i in range(10)]
    p2_texts = [f"p2-{i}" for i in range(10)]

    p1 = mp.Process(target=_child_writer, args=(str(tmp_path), ns, 100, p1_texts))
    p2 = mp.Process(target=_child_writer, args=(str(tmp_path), ns, 200, p2_texts))
    p1.start()
    p2.start()
    p1.join(timeout=30)
    p2.join(timeout=30)
    assert p1.exitcode == 0, f"p1 failed with exit={p1.exitcode}"
    assert p2.exitcode == 0, f"p2 failed with exit={p2.exitcode}"

    # Fresh Store after the fact must see all 20 rows in the DB AND the
    # vector index must still be loadable.
    s = Store(path=tmp_path)
    assert s.count(ns) == 20

    # Search must work — i.e. the on-disk index isn't corrupt.
    out = s.search(_unit_vec(105), ns=ns, top_k=20)
    found_texts = {r["text"] for r in out}
    # We can't guarantee which specific row scored top-K for a synthetic
    # random vector, but the index has to return SOME results without
    # raising, and the count must be sane.
    assert 1 <= len(out) <= 20
    assert all(t.startswith(("p1-", "p2-")) for t in found_texts)


def test_lock_is_per_namespace(tmp_path):
    """Two namespaces use independent locks: writing to one must not block
    reading the other. We verify by acquiring an exclusive lock on ns=A
    manually and then writing to ns=B — should succeed without timeout."""
    s = Store(path=tmp_path)
    import fcntl

    lock_a = tmp_path / "index_A.lock"
    lock_a.touch()
    fh = open(lock_a, "r+")
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    try:
        # If locks weren't per-ns, this would block forever (or until our
        # outer pytest timeout fires).
        s.add(texts=["independent"], vectors=_unit_vec(1).reshape(1, -1), ns="B")
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        fh.close()
    assert s.count("B") == 1


def test_reload_if_stale_noop_when_already_current(tmp_path):
    s = Store(path=tmp_path)
    s.add(texts=["seed"], vectors=_unit_vec(1).reshape(1, -1), ns="stalens")
    cached_mtime = s._index_mtime["stalens"]
    s._reload_if_stale("stalens")
    # Same mtime, no reload should have happened.
    assert s._index_mtime["stalens"] == cached_mtime


def test_reload_if_stale_noop_when_no_index_file(tmp_path):
    s = Store(path=tmp_path)
    # No file yet for this ns — must not raise.
    s._reload_if_stale("does-not-exist")
