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


def test_delete_removes_from_vector_index(tmp_path):
    vecs = make_vecs(1)
    s = Store(tmp_path)
    ids = s.add(["ghost-doc"], vecs)
    deleted_id = ids[0]

    # Search before delete — must find it
    assert any(r["id"] == deleted_id for r in s.search(vecs[0], top_k=5))

    s.delete(deleted_id)

    # After delete: not in search results
    assert all(r["id"] != deleted_id for r in s.search(vecs[0], top_k=5))

    # After reload from disk: still not in results (index was saved correctly)
    s2 = Store(tmp_path)
    assert all(r["id"] != deleted_id for r in s2.search(vecs[0], top_k=5))


def test_gc_removes_from_vector_index(tmp_path):
    vecs = make_vecs(1)
    s = Store(tmp_path)
    ids = s.add(["ambient-doc"], vecs)
    s.set_tier(ids[0], 2)  # make gc-eligible
    deleted_id = ids[0]

    count = s.gc(age_days=0)
    assert count == 1

    assert all(r["id"] != deleted_id for r in s.search(vecs[0], top_k=5))

    # Persist check
    s2 = Store(tmp_path)
    assert all(r["id"] != deleted_id for r in s2.search(vecs[0], top_k=5))


# ── forget ───────────────────────────────────────────────────────────────────

def test_forget_removes_entire_ns(tmp_path):
    vecs = make_vecs(3)
    s = Store(tmp_path)
    ids = s.add(["a", "b", "c"], vecs, ns="test-ns")
    assert s.count("test-ns") == 3

    n = s.forget(ns="test-ns")
    assert n == 3
    assert s.count("test-ns") == 0
    assert all(r["id"] not in ids for r in s.search(vecs[0], ns="test-ns", top_k=5))


def test_forget_skips_pinned_by_default(tmp_path):
    vecs = make_vecs(2)
    s = Store(tmp_path)
    ids = s.add(["pinned", "normal"], vecs, ns="test-ns")
    s.pin(ids[0])  # tier=0

    n = s.forget(ns="test-ns")
    assert n == 1  # only the non-pinned row
    assert s.count("test-ns") == 1


def test_forget_includes_pinned_when_explicit(tmp_path):
    vecs = make_vecs(1)
    s = Store(tmp_path)
    ids = s.add(["pinned"], vecs, ns="test-ns")
    s.pin(ids[0])

    n = s.forget(ns="test-ns", tier=0)
    assert n == 1
    assert s.count("test-ns") == 0


def test_forget_before_date(tmp_path):
    vecs = make_vecs(2)
    s = Store(tmp_path)
    ids = s.add(["old", "new"], vecs, ns="test-ns")
    # Force "old" to appear older via direct SQL
    s._db.execute("UPDATE memories SET created='2020-01-01' WHERE id=?", (ids[0],))
    s._db.commit()

    n = s.forget(ns="test-ns", before="2021-01-01")
    assert n == 1
    assert s.count("test-ns") == 1


def test_forget_candidates_dry_run(tmp_path):
    vecs = make_vecs(2)
    s = Store(tmp_path)
    ids = s.add(["a", "b"], vecs, ns="test-ns")

    candidates = s.forget_candidates(ns="test-ns")
    assert len(candidates) == 2
    assert s.count("test-ns") == 2  # not deleted


def test_forget_removes_from_vector_index(tmp_path):
    vecs = make_vecs(2)
    s = Store(tmp_path)
    s.add(["x", "y"], vecs, ns="test-ns")

    s.forget(ns="test-ns")
    assert s.count("test-ns") == 0
    # Reload and confirm vector index is clean
    s2 = Store(tmp_path)
    assert s2.search(vecs[0], ns="test-ns", top_k=5) == []


def test_forget_nonexistent_ns(tmp_store):
    assert tmp_store.forget(ns="no-such-ns") == 0


# ── health_check / doctor / repair ───────────────────────────────────────────

def test_repair_fixes_orphan_vectors(tmp_path):
    vecs = make_vecs(3)
    s = Store(tmp_path)
    ids = s.add(["a", "b", "c"], vecs, ns="alpha")
    s._db.execute("DELETE FROM memories WHERE id=?", (ids[2],))
    s._db.commit()

    result = s.repair()
    assert len(result["orphan_vectors_fixed"]) == 1
    assert result["orphan_vectors_fixed"][0]["removed"] == 1
    assert result["orphan_indexes_removed"] == []

    report = s.health_check()
    alpha = next(n for n in report["namespaces"] if n["ns"] == "alpha")
    assert alpha["soft_deleted"] == 0


def test_repair_removes_orphan_index(tmp_path):
    vecs = make_vecs(2)
    s = Store(tmp_path)
    s.add(["x", "y"], vecs, ns="beta")
    s._db.execute("DELETE FROM memories WHERE ns='beta'")
    s._db.commit()
    assert (tmp_path / "index_beta.bin").exists()

    result = s.repair()
    assert len(result["orphan_indexes_removed"]) == 1
    assert not (tmp_path / "index_beta.bin").exists()


def test_repair_reports_missing_vectors(tmp_path):
    import hnswlib
    vecs = make_vecs(2)
    s = Store(tmp_path)
    s.add(["x", "y"], vecs, ns="gamma")
    blank = hnswlib.Index(space="cosine", dim=s.dim)
    blank.init_index(max_elements=100, ef_construction=200, M=16)
    blank.save_index(str(tmp_path / "index_gamma.bin"))

    result = s.repair()
    assert any(m["ns"] == "gamma" for m in result["missing_vectors_reported"])


def test_rebuild_ns_index_removes_orphan_vectors(tmp_path):
    vecs = make_vecs(3)
    s = Store(tmp_path)
    ids = s.add(["a", "b", "c"], vecs, ns="alpha")

    # Simulate pre-fix raw-SQL delete (orphan vector in index)
    s._db.execute("DELETE FROM memories WHERE id=?", (ids[2],))
    s._db.commit()
    # Index still has 3 vectors; SQL has 2

    old_n, new_n = s.rebuild_ns_index("alpha")
    assert old_n == 3
    assert new_n == 2

    # Rebuilt index only returns the 2 surviving rows
    results = s.search(vecs[0], ns="alpha", top_k=5)
    returned_ids = {r["id"] for r in results}
    assert ids[2] not in returned_ids
    assert ids[0] in returned_ids or ids[1] in returned_ids


def test_rebuild_ns_index_persists(tmp_path):
    vecs = make_vecs(2)
    s = Store(tmp_path)
    ids = s.add(["x", "y"], vecs, ns="alpha")
    s._db.execute("DELETE FROM memories WHERE id=?", (ids[1],))
    s._db.commit()

    s.rebuild_ns_index("alpha")

    s2 = Store(tmp_path)
    results = s2.search(vecs[0], ns="alpha", top_k=5)
    assert all(r["id"] != ids[1] for r in results)


def test_rebuild_ns_index_empty_ns_no_orphan_bin(tmp_path):
    """rebuild_ns_index on a namespace with no SQL rows must not create an empty .bin file."""
    s = Store(tmp_path)
    # Namespace "ghost" has no SQL rows and no .bin file
    old_n, new_n = s.rebuild_ns_index("ghost")
    assert old_n == 0
    assert new_n == 0
    assert not (tmp_path / "index_ghost.bin").exists(), "empty .bin would become an orphan"


def test_health_check_clean_store(tmp_path):
    vecs = make_vecs(2)
    s = Store(tmp_path)
    s.add(["a", "b"], vecs, ns="alpha")
    report = s.health_check()
    assert report["db_integrity"] == "ok"
    ns_map = {n["ns"]: n for n in report["namespaces"]}
    assert "alpha" in ns_map
    assert ns_map["alpha"]["sql_count"] == 2
    assert ns_map["alpha"]["idx_count"] == 2
    assert ns_map["alpha"]["soft_deleted"] == 0
    assert ns_map["alpha"]["max_elements"] is not None
    assert ns_map["alpha"]["usage_pct"] is not None
    assert not ns_map["alpha"]["capacity_warning"]
    assert report["orphan_indexes"] == []


def test_health_check_capacity_warning(tmp_path):
    import hnswlib
    # Build a tiny index (max_elements=4) and fill it to 100% to trigger warning
    s = Store(tmp_path)
    vecs = make_vecs(4)
    s.add(["a", "b", "c", "d"], vecs, ns="tiny")
    # Force a tiny index with low max_elements
    idx_path = tmp_path / "index_tiny.bin"
    idx = hnswlib.Index(space="cosine", dim=s.dim)
    idx.init_index(max_elements=4, ef_construction=200, M=16)
    ids = [r[0] for r in s._db.execute("SELECT id FROM memories WHERE ns='tiny'").fetchall()]
    idx.add_items(vecs, ids)
    idx.save_index(str(idx_path))

    report = s.health_check()
    ns_map = {n["ns"]: n for n in report["namespaces"]}
    assert ns_map["tiny"]["usage_pct"] == 100.0
    assert ns_map["tiny"]["capacity_warning"]


def test_add_auto_resizes_full_index(tmp_path):
    import hnswlib
    # Create a tiny index at max capacity, then add one more item — must auto-resize
    s = Store(tmp_path)
    vecs = make_vecs(3)
    ids = s.add(["a", "b", "c"], vecs, ns="nano")

    # Shrink the index to exactly current_count so the next add overflows
    idx_path = tmp_path / "index_nano.bin"
    idx = hnswlib.Index(space="cosine", dim=s.dim)
    idx.init_index(max_elements=3, ef_construction=200, M=16)
    idx.add_items(vecs, ids)
    idx.save_index(str(idx_path))
    del s._index["nano"]  # evict cache so it reloads from disk

    extra_vec = make_vecs(1)
    new_ids = s.add(["d"], extra_vec, ns="nano")
    assert len(new_ids) == 1

    # Index must have grown
    idx2 = hnswlib.Index(space="cosine", dim=s.dim)
    idx2.load_index(str(idx_path))
    assert idx2.get_max_elements() > 3
    assert idx2.get_current_count() == 4


def test_health_check_detects_soft_deleted(tmp_path):
    vecs = make_vecs(2)
    s = Store(tmp_path)
    ids = s.add(["x", "y"], vecs, ns="alpha")
    s.delete(ids[0])
    report = s.health_check()
    ns_map = {n["ns"]: n for n in report["namespaces"]}
    assert ns_map["alpha"]["sql_count"] == 1
    assert ns_map["alpha"]["idx_count"] == 2   # mark_deleted keeps element in count
    assert ns_map["alpha"]["soft_deleted"] == 1


def test_health_check_orphan_index(tmp_path):
    vecs = make_vecs(1)
    s = Store(tmp_path)
    ids = s.add(["x"], vecs, ns="orphan-ns")
    # Force-delete from SQL only (bypassing store.delete to simulate old bug)
    s._db.execute("DELETE FROM memories WHERE id=?", (ids[0],))
    s._db.commit()
    report = s.health_check()
    orphan_ns = [o["ns"] for o in report["orphan_indexes"]]
    assert "orphan-ns" in orphan_ns


def test_gc_tier1_targets_default_tier(tmp_path):
    vecs = make_vecs(3)
    s = Store(tmp_path)
    ids = s.add(["ambient", "default-old", "default-new"], vecs, ns="test-ns")
    s.set_tier(ids[0], 2)   # tier-2: gc target
    s.set_tier(ids[1], 1)   # tier-1: gc target with --tier 1
    s.set_tier(ids[2], 1)   # tier-1: too recent (age_days=0 means >0 days, these are just created)

    # Standard gc (tier=2) only gets the ambient row
    cands_t2 = s.gc_candidates(tier=2, age_days=0)
    assert len(cands_t2) == 1
    assert cands_t2[0]["id"] == ids[0]

    # gc with tier=1 gets the two default rows
    cands_t1 = s.gc_candidates(tier=1, age_days=0)
    assert len(cands_t1) == 2
    assert {c["id"] for c in cands_t1} == {ids[1], ids[2]}


def test_gc_tier1_invalid_tier_raises(tmp_store):
    import pytest as _pytest
    with _pytest.raises(ValueError, match="tier must be"):
        tmp_store.gc_candidates(tier=0)


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


# ── summary column (raw + gist hybrid) ───────────────────────────────────────

def test_add_default_summary_is_none(tmp_store):
    tmp_store.add(["raw only"], make_vecs(1))
    results = tmp_store.search(make_vecs(1)[0])
    assert results[0]["summary"] is None


def test_summary_roundtrip(tmp_store):
    tmp_store.add(
        ["raw transcript"],
        make_vecs(1),
        summaries=["one-line gist"],
    )
    results = tmp_store.search(make_vecs(1)[0])
    assert results[0]["summary"] == "one-line gist"
    assert results[0]["text"] == "raw transcript"


def test_summary_length_mismatch_raises(tmp_store):
    with pytest.raises(ValueError):
        tmp_store.add(["a", "b"], make_vecs(2), summaries=["only one"])


def test_bm25_matches_summary_only_term(tmp_store):
    # Raw chunk has no overlap with the query word; the only path to a hit is
    # the BM25 mirror picking up the summary column.
    tmp_store.add(
        ["raw transcript content here"],
        make_vecs(1),
        summaries=["zeppelin disaster review"],
    )
    hits = tmp_store.search_bm25("zeppelin", top_k=5)
    assert len(hits) == 1
    assert hits[0]["summary"] == "zeppelin disaster review"


def test_bm25_still_matches_raw_text(tmp_store):
    tmp_store.add(
        ["the actual word is unicorn"],
        make_vecs(1),
        summaries=["unrelated gist"],
    )
    hits = tmp_store.search_bm25("unicorn", top_k=5)
    assert len(hits) == 1
    assert "unicorn" in hits[0]["text"]


def test_summary_survives_reload(tmp_path):
    s1 = Store(tmp_path)
    s1.add(["raw"], make_vecs(1), summaries=["gist"])
    del s1
    s2 = Store(tmp_path)
    hits = s2.search_bm25("gist", top_k=5)
    assert hits and hits[0]["summary"] == "gist"


def test_migration_adds_summary_column(tmp_path):
    """Pre-summary DBs must self-heal without losing rows or BM25 coverage."""
    import sqlite3 as _sqlite3

    legacy_db = tmp_path / "memories.db"
    # Re-create the v0.2.x schema, exactly as it shipped: single-column FTS
    # mirror, no `summary` column on `memories`. Inserting through it proves
    # the post-migration store can still see the row via BM25.
    conn = _sqlite3.connect(legacy_db)
    conn.executescript("""
        CREATE TABLE memories (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ns            TEXT NOT NULL DEFAULT 'default',
            text          TEXT NOT NULL,
            meta          TEXT NOT NULL DEFAULT '{}',
            created       TEXT NOT NULL DEFAULT (datetime('now')),
            tier          INTEGER NOT NULL DEFAULT 1,
            last_accessed TEXT,
            access_count  INTEGER NOT NULL DEFAULT 0
        );
        CREATE VIRTUAL TABLE memories_fts USING fts5(
            text,
            content='memories',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        );
        INSERT INTO memories (text) VALUES ('legacy row about pterodactyls');
        INSERT INTO memories_fts(rowid, text)
            SELECT id, text FROM memories;
    """)
    conn.commit()
    conn.close()

    # Re-open through Store; _migrate + _migrate_fts must add the column and
    # rebuild the FTS mirror with the two-column layout.
    store = Store(tmp_path)
    cols = {row[1] for row in store._db.execute("PRAGMA table_info(memories)").fetchall()}
    assert "summary" in cols

    fts_cols = {row[1] for row in store._db.execute("PRAGMA table_info(memories_fts)").fetchall()}
    assert "summary" in fts_cols

    # Pre-existing row should still be reachable via BM25 after the rebuild.
    hits = store.search_bm25("pterodactyls", top_k=5)
    assert hits and hits[0]["text"] == "legacy row about pterodactyls"
    assert hits[0]["summary"] is None


# ── Store.get() ───────────────────────────────────────────────────────────────

def test_get_returns_row(tmp_store):
    ids = tmp_store.add(["hello world"], make_vecs(1), ns="test")
    mid = ids[0]
    row = tmp_store.get(mid)
    assert row is not None
    assert row["id"] == mid
    assert row["text"] == "hello world"
    assert row["ns"] == "test"
    assert row["tier"] == 1
    assert "created" in row
    assert "last_accessed" in row
    assert "access_count" in row


def test_get_missing_returns_none(tmp_store):
    assert tmp_store.get(9999) is None


def test_get_after_pin(tmp_store):
    ids = tmp_store.add(["pin me"], make_vecs(1), ns="default")
    mid = ids[0]
    tmp_store.pin(mid)
    row = tmp_store.get(mid)
    assert row["tier"] == 0


# ── Store.update_summary() ────────────────────────────────────────────────────

def test_update_summary_sets_value(tmp_store):
    ids = tmp_store.add(["some text"], make_vecs(1))
    mid = ids[0]
    assert tmp_store.update_summary(mid, "a short gist") is True
    assert tmp_store.get(mid)["summary"] == "a short gist"


def test_update_summary_clears_value(tmp_store):
    ids = tmp_store.add(["some text"], make_vecs(1))
    mid = ids[0]
    tmp_store.update_summary(mid, "first")
    assert tmp_store.update_summary(mid, None) is True
    assert tmp_store.get(mid)["summary"] is None


def test_update_summary_returns_false_on_missing(tmp_store):
    assert tmp_store.update_summary(9999, "x") is False


def test_update_summary_fts_indexed(tmp_store):
    ids = tmp_store.add(["unrelated raw text"], make_vecs(1))
    mid = ids[0]
    tmp_store.update_summary(mid, "very specific gist phrase")
    hits = tmp_store.search_bm25("specific gist phrase", top_k=5)
    assert any(h["id"] == mid for h in hits)


# ── Store.list_memories() ─────────────────────────────────────────────────────

def test_list_memories_basic(tmp_store):
    tmp_store.add(["a", "b", "c"], make_vecs(3))
    rows = tmp_store.list_memories(ns="default", limit=10)
    assert len(rows) == 3
    assert rows[0]["id"] > rows[1]["id"]  # newest first


def test_list_memories_limit(tmp_store):
    tmp_store.add([f"x{i}" for i in range(10)], make_vecs(10))
    rows = tmp_store.list_memories(limit=3)
    assert len(rows) == 3


def test_list_memories_offset(tmp_store):
    tmp_store.add(["a", "b", "c"], make_vecs(3))
    page1 = tmp_store.list_memories(limit=2, offset=0)
    page2 = tmp_store.list_memories(limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 1
    assert {r["id"] for r in page1}.isdisjoint({r["id"] for r in page2})


def test_list_memories_tier_filter(tmp_store):
    ids = tmp_store.add(["p", "d", "a"], make_vecs(3))
    tmp_store.set_tier(ids[0], 0)
    tmp_store.set_tier(ids[1], 1)
    tmp_store.set_tier(ids[2], 2)
    pinned = tmp_store.list_memories(tier=0)
    assert len(pinned) == 1
    assert pinned[0]["tier"] == 0
    ambient = tmp_store.list_memories(tier=2)
    assert len(ambient) == 1


def test_list_memories_empty_ns(tmp_store):
    rows = tmp_store.list_memories(ns="ghost")
    assert rows == []


# ── search_bm25 ns isolation ─────────────────────────────────────────────────

def test_bm25_ns_isolation(tmp_store):
    tmp_store.add(["only in alpha"], make_vecs(1, seed=1), ns="alpha")
    tmp_store.add(["only in beta"], make_vecs(1, seed=2), ns="beta")
    hits_alpha = tmp_store.search_bm25("alpha", ns="alpha", top_k=5)
    hits_beta = tmp_store.search_bm25("alpha", ns="beta", top_k=5)
    assert any("alpha" in h["text"] for h in hits_alpha)
    assert hits_beta == []  # "alpha" text not in beta namespace


def test_bm25_empty_query_returns_empty(tmp_store):
    tmp_store.add(["some content"], make_vecs(1))
    assert tmp_store.search_bm25("", top_k=5) == []
    assert tmp_store.search_bm25("   ", top_k=5) == []


def test_bm25_score_is_positive(tmp_store):
    tmp_store.add(["keyword match here"], make_vecs(1))
    hits = tmp_store.search_bm25("keyword", top_k=5)
    assert hits and hits[0]["score"] > 0


# ── gc_candidates tier=0 guard ────────────────────────────────────────────────

def test_gc_candidates_rejects_tier0(tmp_store):
    with pytest.raises(ValueError, match="tier must be 1 or 2"):
        tmp_store.gc_candidates(tier=0)


def test_gc_candidates_rejects_invalid_tier(tmp_store):
    with pytest.raises(ValueError):
        tmp_store.gc_candidates(tier=3)


# ── search edge cases ─────────────────────────────────────────────────────────

def test_search_bm25_fts_operational_error(tmp_store, monkeypatch):
    """OperationalError during FTS query returns empty list, doesn't raise."""
    # Patch _fts_sanitize to return an unparseable MATCH expression.
    # FTS5 rejects unmatched parens → OperationalError → [] fallback.
    monkeypatch.setattr(tmp_store, "_fts_sanitize", lambda q: "(((")
    tmp_store.add(["some text"], make_vecs(1))
    results = tmp_store.search_bm25("some text", top_k=5)
    assert results == []


def test_search_skips_orphan_vector(tmp_store):
    """Vectors in the index that have no DB row are silently skipped."""
    import numpy as np
    from mnemonics.store import DIM
    vecs = make_vecs(1)
    ids = tmp_store.add(["text to delete"], vecs)
    # Remove from DB but NOT from index — creates an orphan vector
    tmp_store._db.execute("DELETE FROM memories WHERE id=?", (ids[0],))
    tmp_store._db.commit()
    # Ensure index is loaded
    tmp_store._index_for("default")
    # Search should return empty (orphan vector filtered out)
    results = tmp_store.search(vecs[0], ns="default", top_k=5)
    assert all(r["id"] != ids[0] for r in results)


# ── store exception path coverage ────────────────────────────────────────────

def test_set_tier_invalid_raises(tmp_store):
    ids = tmp_store.add(["text"], make_vecs(1))
    with pytest.raises(ValueError, match="tier must be"):
        tmp_store.set_tier(ids[0], 99)


def test_delete_mark_deleted_exception_is_swallowed(tmp_store, monkeypatch):
    from unittest.mock import MagicMock
    ids = tmp_store.add(["text"], make_vecs(1))
    bad_idx = MagicMock()
    bad_idx.mark_deleted.side_effect = Exception("boom")
    bad_idx.save_index.side_effect = Exception("boom")
    monkeypatch.setitem(tmp_store._index, "default", bad_idx)
    result = tmp_store.delete(ids[0])
    assert result is True  # SQL delete succeeded, exception swallowed


def test_gc_mark_deleted_exception_swallowed(tmp_store, monkeypatch):
    from unittest.mock import MagicMock
    ids = tmp_store.add(["old"], make_vecs(1))
    tmp_store.set_tier(ids[0], 2)  # make gc-eligible (tier=2 + access_count=0)
    bad_idx = MagicMock()
    bad_idx.mark_deleted.side_effect = Exception("boom")
    bad_idx.save_index.side_effect = Exception("boom")
    monkeypatch.setitem(tmp_store._index, "default", bad_idx)
    n = tmp_store.gc(ns="default", age_days=0, tier=2)
    assert n >= 0  # didn't raise


def test_forget_mark_deleted_exception_swallowed(tmp_store, monkeypatch):
    from unittest.mock import MagicMock
    tmp_store.add(["text to forget"], make_vecs(1))
    bad_idx = MagicMock()
    bad_idx.mark_deleted.side_effect = Exception("boom")
    bad_idx.save_index.side_effect = Exception("boom")
    monkeypatch.setitem(tmp_store._index, "default", bad_idx)
    n = tmp_store.forget("default")
    assert n >= 1  # SQL delete succeeded, index exception swallowed


def test_index_for_reload_from_disk(tmp_path):
    """_index_for loads existing .bin from disk (lines 252-254 in store.py)."""
    vecs = make_vecs(2)
    s = Store(tmp_path)
    s.add(["a", "b"], vecs)
    # Clear in-memory cache so _index_for must reload from disk.
    # Call _index_for directly (not search, which goes through _reload_if_stale).
    s._index.clear()
    s._index_mtime.clear()
    idx = s._index_for("default")
    assert idx is not None
    assert "default" in s._index_mtime


def test_repair_rebuild_runtime_error(tmp_path, monkeypatch):
    """repair() catches RuntimeError from rebuild_ns_index (lines 710-711)."""
    from unittest.mock import patch as _patch
    s = Store(tmp_path)
    s.add(["text"], make_vecs(1))
    # Force health_check to report soft_deleted > 0 by mocking it
    fake_report = {
        "namespaces": [{"ns": "default", "soft_deleted": 1, "missing_vectors": 0, "idx_missing": False}],
        "orphan_indexes": [],
        "integrity": "ok",
        "wal_kb": 0,
    }
    with _patch.object(s, "health_check", return_value=fake_report), \
         _patch.object(s, "rebuild_ns_index", side_effect=RuntimeError("collision")):
        result = s.repair()
    errors = [x for x in result["orphan_vectors_fixed"] if "error" in x]
    assert any("collision" in e["error"] for e in errors)


def test_repair_orphan_unlink_exception(tmp_path, monkeypatch):
    """repair() catches exception from orphan index unlink (lines 725-726)."""
    from unittest.mock import patch as _patch
    from pathlib import Path
    s = Store(tmp_path)
    orphan_path = str(tmp_path / "index_orphan.bin")
    fake_report = {
        "namespaces": [],
        "orphan_indexes": [{"path": orphan_path}],
        "integrity": "ok",
        "wal_kb": 0,
    }
    with _patch.object(s, "health_check", return_value=fake_report), \
         _patch.object(Path, "unlink", side_effect=OSError("locked")):
        result = s.repair()
    removed = result["orphan_indexes_removed"]
    assert any(isinstance(r, dict) and "error" in r for r in removed)


def test_ns_file_lock_no_fcntl(tmp_store, monkeypatch):
    """_ns_file_lock with _HAS_FCNTL=False yields without POSIX lock (lines 212-213)."""
    import mnemonics.store as _store_mod
    monkeypatch.setattr(_store_mod, "_HAS_FCNTL", False)
    with tmp_store._ns_file_lock("default", exclusive=True):
        # Should yield without error when fcntl is unavailable
        pass


def test_rebuild_ns_index_collision(tmp_path):
    """rebuild_ns_index raises RuntimeError when index path collides (lines 516-518)."""
    import os, shutil
    s = Store(tmp_path)
    s.add(["a", "b"], make_vecs(2), ns="alpha")
    # Simulate collision: create 'beta' namespace whose .bin resolves to same path
    alpha_bin = tmp_path / "index_alpha.bin"
    beta_bin = tmp_path / "index_beta.bin"
    # copy so resolve() gives same inode (symlink)
    beta_bin.symlink_to(alpha_bin)
    import pytest
    with pytest.raises(RuntimeError, match="collides"):
        s.rebuild_ns_index("beta")


def test_migrate_adds_missing_columns(tmp_path):
    """_migrate() adds missing columns when opening an older DB (lines 148, 150, 152)."""
    import sqlite3 as _sqlite
    # Create a bare-bones DB without the new columns
    db_path = tmp_path / "memories.db"
    conn = _sqlite.connect(str(db_path))
    conn.execute("""
        CREATE TABLE memories (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            ns      TEXT NOT NULL DEFAULT 'default',
            text    TEXT NOT NULL,
            meta    TEXT NOT NULL DEFAULT '{}',
            created TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ns ON memories(ns)")
    conn.commit()
    conn.close()
    # Store.__init__ should detect missing columns and add them
    s = Store(tmp_path)
    cols = {row[1] for row in s._db.execute("PRAGMA table_info(memories)").fetchall()}
    assert "tier" in cols
    assert "last_accessed" in cols
    assert "access_count" in cols
    assert "summary" in cols


def test_migrate_fts_rebuild_when_behind(tmp_path):
    """_migrate_fts line 201: 'rebuild' when fts_count(existing) < mem_count."""
    s = Store(tmp_path)
    s.add(["row one", "row two"], make_vecs(2))
    # Drop INSERT trigger so next INSERT goes to memories but NOT to FTS
    s._db.execute("DROP TRIGGER IF EXISTS memories_ai")
    s._db.execute(
        "INSERT INTO memories(ns, text, meta) VALUES('default', 'ghost row', '{}')"
    )
    s._db.commit()
    # Now fts_count(2) < mem_count(3) → elif branch (line 201) fires
    s._migrate_fts()
    count = s._db.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
    assert count >= 1


def test_apply_key_invalid_hex_raises(monkeypatch):
    """_apply_key raises when key is not 64 hex chars (lines 44-51)."""
    import sqlite3 as _sqlite
    import mnemonics.store as _store_mod
    import mnemonics.crypto as _crypto_mod

    monkeypatch.setattr(_store_mod, "_ENCRYPTED", True)
    monkeypatch.setattr(_crypto_mod, "require_key", lambda: "tooshort")
    conn = _sqlite.connect(":memory:")
    with pytest.raises(RuntimeError, match="64-character"):
        _store_mod._apply_key(conn)
    conn.close()


def test_apply_key_valid_hex_executes(monkeypatch):
    """_apply_key runs PRAGMA key when key is valid 64-char hex (line 52)."""
    import sqlite3 as _sqlite
    from unittest.mock import MagicMock
    import mnemonics.store as _store_mod
    import mnemonics.crypto as _crypto_mod

    valid_key = "a" * 64
    monkeypatch.setattr(_store_mod, "_ENCRYPTED", True)
    monkeypatch.setattr(_crypto_mod, "require_key", lambda: valid_key)
    conn = MagicMock()
    _store_mod._apply_key(conn)
    conn.execute.assert_called_once()
    call_arg = conn.execute.call_args[0][0]
    assert "PRAGMA key" in call_arg
    assert valid_key in call_arg


def test_wal_switch_error_swallowed(tmp_path, monkeypatch):
    """lines 130-133: OperationalError during WAL switch is silently swallowed."""
    import mnemonics.store as _s
    import sqlite3 as _sqlite

    original_connect = _sqlite.connect
    triggered = [False]

    class _WrapConn:
        """Thin wrapper so we can intercept execute without mutating C extension."""
        def __init__(self, real):
            self._real = real
        def __getattr__(self, name):
            return getattr(self._real, name)
        def execute(self, sql, *a, **k):
            if "journal_mode=WAL" in sql and not triggered[0]:
                triggered[0] = True
                raise _sqlite.OperationalError("locked by peer")
            return self._real.execute(sql, *a, **k)
        def executescript(self, sql):
            return self._real.executescript(sql)

    def patched_connect(path, **kw):
        return _WrapConn(original_connect(path, **kw))

    monkeypatch.setattr(_s.sqlite3, "connect", patched_connect)
    s = _s.Store(tmp_path)
    assert s is not None
    assert triggered[0]


def test_rebuild_ns_index_get_items_exception_skips(tmp_path, monkeypatch):
    """lines 553-554: get_items exception in rebuild_ns_index → item skipped."""
    from unittest.mock import MagicMock, patch as _patch
    import mnemonics.store as _s

    s = Store(tmp_path)
    s.add(["a", "b"], make_vecs(2), ns="ns1")

    # patch hnswlib.Index.get_items to fail for the second call
    call_count = [0]
    orig_get_items = None

    import hnswlib
    real_get_items = hnswlib.Index.get_items

    def flaky_get_items(self, ids):
        call_count[0] += 1
        if call_count[0] == 2:
            raise Exception("disk read error")
        return real_get_items(self, ids)

    monkeypatch.setattr(hnswlib.Index, "get_items", flaky_get_items)
    # Force rebuild — should succeed despite one item's vector being unreadable
    s.rebuild_ns_index("ns1")
    # Index still exists and is usable
    results = s.search(make_vecs(1)[0], ns="ns1", top_k=5)
    assert isinstance(results, list)


def test_health_check_load_index_exception(tmp_path):
    """lines 654-655: corrupted .bin file → load_index Exception → idx_count=None."""
    s = Store(tmp_path)
    s.add(["x"], make_vecs(1))
    # Corrupt the index file
    bin_path = tmp_path / "index_default.bin"
    bin_path.write_bytes(b"this is not a valid hnswlib index")
    s._index.clear()
    report = s.health_check()
    # Should not raise; idx_count falls back to None → soft_deleted=0, missing_vectors=0
    ns = next(r for r in report["namespaces"] if r["ns"] == "default")
    assert ns["soft_deleted"] == 0
    assert ns["missing_vectors"] == 0


def test_count_all_namespaces_with_none(tmp_store):
    """count(ns=None) sums across all namespaces."""
    vecs = make_vecs(3)
    tmp_store.add(["a"], vecs[:1], ns="ns1")
    tmp_store.add(["b", "c"], vecs[1:], ns="ns2")
    assert tmp_store.count(ns=None) == 3
    assert tmp_store.count("ns1") == 1
    assert tmp_store.count("ns2") == 2


def test_search_bm25_min_tier_filter(tmp_path):
    """search_bm25 min_tier excludes items below threshold."""
    from mnemonics.ingest import ingest
    s = Store(tmp_path)
    ids = s.add(["pinned document keyword", "ambient document keyword"], make_vecs(2))
    s.pin(ids[0])      # tier=0
    s.set_tier(ids[1], 2)  # tier=2

    # min_tier=1 → excludes tier-0 (pinned)
    results = s.search_bm25("keyword", min_tier=1)
    result_ids = {r["id"] for r in results}
    assert ids[0] not in result_ids
    assert ids[1] in result_ids


def test_search_bm25_max_tier_filter(tmp_path):
    """search_bm25 max_tier excludes items above threshold."""
    s = Store(tmp_path)
    ids = s.add(["pinned document keyword", "ambient document keyword"], make_vecs(2))
    s.pin(ids[0])      # tier=0
    s.set_tier(ids[1], 2)  # tier=2

    # max_tier=1 → excludes tier-2 (ambient)
    results = s.search_bm25("keyword", max_tier=1)
    result_ids = {r["id"] for r in results}
    assert ids[0] in result_ids
    assert ids[1] not in result_ids


def test_search_bm25_tier_range_filter(tmp_path):
    """search_bm25 min_tier + max_tier narrow to exact tier."""
    s = Store(tmp_path)
    ids = s.add(
        ["pin keyword", "def keyword", "amb keyword"],
        make_vecs(3),
    )
    s.pin(ids[0])
    # ids[1] stays tier=1 (default)
    s.set_tier(ids[2], 2)

    # only tier=1 (default)
    results = s.search_bm25("keyword", min_tier=1, max_tier=1)
    result_ids = {r["id"] for r in results}
    assert ids[1] in result_ids
    assert ids[0] not in result_ids
    assert ids[2] not in result_ids


def test_search_min_tier_excludes_pinned(tmp_path):
    """search min_tier=1 excludes tier-0 (pinned) items."""
    import numpy as np
    from mnemonics.store import DIM, Store
    rng = np.random.default_rng(42)
    s = Store(tmp_path)
    v = rng.random((2, DIM)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    ids = s.add(["pinned memory", "default memory"], v)
    s.pin(ids[0])  # tier=0
    results = s.search(v[0], top_k=5, min_tier=1)
    result_ids = {r["id"] for r in results}
    assert ids[0] not in result_ids
    assert ids[1] in result_ids


def test_search_max_tier_excludes_ambient(tmp_path):
    """search max_tier=1 excludes tier-2 (ambient) items."""
    import numpy as np
    from mnemonics.store import DIM, Store
    rng = np.random.default_rng(99)
    s = Store(tmp_path)
    v = rng.random((2, DIM)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    ids = s.add(["default memory", "ambient memory"], v)
    s.set_tier(ids[1], 2)  # tier=2
    results = s.search(v[1], top_k=5, max_tier=1)
    result_ids = {r["id"] for r in results}
    assert ids[1] not in result_ids
    assert ids[0] in result_ids


def test_get_many_returns_in_request_order(tmp_path):
    """get_many preserves the order of the requested IDs."""
    s = Store(tmp_path)
    ids = s.add(["alpha", "beta", "gamma"], make_vecs(3))
    # request in reverse order
    result = s.get_many([ids[2], ids[0], ids[1]])
    assert [r["text"] for r in result] == ["gamma", "alpha", "beta"]


def test_get_many_skips_missing_ids(tmp_path):
    """get_many silently omits IDs that don't exist."""
    s = Store(tmp_path)
    ids = s.add(["only-one"], make_vecs(1))
    result = s.get_many([ids[0], 99999])
    assert len(result) == 1
    assert result[0]["text"] == "only-one"


def test_get_many_empty_input(tmp_path):
    """get_many([]) returns []."""
    s = Store(tmp_path)
    assert s.get_many([]) == []
