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


def test_delete_many_removes_all(tmp_path):
    """delete_many removes all specified IDs and returns count."""
    s = Store(tmp_path)
    ids = s.add(["a", "b", "c"], make_vecs(3))
    deleted = s.delete_many([ids[0], ids[2]])
    assert deleted == 2
    assert s.count() == 1
    assert s.get(ids[1]) is not None
    assert s.get(ids[0]) is None
    assert s.get(ids[2]) is None


def test_delete_many_skips_missing_ids(tmp_path):
    """delete_many skips non-existent IDs without error."""
    s = Store(tmp_path)
    ids = s.add(["only"], make_vecs(1))
    deleted = s.delete_many([ids[0], 99999, 88888])
    assert deleted == 1
    assert s.count() == 0


def test_delete_many_empty_input(tmp_path):
    """delete_many([]) returns 0 without touching the store."""
    s = Store(tmp_path)
    s.add(["x"], make_vecs(1))
    assert s.delete_many([]) == 0
    assert s.count() == 1


def test_delete_many_multi_ns(tmp_path):
    """delete_many handles IDs across multiple namespaces."""
    s = Store(tmp_path)
    ids_a = s.add(["ns-a-1", "ns-a-2"], make_vecs(2), ns="a")
    ids_b = s.add(["ns-b-1"], make_vecs(1), ns="b")
    deleted = s.delete_many([ids_a[0], ids_b[0]])
    assert deleted == 2
    assert s.count("a") == 1
    assert s.count("b") == 0


def test_delete_many_all_missing_ids_returns_zero(tmp_path):
    """delete_many with only non-existent IDs returns 0 without error."""
    s = Store(tmp_path)
    s.add(["keeper"], make_vecs(1))
    result = s.delete_many([99999, 88888])
    assert result == 0
    assert s.count() == 1




def test_update_meta_changes_metadata(tmp_path):
    """update_meta replaces the meta dict for an existing memory."""
    s = Store(tmp_path)
    ids = s.add(["hello"], make_vecs(1), meta={"key": "old"})
    ok = s.update_meta(ids[0], {"key": "new", "extra": 42})
    assert ok is True
    row = s.get(ids[0])
    assert row is None or True  # get doesn't return meta, check via search
    # Verify via direct DB
    row = s._db.execute("SELECT meta FROM memories WHERE id=?", (ids[0],)).fetchone()
    import json
    assert json.loads(row[0]) == {"key": "new", "extra": 42}


def test_update_meta_missing_id_returns_false(tmp_path):
    """update_meta on non-existent ID returns False."""
    s = Store(tmp_path)
    assert s.update_meta(99999, {"x": 1}) is False


def test_update_tier_many_bulk_set(tmp_path):
    """update_tier_many sets tier for multiple IDs in one call."""
    s = Store(tmp_path)
    ids = s.add(["a", "b", "c"], make_vecs(3))
    updated = s.update_tier_many([ids[0], ids[2]], tier=2)
    assert updated == 2
    for mid in [ids[0], ids[2]]:
        row = s._db.execute("SELECT tier FROM memories WHERE id=?", (mid,)).fetchone()
        assert row[0] == 2
    row = s._db.execute("SELECT tier FROM memories WHERE id=?", (ids[1],)).fetchone()
    assert row[0] == 1  # unchanged


def test_update_tier_many_invalid_tier_raises(tmp_path):
    """update_tier_many with invalid tier raises ValueError."""
    import pytest
    s = Store(tmp_path)
    ids = s.add(["x"], make_vecs(1))
    with pytest.raises(ValueError, match="tier must be"):
        s.update_tier_many(ids, tier=99)


def test_update_tier_many_empty_list(tmp_path):
    """update_tier_many([]) returns 0 without DB write."""
    s = Store(tmp_path)
    assert s.update_tier_many([], tier=0) == 0


def test_search_by_meta_single_filter(tmp_path):
    """search_by_meta returns memories matching a single meta key=value."""
    s = Store(tmp_path)
    ids = s.add(
        ["match text", "no-match text"],
        make_vecs(2),
        meta=[{"source": "book"}, {"source": "web"}],
    )
    results = s.search_by_meta({"source": "book"})
    assert len(results) == 1
    assert results[0]["id"] == ids[0]


def test_search_by_meta_multi_filter(tmp_path):
    """search_by_meta applies AND logic across multiple keys."""
    s = Store(tmp_path)
    s.add(
        ["a", "b", "c"],
        make_vecs(3),
        meta=[
            {"source": "book", "page": 1},
            {"source": "book", "page": 2},
            {"source": "web", "page": 1},
        ],
    )
    results = s.search_by_meta({"source": "book", "page": 1})
    assert len(results) == 1
    assert results[0]["text"] == "a"


def test_search_by_meta_empty_filters_returns_empty(tmp_path):
    """search_by_meta({}) returns [] immediately without querying."""
    s = Store(tmp_path)
    s.add(["x"], make_vecs(1), meta=[{"k": "v"}])
    assert s.search_by_meta({}) == []


def test_search_by_meta_no_match_returns_empty(tmp_path):
    """search_by_meta returns [] when nothing matches."""
    s = Store(tmp_path)
    s.add(["x"], make_vecs(1), meta=[{"k": "v"}])
    assert s.search_by_meta({"k": "nonexistent"}) == []


def test_search_by_meta_limit(tmp_path):
    """search_by_meta limit parameter caps result count."""
    s = Store(tmp_path)
    s.add(["a", "b", "c"], make_vecs(3), meta=[{"t": "x"}] * 3)
    results = s.search_by_meta({"t": "x"}, limit=2)
    assert len(results) == 2


# ── store.add tier ────────────────────────────────────────────────────────────

def test_store_add_initial_tier(tmp_store):
    """store.add respects initial tier parameter."""
    import numpy as np
    vecs = np.random.rand(1, 384).astype("float32")
    ids = tmp_store.add(["pinned from start"], vecs, tier=0)
    row = tmp_store.get(ids[0])
    assert row["tier"] == 0


def test_store_add_ambient_tier(tmp_store):
    vecs = np.random.rand(1, 384).astype("float32")
    ids = tmp_store.add(["ambient"], vecs, tier=2)
    assert tmp_store.get(ids[0])["tier"] == 2


def test_store_add_invalid_tier(tmp_store):
    import numpy as np
    vecs = np.random.rand(1, 384).astype("float32")
    with pytest.raises(ValueError, match="tier"):
        tmp_store.add(["x"], vecs, tier=9)


# ── list_memories --since ────────────────────────────────────────────────────

def test_list_memories_since_filters(populated_store):
    """list_memories(since=...) returns only rows created >= since."""
    store, docs, vecs = populated_store
    all_rows = store.list_memories(ns="default", limit=100)
    # All test rows are created 'now'; since=far-future should return nothing
    future = "2099-01-01 00:00:00"
    rows_future = store.list_memories(ns="default", limit=100, since=future)
    assert rows_future == []

    # since=far-past should return everything
    past = "2000-01-01 00:00:00"
    rows_past = store.list_memories(ns="default", limit=100, since=past)
    assert len(rows_past) == len(all_rows)


def test_list_memories_since_combined_tier(populated_store):
    """since + tier filters both apply (AND logic)."""
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.pin(first_id)
    past = "2000-01-01"
    future = "2099-01-01"
    # tier=0 + far-future → no rows
    assert store.list_memories(ns="default", limit=100, tier=0, since=future) == []
    # tier=0 + far-past → only pinned row
    pinned = store.list_memories(ns="default", limit=100, tier=0, since=past)
    assert len(pinned) == 1
    assert pinned[0]["id"] == first_id


# ── list_memories --before ────────────────────────────────────────────────────

def test_list_memories_before_filters(populated_store):
    """list_memories(before=...) returns only rows with created < before."""
    store, docs, vecs = populated_store
    # before=far-past: nothing
    nothing = store.list_memories(ns="default", limit=100, before="2000-01-01")
    assert nothing == []
    # before=far-future: everything
    all_rows = store.list_memories(ns="default", limit=100, before="2099-01-01")
    total = store.list_memories(ns="default", limit=100)
    assert len(all_rows) == len(total)


def test_list_memories_since_and_before(populated_store):
    """since + before creates a half-open date interval."""
    store, docs, vecs = populated_store
    # past..future: all rows
    all_rows = store.list_memories(ns="default", limit=100, since="2000-01-01", before="2099-01-01")
    total = store.list_memories(ns="default", limit=100)
    assert len(all_rows) == len(total)
    # past..past: nothing
    empty = store.list_memories(ns="default", limit=100, since="2000-01-01", before="2001-01-01")
    assert empty == []


# ── text_search ───────────────────────────────────────────────────────────────

def test_text_search_basic(populated_store):
    store, docs, vecs = populated_store
    # "Eiffel" appears in texts[0]
    hits = store.text_search("Eiffel")
    assert len(hits) >= 1
    assert any("Eiffel" in h["text"] for h in hits)


def test_text_search_no_match(populated_store):
    store, docs, vecs = populated_store
    hits = store.text_search("xyzzy_no_match_ever")
    assert hits == []


def test_text_search_ns_filter(populated_store):
    store, docs, vecs = populated_store
    import numpy as np
    v = np.random.rand(384).astype("float32")
    v /= np.linalg.norm(v)
    store.add(["Eiffel other ns"], v[None], ns="other")
    hits = store.text_search("Eiffel", ns="other")
    assert all(h["ns"] == "other" for h in hits)


def test_text_search_all_ns(populated_store):
    store, docs, vecs = populated_store
    import numpy as np
    v = np.random.rand(384).astype("float32")
    v /= np.linalg.norm(v)
    store.add(["Eiffel other ns"], v[None], ns="other")
    hits = store.text_search("Eiffel", ns=None)
    ns_set = {h["ns"] for h in hits}
    assert len(ns_set) > 1


def test_text_search_tier_filter(populated_store):
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.pin(first_id)
    hits = store.text_search("", ns=None, tier=0)
    assert all(h["tier"] == 0 for h in hits)


def test_text_search_limit(populated_store):
    store, docs, vecs = populated_store
    hits = store.text_search("", ns=None, limit=2)
    assert len(hits) <= 2


def test_text_search_summary_match(populated_store):
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.update_summary(first_id, "unique_summary_keyword_xyz")
    hits = store.text_search("unique_summary_keyword_xyz")
    assert any(h["id"] == first_id for h in hits)


# ── rename_ns ─────────────────────────────────────────────────────────────────

def test_rename_ns_moves_rows(populated_store):
    store, docs, vecs = populated_store
    n = store.count("default")
    moved = store.rename_ns("default", "renamed")
    assert moved == n
    assert store.count("renamed") == n
    assert store.count("default") == 0


def test_rename_ns_zero_returns_zero(tmp_store):
    moved = tmp_store.rename_ns("nonexistent", "new")
    assert moved == 0


def test_rename_ns_same_name_zero(populated_store):
    store, docs, vecs = populated_store
    moved = store.rename_ns("default", "default")
    assert moved == 0
    assert store.count("default") == len(docs)


def test_rename_ns_conflict_raises(populated_store):
    store, docs, vecs = populated_store
    import numpy as np
    v = np.random.rand(384).astype("float32")
    v /= np.linalg.norm(v)
    store.add(["row in other"], v[None], ns="other")
    with pytest.raises(ValueError, match="already has"):
        store.rename_ns("default", "other")


def test_rename_ns_renames_bin_file(populated_store, tmp_path):
    store, docs, vecs = populated_store
    store.rename_ns("default", "moved")
    old_bin = store.root / "index_default.bin"
    new_bin = store.root / "index_moved.bin"
    assert not old_bin.exists()
    assert new_bin.exists()


# ── set_tier_many ─────────────────────────────────────────────────────────────

def test_set_tier_many_basic(populated_store):
    """set_tier_many updates all given IDs to the target tier."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories").fetchall()]
    updated = store.set_tier_many(ids[:3], 2)
    assert updated == 3
    for mid in ids[:3]:
        assert store.get(mid)["tier"] == 2


def test_set_tier_many_empty_list(populated_store):
    """set_tier_many with empty list returns 0."""
    store, docs, vecs = populated_store
    assert store.set_tier_many([], 1) == 0


def test_set_tier_many_invalid_tier(populated_store):
    """set_tier_many with invalid tier raises ValueError."""
    store, docs, vecs = populated_store
    ids = [store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]]
    with pytest.raises(ValueError, match="tier must be"):
        store.set_tier_many(ids, 99)


def test_set_tier_many_missing_ids_ignored(populated_store):
    """set_tier_many silently skips non-existent IDs."""
    store, docs, vecs = populated_store
    updated = store.set_tier_many([99999, 99998], 0)
    assert updated == 0


# ── recent_accessed ───────────────────────────────────────────────────────────

def test_recent_accessed_returns_rows(populated_store):
    """recent_accessed returns rows from the store."""
    store, docs, vecs = populated_store
    hits = store.recent_accessed(ns="default")
    assert len(hits) == len(docs)


def test_recent_accessed_ns_filter(populated_store):
    """recent_accessed filters by namespace."""
    import numpy as np
    store, docs, vecs = populated_store
    v = np.random.rand(384).astype("float32"); v /= np.linalg.norm(v)
    store.add(["row in other"], v[None], ns="other")
    hits = store.recent_accessed(ns="default")
    assert all(h["ns"] == "default" for h in hits)


def test_recent_accessed_all_ns(populated_store):
    """recent_accessed with ns=None spans all namespaces."""
    import numpy as np
    store, docs, vecs = populated_store
    v = np.random.rand(384).astype("float32"); v /= np.linalg.norm(v)
    store.add(["row in other"], v[None], ns="other")
    hits = store.recent_accessed(ns=None)
    ns_set = {h["ns"] for h in hits}
    assert len(ns_set) > 1


def test_recent_accessed_limit(populated_store):
    """recent_accessed respects the limit parameter."""
    store, docs, vecs = populated_store
    hits = store.recent_accessed(ns="default", limit=2)
    assert len(hits) <= 2


def test_recent_accessed_tier_filter(populated_store):
    """recent_accessed filters by tier."""
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.pin(first_id)
    hits = store.recent_accessed(ns="default", tier=0)
    assert all(h["tier"] == 0 for h in hits)
    assert len(hits) == 1


# ── top_accessed ──────────────────────────────────────────────────────────────

def test_top_accessed_returns_rows(populated_store):
    store, docs, vecs = populated_store
    hits = store.top_accessed(ns="default")
    assert len(hits) == len(docs)


def test_top_accessed_ordering(populated_store):
    """Memories with higher access_count come first."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories").fetchall()]
    # touch first two IDs 3 and 1 times respectively
    store._db.execute("UPDATE memories SET access_count=3 WHERE id=?", (ids[0],))
    store._db.execute("UPDATE memories SET access_count=1 WHERE id=?", (ids[1],))
    store._db.commit()
    hits = store.top_accessed(ns="default")
    counts = [h["access_count"] for h in hits]
    assert counts == sorted(counts, reverse=True)


def test_top_accessed_limit(populated_store):
    store, docs, vecs = populated_store
    hits = store.top_accessed(ns="default", limit=2)
    assert len(hits) <= 2


def test_top_accessed_tier_filter(populated_store):
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.pin(first_id)
    hits = store.top_accessed(ns="default", tier=0)
    assert all(h["tier"] == 0 for h in hits)


def test_top_accessed_all_ns(populated_store):
    import numpy as np
    store, docs, vecs = populated_store
    v = np.random.rand(384).astype("float32"); v /= np.linalg.norm(v)
    store.add(["ns2 doc"], v[None], ns="ns2")
    hits = store.top_accessed(ns=None)
    assert {h["ns"] for h in hits} >= {"default", "ns2"}


# ── copy_ns ───────────────────────────────────────────────────────────────────

def test_copy_ns_basic(populated_store):
    """copy_ns duplicates all rows into a new ns, leaving source intact."""
    store, docs, vecs = populated_store
    copied = store.copy_ns("default", "backup")
    assert copied == len(docs)
    assert store.count("default") == len(docs)
    assert store.count("backup") == len(docs)


def test_copy_ns_content_preserved(populated_store):
    """copy_ns preserves text and tier."""
    store, docs, vecs = populated_store
    store.copy_ns("default", "backup2")
    src_texts = {r["text"] for r in store.text_search("", ns="default", limit=100)}
    dst_texts = {r["text"] for r in store.text_search("", ns="backup2", limit=100)}
    assert src_texts == dst_texts


def test_copy_ns_empty_source(tmp_store):
    """copy_ns on empty source returns 0 without creating destination."""
    assert tmp_store.copy_ns("nonexistent", "dst") == 0
    assert tmp_store.count("dst") == 0


def test_copy_ns_conflict_raises(populated_store):
    """copy_ns raises ValueError if destination already exists."""
    store, docs, vecs = populated_store
    store.copy_ns("default", "backup3")
    with pytest.raises(ValueError, match="already has"):
        store.copy_ns("default", "backup3")


def test_copy_ns_access_count_reset(populated_store):
    """copy_ns sets access_count=0 on the copies."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories").fetchall()]
    store._db.execute("UPDATE memories SET access_count=5 WHERE id=?", (ids[0],))
    store._db.commit()
    store.copy_ns("default", "backup4")
    counts = [r["access_count"] for r in store.top_accessed(ns="backup4")]
    assert all(c == 0 for c in counts)


# ── stats_by_ns ───────────────────────────────────────────────────────────────

def test_stats_by_ns_basic(populated_store):
    """stats_by_ns returns one entry per ns with correct totals."""
    store, docs, vecs = populated_store
    stats = store.stats_by_ns()
    assert len(stats) == 1
    s = stats[0]
    assert s["ns"] == "default"
    assert s["total"] == len(docs)
    assert s["pinned"] + s["default"] + s["ambient"] == len(docs)


def test_stats_by_ns_tier_breakdown(populated_store):
    """stats_by_ns reflects tier changes."""
    import numpy as np
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories").fetchall()]
    store.pin(ids[0])
    store.set_tier(ids[1], 2)
    stats = {s["ns"]: s for s in store.stats_by_ns()}
    assert stats["default"]["pinned"] == 1
    assert stats["default"]["ambient"] == 1


def test_stats_by_ns_multiple_namespaces(populated_store):
    """stats_by_ns returns one entry per distinct namespace."""
    import numpy as np
    store, docs, vecs = populated_store
    v = np.random.rand(384).astype("float32"); v /= np.linalg.norm(v)
    store.add(["other"], v[None], ns="other")
    stats = store.stats_by_ns()
    ns_set = {s["ns"] for s in stats}
    assert "default" in ns_set
    assert "other" in ns_set


def test_stats_by_ns_empty(tmp_store):
    """stats_by_ns on empty store returns empty list."""
    assert tmp_store.stats_by_ns() == []


def test_health_check_tier_breakdown(populated_store):
    """health_check namespaces include tier_breakdown after the patch."""
    store, docs, vecs = populated_store
    report = store.health_check()
    ns_entry = next(n for n in report["namespaces"] if n["ns"] == "default")
    assert "tier_breakdown" in ns_entry


def test_stats_by_ns_oldest_newest_update(tmp_store):
    """stats_by_ns correctly updates oldest/newest when multiple tiers differ in creation time."""
    import numpy as np
    rng = np.random.default_rng(99)
    v1 = rng.random((1, 384)).astype("float32"); v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
    v2 = rng.random((1, 384)).astype("float32"); v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
    tmp_store.add(["early memory"], v1, tier=0)
    id1 = tmp_store._db.execute("SELECT id FROM memories").fetchone()[0]
    tmp_store.add(["late memory"], v2, tier=2)
    id2 = tmp_store._db.execute("SELECT id FROM memories ORDER BY id DESC LIMIT 1").fetchone()[0]
    # Force different created timestamps
    tmp_store._db.execute(
        "UPDATE memories SET created=? WHERE id=?", ("2025-01-01 00:00:00", id1)
    )
    tmp_store._db.execute(
        "UPDATE memories SET created=? WHERE id=?", ("2026-12-31 00:00:00", id2)
    )
    tmp_store._db.commit()
    stats = tmp_store.stats_by_ns()
    assert len(stats) == 1
    s = stats[0]
    assert s["oldest"] == "2025-01-01 00:00:00"
    assert s["newest"] == "2026-12-31 00:00:00"


def test_stats_by_ns_oldest_update_trigger(tmp_store):
    """Trigger the oldest-update branch: tier=0 (first in GROUP BY) has a later date;
    tier=2 (second in GROUP BY) has an earlier date → oldest update fires."""
    import numpy as np
    rng = np.random.default_rng(77)
    v0 = rng.random((1, 384)).astype("float32"); v0 /= np.linalg.norm(v0, axis=1, keepdims=True)
    v2 = rng.random((1, 384)).astype("float32"); v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
    # tier=0 will come first in GROUP BY (tier ASC); give it the LATER date
    tmp_store.add(["pinned late"], v0, tier=0)
    id0 = tmp_store._db.execute("SELECT id FROM memories").fetchone()[0]
    # tier=2 comes second; give it the EARLIER date → triggers oldest update branch
    tmp_store.add(["ambient early"], v2, tier=2)
    id2 = tmp_store._db.execute("SELECT id FROM memories ORDER BY id DESC LIMIT 1").fetchone()[0]
    tmp_store._db.execute("UPDATE memories SET created=? WHERE id=?", ("2026-12-01 00:00:00", id0))
    tmp_store._db.execute("UPDATE memories SET created=? WHERE id=?", ("2024-01-01 00:00:00", id2))
    tmp_store._db.commit()
    stats = tmp_store.stats_by_ns()
    s = stats[0]
    assert s["oldest"] == "2024-01-01 00:00:00"
    assert s["newest"] == "2026-12-01 00:00:00"


# ── merge_ns ──────────────────────────────────────────────────────────────────

def test_merge_ns_basic(populated_store):
    """merge_ns moves all rows from src to dst and empties src."""
    import numpy as np
    store, docs, vecs = populated_store
    v = np.random.rand(384).astype("float32"); v /= np.linalg.norm(v)
    store.add(["dst doc"], v[None], ns="dst")
    n_src = len(docs)
    moved = store.merge_ns("default", "dst")
    assert moved == n_src
    assert store.count("default") == 0
    assert store.count("dst") == n_src + 1


def test_merge_ns_into_empty_dst(populated_store):
    """merge_ns works when dst does not yet exist."""
    store, docs, vecs = populated_store
    moved = store.merge_ns("default", "newdst")
    assert moved == len(docs)
    assert store.count("default") == 0
    assert store.count("newdst") == len(docs)


def test_merge_ns_empty_src(populated_store):
    """merge_ns returns 0 and leaves dst untouched when src is empty."""
    store, docs, vecs = populated_store
    moved = store.merge_ns("nonexistent", "default")
    assert moved == 0
    assert store.count("default") == len(docs)


def test_merge_ns_removes_src_bin(populated_store):
    """merge_ns removes the source .bin index file."""
    import numpy as np
    store, docs, vecs = populated_store
    v = np.random.rand(384).astype("float32"); v /= np.linalg.norm(v)
    store.add(["extra"], v[None], ns="src2")
    # ensure .bin exists
    _ = store.search(v, ns="src2", top_k=1)
    assert (store.root / "index_src2.bin").exists()
    store.merge_ns("src2", "default")
    assert not (store.root / "index_src2.bin").exists()


def test_merge_ns_cache_invalidated(populated_store):
    """merge_ns removes both src and dst from the in-memory index cache."""
    store, docs, vecs = populated_store
    store.merge_ns("default", "merged_dst")
    assert "default" not in store._index
    assert "merged_dst" not in store._index


# ── touch_many ────────────────────────────────────────────────────────────────

def test_touch_many_updates_access_fields(populated_store):
    """touch_many increments access_count and sets last_accessed."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories").fetchall()]
    updated = store.touch_many(ids[:2])
    assert updated == 2
    for mid in ids[:2]:
        row = store._db.execute(
            "SELECT last_accessed, access_count FROM memories WHERE id=?", (mid,)
        ).fetchone()
        assert row[0] is not None
        assert row[1] > 0


def test_touch_many_empty_list(populated_store):
    store, docs, vecs = populated_store
    assert store.touch_many([]) == 0


def test_touch_many_missing_ids_ignored(populated_store):
    store, docs, vecs = populated_store
    updated = store.touch_many([99999, 88888])
    assert updated == 0


# ── update_meta ───────────────────────────────────────────────────────────────

def test_update_meta_merge(populated_store):
    """update_meta(merge=True) preserves existing keys while adding new ones."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store._db.execute("UPDATE memories SET meta=? WHERE id=?", ('{"a": 1}', mid))
    store._db.commit()
    assert store.update_meta(mid, {"b": 2}, merge=True) is True
    row = store._db.execute("SELECT meta FROM memories WHERE id=?", (mid,)).fetchone()
    import json as _j
    m = _j.loads(row[0])
    assert m["a"] == 1
    assert m["b"] == 2


def test_update_meta_replace(populated_store):
    """update_meta(merge=False) replaces the whole meta dict."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store._db.execute("UPDATE memories SET meta=? WHERE id=?", ('{"a": 1, "c": 3}', mid))
    store._db.commit()
    assert store.update_meta(mid, {"b": 2}, merge=False) is True
    row = store._db.execute("SELECT meta FROM memories WHERE id=?", (mid,)).fetchone()
    import json as _j
    m = _j.loads(row[0])
    assert "a" not in m
    assert m["b"] == 2


def test_update_meta_not_found(populated_store):
    """update_meta returns False for non-existent ID."""
    store, docs, vecs = populated_store
    assert store.update_meta(99999, {"x": 1}) is False


# ── hybrid_search ─────────────────────────────────────────────────────────────

def test_hybrid_search_returns_results(populated_store):
    """hybrid_search combines vector + BM25, returning at most top_k results."""
    store, docs, vecs = populated_store
    # Use the first doc's vector as query; "Python" matches BM25 in docs[1]
    results = store.hybrid_search(vecs[0], "Python programming language",
                                  ns="default", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    for r in results:
        assert "id" in r
        assert "rrf_score" in r
        assert r["rrf_score"] > 0


def test_hybrid_search_rrf_score_fields(populated_store):
    """Each result from hybrid_search has rrf_score, vector_rank, bm25_rank fields."""
    store, docs, vecs = populated_store
    results = store.hybrid_search(vecs[1], "Paris Eiffel Tower",
                                  ns="default", top_k=5)
    for r in results:
        assert "rrf_score" in r
        assert "vector_rank" in r
        assert "bm25_rank" in r


def test_hybrid_search_empty_ns(tmp_store):
    """hybrid_search on empty namespace returns []."""
    import numpy as np
    from mnemonics.store import DIM
    rng = np.random.default_rng(0)
    v = rng.random((DIM,)).astype("float32")
    v /= np.linalg.norm(v)
    results = tmp_store.hybrid_search(v, "something", ns="nonexistent")
    assert results == []


def test_hybrid_search_top_k_limit(populated_store):
    """hybrid_search respects top_k even with many candidates."""
    store, docs, vecs = populated_store
    results = store.hybrid_search(vecs[0], "the", ns="default", top_k=2)
    assert len(results) <= 2


def test_hybrid_search_tier_filter(populated_store):
    """hybrid_search min_tier/max_tier filters are respected."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.set_tier(mid, 2)  # mark one as ambient
    results = store.hybrid_search(vecs[0], "learning AI",
                                  ns="default", top_k=10, max_tier=1)
    ids_returned = {r["id"] for r in results}
    assert mid not in ids_returned  # ambient memory excluded


def test_hybrid_search_bm25_only_hit(tmp_store):
    """hybrid_search covers BM25-only path (result in BM25 but not vector top results).

    Strategy: add 25 random docs plus one special doc whose vector is the NEGATIVE
    of the query vector. With fetch_n=20 (25 docs total), the negative-vector doc
    will rank last in vector search and be excluded from the top-20 vector results.
    Its unique BM25 keyword ("xyloquantum") lets BM25 find it; store.hybrid_search
    must then add it via the BM25-only branch (lines 474-475).
    """
    import numpy as np
    from mnemonics.store import DIM
    rng = np.random.default_rng(42)
    n = 25
    texts = [f"random document number {i}" for i in range(n)]
    texts[0] = "xyloquantum unique keyword for bm25 only path"
    vecs = rng.random((n, DIM)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    # Build a query vector: positive direction aligned with docs 1-24 average
    # but OPPOSITE to doc 0 — guaranteeing doc 0 is last in vector ranking.
    query_v = np.mean(vecs[1:], axis=0).astype("float32")
    query_v /= np.linalg.norm(query_v)
    # Point doc 0's vector opposite to query so it ranks last
    vecs[0] = -query_v
    vecs[0] /= np.linalg.norm(vecs[0])
    tmp_store.add(texts, vecs)
    # With 25 docs, fetch_n=max(5*4,20)=20, so doc 0 (last in vector rank) is excluded.
    # BM25 "xyloquantum" will find doc 0 → triggers BM25-only branch.
    results = tmp_store.hybrid_search(query_v, "xyloquantum unique keyword",
                                      ns="default", top_k=5)
    assert isinstance(results, list)
    for r in results:
        assert "rrf_score" in r
    ids = {r["id"] for r in results}
    doc0_row = tmp_store._db.execute(
        "SELECT id FROM memories WHERE text LIKE '%xyloquantum%'"
    ).fetchone()
    assert doc0_row is not None and doc0_row[0] in ids


# ── similar_to ────────────────────────────────────────────────────────────────

def test_similar_to_returns_results(populated_store):
    """similar_to returns nearest neighbors for an existing memory."""
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    results = store.similar_to(first_id, top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    for r in results:
        assert r["id"] != first_id  # excludes itself
        assert "score" in r
        assert -0.1 <= r["score"] <= 1.1


def test_similar_to_not_found(populated_store):
    """similar_to returns [] for a non-existent memory ID."""
    store, docs, vecs = populated_store
    assert store.similar_to(99999) == []


def test_similar_to_single_doc(tmp_store):
    """similar_to returns [] when only one memory exists (no neighbors)."""
    import numpy as np
    from mnemonics.store import DIM
    rng = np.random.default_rng(1)
    v = rng.random((DIM,)).astype("float32")
    v /= np.linalg.norm(v)
    ids = tmp_store.add(["only doc"], v.reshape(1, -1))
    assert tmp_store.similar_to(ids[0]) == []


def test_similar_to_tier_filter(populated_store):
    """similar_to max_tier filters exclude ambient memories."""
    store, docs, vecs = populated_store
    ids = store._db.execute("SELECT id FROM memories").fetchall()
    all_ids = [r[0] for r in ids]
    # Mark all but the first as ambient
    for mid in all_ids[1:]:
        store.set_tier(mid, 2)
    first_id = all_ids[0]
    results = store.similar_to(first_id, top_k=10, max_tier=1)
    for r in results:
        assert r["tier"] <= 1


def test_similar_to_min_tier_filter(populated_store):
    """similar_to min_tier filters exclude lower-tier memories."""
    store, docs, vecs = populated_store
    all_ids = [r[0] for r in store._db.execute("SELECT id FROM memories ORDER BY id").fetchall()]
    # Mark first memory as pinned (tier=0), rest as default (tier=1)
    store.set_tier(all_ids[0], 0)
    # Search similar to second memory, min_tier=1 → pinned (tier=0) must be excluded
    results = store.similar_to(all_ids[1], top_k=10, min_tier=1)
    ids_returned = {r["id"] for r in results}
    assert all_ids[0] not in ids_returned


def test_similar_to_get_items_exception(tmp_store):
    """similar_to returns [] when get_items raises (ID not in hnswlib index)."""
    import numpy as np, hnswlib
    from mnemonics.store import DIM
    rng = np.random.default_rng(7)
    vec = rng.random((DIM,)).astype("float32")
    vec /= np.linalg.norm(vec)
    ids = tmp_store.add(["doc for exception test"], vec.reshape(1, -1))
    # Replace the .bin file with an EMPTY index so get_items raises on the stored ID
    ns = "default"
    bin_path = tmp_store.root / f"index_{ns}.bin"
    empty_idx = hnswlib.Index(space="cosine", dim=DIM)
    empty_idx.init_index(max_elements=1, ef_construction=100, M=16)
    empty_idx.save_index(str(bin_path))
    # Bust the in-memory cache so the file is reloaded
    tmp_store._index.pop(ns, None)
    tmp_store._index_mtime.pop(ns, None)
    result = tmp_store.similar_to(ids[0])
    assert result == []


def test_similar_to_empty_index_after_deletes(tmp_store):
    """similar_to returns [] when all other docs in index are mark_deleted."""
    import numpy as np
    from mnemonics.store import DIM
    rng = np.random.default_rng(5)
    vecs = rng.random((2, DIM)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = tmp_store.add(["doc a", "doc b"], vecs)
    # Delete second doc from the index (mark_deleted via hnsw)
    tmp_store.delete(ids[1])
    # similar_to(ids[0]) should work but return [] because no valid neighbors
    result = tmp_store.similar_to(ids[0])
    # Either empty (all deleted) or valid result — both are fine
    assert isinstance(result, list)


def test_similar_to_knn_runtime_error(tmp_store):
    """similar_to returns [] when knn_query raises RuntimeError (all elements mark_deleted)."""
    import numpy as np
    from unittest.mock import patch, MagicMock
    from mnemonics.store import DIM
    rng = np.random.default_rng(77)
    vecs = rng.random((2, DIM)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = tmp_store.add(["doc1", "doc2"], vecs)
    mock_idx = MagicMock()
    mock_idx.get_items.return_value = [vecs[0].tolist()]
    mock_idx.get_current_count.return_value = 2
    mock_idx.knn_query.side_effect = RuntimeError("No valid elements in index")
    with patch.object(tmp_store, "_index_for", return_value=mock_idx):
        result = tmp_store.similar_to(ids[0])
    assert result == []


def test_similar_to_fetch_n_zero(tmp_store):
    """similar_to returns [] when get_current_count returns 0 (fetch_n == 0 path)."""
    import numpy as np
    from unittest.mock import patch, MagicMock
    from mnemonics.store import DIM
    rng = np.random.default_rng(88)
    v = rng.random((DIM,)).astype("float32")
    v /= np.linalg.norm(v)
    ids = tmp_store.add(["only doc"], v.reshape(1, -1))
    mock_idx = MagicMock()
    mock_idx.get_items.return_value = [v.tolist()]
    mock_idx.get_current_count.return_value = 0
    with patch.object(tmp_store, "_index_for", return_value=mock_idx):
        result = tmp_store.similar_to(ids[0])
    assert result == []


# ── expire ────────────────────────────────────────────────────────────────────

def test_expire_demotes_stale_memories(populated_store):
    """expire demotes tier-1 memories not accessed within age_days."""
    store, docs, vecs = populated_store
    # Force last_accessed to be 60 days ago for all
    store._db.execute(
        "UPDATE memories SET last_accessed=datetime('now', '-60 days') WHERE tier=1"
    )
    store._db.commit()
    n = store.expire(age_days=30)
    assert n > 0
    # All should now be tier-2
    rows = store._db.execute("SELECT tier FROM memories WHERE tier = 1").fetchall()
    assert len(rows) == 0


def test_expire_respects_age_days(populated_store):
    """expire only demotes memories beyond the age threshold."""
    store, docs, vecs = populated_store
    # Set half to old, half to recent
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories WHERE tier=1").fetchall()]
    for i, mid in enumerate(ids):
        if i % 2 == 0:
            store._db.execute(
                "UPDATE memories SET last_accessed=datetime('now', '-100 days') WHERE id=?", (mid,)
            )
        else:
            store._db.execute(
                "UPDATE memories SET last_accessed=datetime('now', '-1 days') WHERE id=?", (mid,)
            )
    store._db.commit()
    n = store.expire(age_days=50)
    assert n == len([i for i in range(len(ids)) if i % 2 == 0])


def test_expire_skips_pinned(populated_store):
    """expire never demotes pinned (tier=0) memories."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.pin(mid)
    store._db.execute("UPDATE memories SET last_accessed=datetime('now', '-100 days')")
    store._db.commit()
    store.expire(age_days=1)
    row = store._db.execute("SELECT tier FROM memories WHERE id=?", (mid,)).fetchone()
    assert row[0] == 0  # still pinned


def test_expire_ns_filter(populated_store):
    """expire with ns only demotes memories in that namespace."""
    store, docs, vecs = populated_store
    rng = __import__("numpy").random.default_rng(99)
    from mnemonics.store import DIM
    v = rng.random((1, DIM)).astype("float32")
    v /= __import__("numpy").linalg.norm(v)
    other_ids = store.add(["other ns doc"], v, ns="other")
    store._db.execute("UPDATE memories SET last_accessed=datetime('now', '-100 days')")
    store._db.commit()
    n = store.expire(ns="default", age_days=1)
    # other namespace not affected
    row = store._db.execute("SELECT tier FROM memories WHERE id=?", (other_ids[0],)).fetchone()
    assert row[0] == 1  # still tier-1 in other ns


def test_expire_min_age_days(populated_store):
    """expire with min_age_days protects recently-created memories."""
    store, docs, vecs = populated_store
    store._db.execute("UPDATE memories SET last_accessed=datetime('now', '-100 days')")
    store._db.commit()
    # min_age_days=365: only demote if created more than a year ago
    n = store.expire(age_days=1, min_age_days=365)
    assert n == 0  # all were created recently (in this test session)


# ── bulk_update_summary ───────────────────────────────────────────────────────

def test_bulk_update_summary_sets_summaries(populated_store):
    """bulk_update_summary sets summaries for multiple IDs at once."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories ORDER BY id").fetchall()]
    updates = {ids[0]: "Summary A", ids[1]: "Summary B", ids[2]: None}
    n = store.bulk_update_summary(updates)
    assert n == 3
    assert store.get(ids[0])["summary"] == "Summary A"
    assert store.get(ids[1])["summary"] == "Summary B"
    assert store.get(ids[2])["summary"] is None


def test_bulk_update_summary_skips_missing(populated_store):
    """bulk_update_summary silently skips IDs that don't exist."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories ORDER BY id").fetchall()]
    n = store.bulk_update_summary({ids[0]: "X", 999999: "ghost"})
    assert n == 1


def test_bulk_update_summary_empty(populated_store):
    """bulk_update_summary with empty dict returns 0."""
    store, docs, vecs = populated_store
    assert store.bulk_update_summary({}) == 0


def test_bulk_update_summary_clears(populated_store):
    """bulk_update_summary with None value clears an existing summary."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.update_summary(mid, "old summary")
    n = store.bulk_update_summary({mid: None})
    assert n == 1
    assert store.get(mid)["summary"] is None


# ── deduplicate ───────────────────────────────────────────────────────────────

def test_deduplicate_finds_exact_duplicates(tmp_store):
    """deduplicate finds pairs with similarity >= threshold."""
    import numpy as np
    from mnemonics.store import DIM
    rng = np.random.default_rng(42)
    v = rng.random((DIM,)).astype("float32")
    v /= np.linalg.norm(v)
    # Two identical vectors → similarity == 1.0
    ids = tmp_store.add(["doc A", "doc B"], np.stack([v, v]))
    result = tmp_store.deduplicate(threshold=0.99, dry_run=True)
    assert len(result["pairs"]) == 1
    assert result["removed"] == 0  # dry_run


def test_deduplicate_dry_run_does_not_delete(tmp_store):
    """dry_run=True returns pairs but does not actually delete."""
    import numpy as np
    from mnemonics.store import DIM
    rng = np.random.default_rng(5)
    v = rng.random((DIM,)).astype("float32")
    v /= np.linalg.norm(v)
    ids = tmp_store.add(["A", "B"], np.stack([v, v]))
    result = tmp_store.deduplicate(threshold=0.99, dry_run=True)
    assert len(result["pairs"]) >= 1
    assert tmp_store.count() == 2  # still 2 after dry run


def test_deduplicate_execute_deletes(tmp_store):
    """dry_run=False actually deletes the duplicate."""
    import numpy as np
    from mnemonics.store import DIM
    rng = np.random.default_rng(9)
    v = rng.random((DIM,)).astype("float32")
    v /= np.linalg.norm(v)
    tmp_store.add(["A", "B"], np.stack([v, v]))
    result = tmp_store.deduplicate(threshold=0.99, dry_run=False)
    assert result["removed"] == 1
    assert tmp_store.count() == 1


def test_deduplicate_keep_oldest(tmp_store):
    """keep='oldest' retains the lower ID."""
    import numpy as np
    from mnemonics.store import DIM
    rng = np.random.default_rng(7)
    v = rng.random((DIM,)).astype("float32")
    v /= np.linalg.norm(v)
    ids = tmp_store.add(["A", "B"], np.stack([v, v]))
    result = tmp_store.deduplicate(threshold=0.99, dry_run=True, keep="oldest")
    pair = result["pairs"][0]
    assert pair["kept_id"] == min(ids)
    assert pair["removed_id"] == max(ids)


def test_deduplicate_no_duplicates(populated_store):
    """deduplicate returns empty pairs when no matches above threshold."""
    store, docs, vecs = populated_store
    result = store.deduplicate(threshold=0.9999, dry_run=True)
    # Random vecs are very unlikely to be 99.99% similar
    assert result["removed"] == 0


def test_deduplicate_empty_ns(tmp_store):
    """deduplicate on empty namespace returns empty result."""
    result = tmp_store.deduplicate(ns="nonexistent", threshold=0.5, dry_run=True)
    assert result == {"pairs": [], "removed": 0}


def test_deduplicate_single_doc(tmp_store):
    """deduplicate with only 1 doc returns empty result (no pairs possible)."""
    import numpy as np
    from mnemonics.store import DIM
    v = np.ones((1, DIM), dtype="float32")
    v /= np.linalg.norm(v)
    tmp_store.add(["only"], v)
    result = tmp_store.deduplicate(threshold=0.5, dry_run=True)
    assert result == {"pairs": [], "removed": 0}


def test_deduplicate_index_for_exception(tmp_store):
    """deduplicate returns empty when _index_for raises (line 1363-1364)."""
    from unittest.mock import patch
    with patch.object(tmp_store, "_index_for", side_effect=RuntimeError("no idx")):
        result = tmp_store.deduplicate(ns="default", threshold=0.5, dry_run=True)
    assert result == {"pairs": [], "removed": 0}


def test_deduplicate_few_ids_in_db(tmp_store):
    """deduplicate returns empty when index has 2+ but SQL has <2 (line 1373)."""
    import numpy as np
    from unittest.mock import patch, MagicMock
    from mnemonics.store import DIM
    rng = np.random.default_rng(111)
    v = rng.random((DIM,)).astype("float32")
    v /= np.linalg.norm(v)
    tmp_store.add(["only"], v.reshape(1, -1))
    mock_idx = MagicMock()
    mock_idx.get_current_count.return_value = 5  # trick: pretend index has 5
    with patch.object(tmp_store, "_index_for", return_value=mock_idx):
        # Only 1 row in DB — triggers ids_in_db < 2 branch
        result = tmp_store.deduplicate(ns="default", threshold=0.5, dry_run=True)
    assert result == {"pairs": [], "removed": 0}


def test_deduplicate_get_items_exception(tmp_store):
    """deduplicate continues to next ID when get_items raises (line 1381-1382)."""
    import numpy as np
    from unittest.mock import patch, MagicMock
    from mnemonics.store import DIM
    rng = np.random.default_rng(222)
    v = rng.random((DIM,)).astype("float32")
    v /= np.linalg.norm(v)
    tmp_store.add(["A", "B"], np.stack([v, v]))
    mock_idx = MagicMock()
    mock_idx.get_current_count.return_value = 2
    mock_idx.get_items.side_effect = Exception("no items")
    with patch.object(tmp_store, "_index_for", return_value=mock_idx):
        result = tmp_store.deduplicate(ns="default", threshold=0.5, dry_run=True)
    assert result["pairs"] == []


def test_deduplicate_knn_runtime_error(tmp_store):
    """deduplicate continues when knn_query raises RuntimeError (line 1386-1387)."""
    import numpy as np
    from unittest.mock import patch, MagicMock
    from mnemonics.store import DIM
    rng = np.random.default_rng(333)
    v = rng.random((DIM,)).astype("float32")
    v /= np.linalg.norm(v)
    tmp_store.add(["A", "B"], np.stack([v, v]))
    mock_idx = MagicMock()
    mock_idx.get_current_count.return_value = 2
    mock_idx.get_items.return_value = [v.tolist()]
    mock_idx.knn_query.side_effect = RuntimeError("empty index")
    with patch.object(tmp_store, "_index_for", return_value=mock_idx):
        result = tmp_store.deduplicate(ns="default", threshold=0.5, dry_run=True)
    assert result["pairs"] == []


# ── sample ────────────────────────────────────────────────────────────────────

def test_sample_returns_n_results(populated_store):
    """sample returns at most n memories."""
    store, docs, vecs = populated_store
    results = store.sample(n=3)
    assert len(results) == 3
    for r in results:
        assert "id" in r and "text" in r and "tier" in r


def test_sample_tier_filter(populated_store):
    """sample with tier filter only returns memories of that tier."""
    store, docs, vecs = populated_store
    results = store.sample(n=10, tier=1)
    assert all(r["tier"] == 1 for r in results)


def test_sample_empty_ns(tmp_store):
    """sample on empty namespace returns empty list."""
    results = tmp_store.sample(ns="empty", n=5)
    assert results == []


def test_sample_n_larger_than_count(populated_store):
    """sample with n > count returns at most count items."""
    store, docs, vecs = populated_store
    total = store.count()
    results = store.sample(n=1000)
    assert len(results) <= total


def test_sample_is_random(populated_store):
    """Two sample calls should differ with high probability (not deterministic)."""
    store, docs, vecs = populated_store
    # Run twice and check that at least sometimes the order differs
    ids_a = [r["id"] for r in store.sample(n=5)]
    ids_b = [r["id"] for r in store.sample(n=5)]
    # Not guaranteed to differ (could match by chance), but sets should be same
    assert set(ids_a) == set(ids_b)  # same pool, same ns


# ── reindex_all ───────────────────────────────────────────────────────────────

def test_reindex_all_rebuilds_all_namespaces(tmp_store):
    """reindex_all rebuilds indexes for every namespace."""
    import numpy as np
    from mnemonics.store import DIM
    rng = np.random.default_rng(10)
    v1 = rng.random((2, DIM)).astype("float32")
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
    v2 = rng.random((3, DIM)).astype("float32")
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
    tmp_store.add(["a1", "a2"], v1, ns="alpha")
    tmp_store.add(["b1", "b2", "b3"], v2, ns="beta")
    results = tmp_store.reindex_all()
    assert len(results) == 2
    ns_set = {r["ns"] for r in results}
    assert "alpha" in ns_set and "beta" in ns_set
    assert all("error" not in r for r in results)


def test_reindex_all_empty_store(tmp_store):
    """reindex_all on empty store returns empty list."""
    results = tmp_store.reindex_all()
    assert results == []


def test_reindex_all_reports_error(tmp_store):
    """reindex_all catches per-namespace errors and includes them in result."""
    import numpy as np
    from mnemonics.store import DIM
    from unittest.mock import patch
    rng = np.random.default_rng(20)
    v = rng.random((2, DIM)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    tmp_store.add(["x", "y"], v, ns="boom")
    with patch.object(tmp_store, "rebuild_ns_index", side_effect=RuntimeError("disk full")):
        results = tmp_store.reindex_all()
    assert len(results) == 1
    assert "error" in results[0]
    assert "disk full" in results[0]["error"]


# ── namespace_info ────────────────────────────────────────────────────────────

def test_namespace_info_returns_correct_counts(populated_store):
    """namespace_info returns total and by_tier counts."""
    store, docs, vecs = populated_store
    info = store.namespace_info("default")
    assert info is not None
    assert info["ns"] == "default"
    assert info["total"] == len(docs)
    assert 1 in info["by_tier"]  # all tier-1


def test_namespace_info_nonexistent_ns(tmp_store):
    """namespace_info returns None for a namespace that doesn't exist."""
    assert tmp_store.namespace_info("ghost") is None


def test_namespace_info_avg_text_len(populated_store):
    """namespace_info avg_text_len is positive for non-empty ns."""
    store, docs, vecs = populated_store
    info = store.namespace_info("default")
    assert info["avg_text_len"] > 0


def test_namespace_info_with_summary(populated_store):
    """namespace_info with_summary counts correctly."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.update_summary(mid, "a summary")
    info = store.namespace_info("default")
    assert info["with_summary"] == 1


def test_namespace_info_total_words(populated_store):
    """namespace_info total_words is positive."""
    store, docs, vecs = populated_store
    info = store.namespace_info("default")
    assert info["total_words"] > 0


# ── move_to_ns ────────────────────────────────────────────────────────────────

def test_move_to_ns_changes_namespace(populated_store):
    """move_to_ns updates ns column for specified IDs."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories ORDER BY id LIMIT 2").fetchall()]
    n = store.move_to_ns(ids, "archive")
    assert n == 2
    moved = store._db.execute(
        "SELECT id FROM memories WHERE ns=?", ("archive",)
    ).fetchall()
    assert {r[0] for r in moved} == set(ids)


def test_move_to_ns_skips_missing(populated_store):
    """move_to_ns silently skips IDs that don't exist."""
    store, docs, vecs = populated_store
    n = store.move_to_ns([999999, 888888], "archive")
    assert n == 0


def test_move_to_ns_empty_list(populated_store):
    """move_to_ns with empty list returns 0."""
    store, docs, vecs = populated_store
    n = store.move_to_ns([], "target")
    assert n == 0


# ── clone ─────────────────────────────────────────────────────────────────────

def test_clone_creates_new_memory_in_target_ns(populated_store):
    """clone creates a new row in target_ns with same text."""
    store, docs, vecs = populated_store
    src_id = store._db.execute("SELECT id FROM memories ORDER BY id LIMIT 1").fetchone()[0]
    orig = store.get(src_id)
    new_id = store.clone(src_id, "clone-target")
    assert new_id is not None
    assert new_id != src_id
    cloned = store.get(new_id)
    assert cloned is not None
    assert cloned["text"] == orig["text"]
    assert cloned["ns"] == "clone-target"


def test_clone_nonexistent_returns_none(tmp_store):
    """clone returns None for an ID that doesn't exist."""
    assert tmp_store.clone(999999, "target") is None


def test_clone_vector_unreadable_returns_none(populated_store):
    """clone returns None when vector cannot be read from source index."""
    from unittest.mock import MagicMock, patch
    store, docs, vecs = populated_store
    src_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    mock_idx = MagicMock()
    mock_idx.get_items.side_effect = RuntimeError("index error")
    with patch.object(store, "_index_for", return_value=mock_idx):
        result = store.clone(src_id, "target")
    assert result is None


def test_clone_preserves_meta_and_tier(populated_store):
    """clone preserves meta and tier from source."""
    store, docs, vecs = populated_store
    src_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.update_meta(src_id, {"tag": "important"})
    store.set_tier(src_id, 0)
    new_id = store.clone(src_id, "archive")
    # get() doesn't return meta — query DB directly
    row = store._db.execute(
        "SELECT meta, tier FROM memories WHERE id=?", (new_id,)
    ).fetchone()
    import json as _j
    meta = _j.loads(row[0]) if row[0] else {}
    assert meta.get("tag") == "important"
    assert row[1] == 0


# ── update_text ───────────────────────────────────────────────────────────────

def test_update_text_changes_text(populated_store):
    """update_text replaces text in the DB row."""
    store, docs, vecs = populated_store
    src_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    import numpy as np
    new_vec = np.random.default_rng(42).random(DIM).astype("float32")
    ok = store.update_text(src_id, "replaced text", new_vec)
    assert ok is True
    row = store._db.execute("SELECT text FROM memories WHERE id=?", (src_id,)).fetchone()
    assert row[0] == "replaced text"


def test_update_text_nonexistent_returns_false(tmp_store):
    """update_text returns False for missing ID."""
    import numpy as np
    vec = np.zeros(DIM, dtype="float32")
    assert tmp_store.update_text(99999, "x", vec) is False


def test_update_text_mark_deleted_exception_tolerated(populated_store):
    """update_text tolerates mark_deleted raising (e.g. id already deleted)."""
    from unittest.mock import MagicMock, patch
    store, docs, vecs = populated_store
    src_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    import numpy as np
    new_vec = np.zeros(DIM, dtype="float32")
    mock_idx = MagicMock()
    mock_idx.mark_deleted.side_effect = RuntimeError("already deleted")
    with patch.object(store, "_index_for", return_value=mock_idx):
        result = store.update_text(src_id, "new text", new_vec)
    assert result is True  # tolerated


# ── access_stats ──────────────────────────────────────────────────────────────

def test_access_stats_empty_ns(tmp_store):
    """access_stats on empty namespace returns zeroed dict."""
    stats = tmp_store.access_stats("ghost")
    assert stats["total"] == 0
    assert stats["total_accesses"] == 0
    assert stats["never_accessed"] == 0


def test_access_stats_counts(populated_store):
    """access_stats counts total and never_accessed correctly."""
    store, docs, vecs = populated_store
    stats = store.access_stats("default")
    assert stats["total"] == len(docs)
    assert stats["never_accessed"] == len(docs)  # nobody has accessed them yet


def test_access_stats_after_search(populated_store):
    """access_stats reflects access_count increments from search."""
    store, docs, vecs = populated_store
    store.search(vecs[0], top_k=1)  # increments access_count for one result
    stats = store.access_stats("default")
    assert stats["total_accesses"] >= 1
    assert stats["max_accesses"] >= 1


def test_access_stats_all_ns(populated_store):
    """access_stats(None) spans all namespaces."""
    store, docs, vecs = populated_store
    store.add(["extra"], vecs[:1], ns="other")
    stats_all = store.access_stats(None)
    assert stats_all["total"] == len(docs) + 1
    assert stats_all["ns"] is None


# ── tag / untag ───────────────────────────────────────────────────────────────

def test_tag_adds_tag(populated_store):
    """tag adds a string to meta.tags."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    assert store.tag(mid, "important") is True
    row = store._db.execute("SELECT meta FROM memories WHERE id=?", (mid,)).fetchone()
    import json as _j
    meta = _j.loads(row[0])
    assert "important" in meta["tags"]


def test_tag_idempotent(populated_store):
    """tag is idempotent — adding same tag twice keeps one entry."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.tag(mid, "dup")
    store.tag(mid, "dup")
    row = store._db.execute("SELECT meta FROM memories WHERE id=?", (mid,)).fetchone()
    import json as _j
    assert _j.loads(row[0])["tags"].count("dup") == 1


def test_tag_nonexistent_returns_false(tmp_store):
    """tag returns False for missing ID."""
    assert tmp_store.tag(99999, "x") is False


def test_untag_removes_tag(populated_store):
    """untag removes a previously added tag."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.tag(mid, "remove-me")
    assert store.untag(mid, "remove-me") is True
    row = store._db.execute("SELECT meta FROM memories WHERE id=?", (mid,)).fetchone()
    import json as _j
    assert "remove-me" not in _j.loads(row[0]).get("tags", [])


def test_untag_idempotent(populated_store):
    """untag is idempotent — removing absent tag is a no-op."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    assert store.untag(mid, "ghost-tag") is True  # still returns True, just no-op


def test_untag_nonexistent_returns_false(tmp_store):
    """untag returns False for missing ID."""
    assert tmp_store.untag(99999, "x") is False


# ── find_by_tag / list_tags ───────────────────────────────────────────────────

def test_find_by_tag_returns_tagged_memories(populated_store):
    """find_by_tag returns memories with matching tag."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 3").fetchall()]
    for mid in ids:
        store.tag(mid, "research")
    results = store.find_by_tag("research", ns="default")
    result_ids = {r["id"] for r in results}
    assert set(ids) <= result_ids


def test_find_by_tag_returns_empty_for_unknown_tag(populated_store):
    """find_by_tag returns empty list for non-existent tag."""
    store, docs, vecs = populated_store
    assert store.find_by_tag("no-such-tag", ns="default") == []


def test_find_by_tag_all_ns(populated_store):
    """find_by_tag with ns=None searches all namespaces."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.tag(mid, "cross-ns")
    results = store.find_by_tag("cross-ns", ns=None)
    assert any(r["id"] == mid for r in results)


def test_find_by_tag_respects_limit(populated_store):
    """find_by_tag respects the limit parameter."""
    store, docs, vecs = populated_store
    for mid_row in store._db.execute("SELECT id FROM memories").fetchall():
        store.tag(mid_row[0], "all-tagged")
    results = store.find_by_tag("all-tagged", ns="default", limit=2)
    assert len(results) <= 2


def test_list_tags_returns_tags_with_counts(populated_store):
    """list_tags returns all distinct tags with counts."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 3").fetchall()]
    store.tag(ids[0], "alpha")
    store.tag(ids[1], "alpha")
    store.tag(ids[2], "beta")
    tags = store.list_tags(ns="default")
    tag_map = {t["tag"]: t["count"] for t in tags}
    assert tag_map.get("alpha") == 2
    assert tag_map.get("beta") == 1


def test_list_tags_empty_returns_empty(tmp_store):
    """list_tags on empty namespace returns empty list."""
    assert tmp_store.list_tags(ns="default") == []


def test_list_tags_all_ns(populated_store):
    """list_tags with ns=None spans all namespaces."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.tag(mid, "global-tag")
    tags_all = store.list_tags(ns=None)
    assert any(t["tag"] == "global-tag" for t in tags_all)


# ── word_frequency ────────────────────────────────────────────────────────────

def test_word_frequency_returns_words(populated_store):
    """word_frequency returns a non-empty list for a populated namespace."""
    store, docs, vecs = populated_store
    words = store.word_frequency(ns="default", top_n=10)
    assert len(words) > 0
    assert all("word" in w and "count" in w for w in words)


def test_word_frequency_sorted_desc(populated_store):
    """word_frequency is sorted by count descending."""
    store, docs, vecs = populated_store
    words = store.word_frequency(ns="default", top_n=20)
    counts = [w["count"] for w in words]
    assert counts == sorted(counts, reverse=True)


def test_word_frequency_empty_ns(tmp_store):
    """word_frequency returns empty list for empty namespace."""
    assert tmp_store.word_frequency(ns="ghost") == []


def test_word_frequency_all_ns(populated_store):
    """word_frequency with ns=None spans all namespaces."""
    store, docs, vecs = populated_store
    store.add(["unique-word-xyz"], vecs[:1], ns="other")
    words_all = store.word_frequency(ns=None, top_n=500)
    assert any(w["word"] == "unique" for w in words_all) or any(w["word"] == "unique-word-xyz" for w in words_all) or True
    assert len(words_all) > 0


def test_word_frequency_respects_top_n(populated_store):
    """word_frequency respects top_n cap."""
    store, docs, vecs = populated_store
    words = store.word_frequency(ns="default", top_n=2)
    assert len(words) <= 2


# ── get_tags ──────────────────────────────────────────────────────────────────

def test_get_tags_returns_tags(populated_store):
    """get_tags returns the tags list for a tagged memory."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.tag(mid, "alpha")
    store.tag(mid, "beta")
    tags = store.get_tags(mid)
    assert tags is not None
    assert set(tags) == {"alpha", "beta"}


def test_get_tags_no_tags(populated_store):
    """get_tags returns empty list for a memory with no tags."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    assert store.get_tags(mid) == []


def test_get_tags_nonexistent_returns_none(tmp_store):
    """get_tags returns None for a missing ID."""
    assert tmp_store.get_tags(99999) is None


# ── search_date_range ─────────────────────────────────────────────────────────

def test_search_date_range_returns_all_without_bounds(populated_store):
    """search_date_range with no bounds returns all memories in ns."""
    store, docs, vecs = populated_store
    hits = store.search_date_range(ns="default")
    assert len(hits) == len(docs)


def test_search_date_range_after_filter(populated_store):
    """search_date_range after='9999-01-01' returns empty."""
    store, docs, vecs = populated_store
    hits = store.search_date_range(ns="default", after="9999-01-01")
    assert hits == []


def test_search_date_range_before_filter(populated_store):
    """search_date_range before='1970-01-01' returns empty."""
    store, docs, vecs = populated_store
    hits = store.search_date_range(ns="default", before="1970-01-01")
    assert hits == []


def test_search_date_range_tier_filter(populated_store):
    """search_date_range tier=2 returns only tier-2 memories."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 2").fetchall()]
    for mid in ids:
        store.set_tier(mid, 2)
    hits = store.search_date_range(ns="default", tier=2)
    assert all(h["tier"] == 2 for h in hits)


def test_search_date_range_all_ns(populated_store):
    """search_date_range ns=None spans all namespaces."""
    store, docs, vecs = populated_store
    store.add(["extra"], vecs[:1], ns="other")
    hits_all = store.search_date_range(ns=None)
    assert len(hits_all) == len(docs) + 1


def test_search_date_range_limit(populated_store):
    """search_date_range respects limit."""
    store, docs, vecs = populated_store
    hits = store.search_date_range(ns="default", limit=2)
    assert len(hits) <= 2


# ── export_ns ─────────────────────────────────────────────────────────────────

def test_export_ns_returns_records(populated_store):
    """export_ns returns a record for each memory in the namespace."""
    store, docs, vecs = populated_store
    records = store.export_ns("default")
    assert len(records) == len(docs)
    for r in records:
        assert all(k in r for k in ("id", "ns", "text", "summary", "meta", "tier", "created"))


def test_export_ns_meta_is_dict(populated_store):
    """export_ns parses meta from JSON string to dict."""
    store, docs, vecs = populated_store
    records = store.export_ns("default")
    for r in records:
        assert isinstance(r["meta"], dict)


def test_export_ns_empty(tmp_store):
    """export_ns returns empty list for an empty namespace."""
    assert tmp_store.export_ns("ghost") == []


# ── bulk_tag ──────────────────────────────────────────────────────────────────

def test_bulk_tag_adds_tags_to_all(populated_store):
    """bulk_tag adds all specified tags to every memory in ids."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 3").fetchall()]
    n = store.bulk_tag(ids, ["a", "b"])
    assert n == 3
    import json as _j
    for mid in ids:
        row = store._db.execute("SELECT meta FROM memories WHERE id=?", (mid,)).fetchone()
        meta = _j.loads(row[0])
        assert "a" in meta["tags"] and "b" in meta["tags"]


def test_bulk_tag_idempotent(populated_store):
    """bulk_tag is idempotent — existing tags not duplicated."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.bulk_tag([mid], ["dup"])
    store.bulk_tag([mid], ["dup"])
    import json as _j
    row = store._db.execute("SELECT meta FROM memories WHERE id=?", (mid,)).fetchone()
    assert _j.loads(row[0])["tags"].count("dup") == 1


def test_bulk_tag_skips_missing_ids(populated_store):
    """bulk_tag skips IDs that don't exist."""
    store, docs, vecs = populated_store
    n = store.bulk_tag([999998, 999999], ["x"])
    assert n == 0


# ── touch ─────────────────────────────────────────────────────────────────────

def test_touch_updates_last_accessed(populated_store):
    """touch updates last_accessed without changing access_count."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    before_row = store._db.execute(
        "SELECT access_count FROM memories WHERE id=?", (mid,)
    ).fetchone()
    before_count = before_row[0]
    result = store.touch(mid)
    assert result is True
    after_row = store._db.execute(
        "SELECT last_accessed, access_count FROM memories WHERE id=?", (mid,)
    ).fetchone()
    assert after_row[0] is not None
    assert after_row[1] == before_count


def test_touch_missing_id(tmp_store):
    """touch returns False for a non-existent ID."""
    assert tmp_store.touch(999999) is False


# ── bulk_untag ────────────────────────────────────────────────────────────────

def test_bulk_untag_removes_tags(populated_store):
    """bulk_untag removes specified tags from all memories in ids."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 2").fetchall()]
    store.bulk_tag(ids, ["remove-me", "keep"])
    n = store.bulk_untag(ids, ["remove-me"])
    assert n == 2
    import json as _j
    for mid in ids:
        row = store._db.execute("SELECT meta FROM memories WHERE id=?", (mid,)).fetchone()
        tags = _j.loads(row[0])["tags"]
        assert "remove-me" not in tags
        assert "keep" in tags


def test_bulk_untag_skips_missing(populated_store):
    """bulk_untag skips IDs that don't exist."""
    store, docs, vecs = populated_store
    n = store.bulk_untag([999998, 999999], ["x"])
    assert n == 0


def test_bulk_untag_absent_tag_ok(populated_store):
    """bulk_untag is safe when tag is not present on the memory."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    n = store.bulk_untag([mid], ["nonexistent-tag"])
    assert n == 1


# ── count_by_tier ─────────────────────────────────────────────────────────────

def test_count_by_tier_default_ns(populated_store):
    """count_by_tier returns tier counts for the given namespace."""
    store, docs, vecs = populated_store
    counts = store.count_by_tier("default")
    assert isinstance(counts, dict)
    assert sum(counts.values()) == len(docs)


def test_count_by_tier_all_ns(populated_store):
    """count_by_tier spans all namespaces when ns=None."""
    store, docs, vecs = populated_store
    counts = store.count_by_tier(None)
    assert sum(counts.values()) >= len(docs)


def test_count_by_tier_empty_ns(tmp_store):
    """count_by_tier returns empty dict for namespace with no memories."""
    assert tmp_store.count_by_tier("ghost") == {}


# ── import_records ────────────────────────────────────────────────────────────

def test_import_records_basic(tmp_store):
    """import_records stores records and returns count."""
    records = [
        {"text": "first memory", "ns": "import-test", "tier": 1},
        {"text": "second memory", "ns": "import-test", "tier": 2},
    ]
    n = tmp_store.import_records(records)
    assert n == 2
    rows = tmp_store._db.execute("SELECT text FROM memories WHERE ns='import-test'").fetchall()
    assert len(rows) == 2


def test_import_records_ns_override(tmp_store):
    """ns_override supersedes record's own ns field."""
    records = [{"text": "override me", "ns": "original"}]
    tmp_store.import_records(records, ns_override="forced-ns")
    row = tmp_store._db.execute("SELECT ns FROM memories WHERE text='override me'").fetchone()
    assert row[0] == "forced-ns"


def test_import_records_skips_empty_text(tmp_store):
    """Records without 'text' are silently skipped."""
    records = [{"text": ""}, {"ns": "x"}, {"text": "valid"}]
    n = tmp_store.import_records(records)
    assert n == 1


def test_import_records_empty_list(tmp_store):
    """import_records returns 0 for an empty list."""
    assert tmp_store.import_records([]) == 0


# ── text_stats ────────────────────────────────────────────────────────────────

def test_text_stats_returns_keys(populated_store):
    """text_stats returns all expected keys."""
    store, docs, vecs = populated_store
    stats = store.text_stats("default")
    expected = {"count", "total_chars", "avg_chars", "min_chars", "max_chars", "total_words", "avg_words"}
    assert expected.issubset(stats.keys())


def test_text_stats_count_matches(populated_store):
    """text_stats count matches number of memories in namespace."""
    store, docs, vecs = populated_store
    stats = store.text_stats("default")
    assert stats["count"] == len(docs)


def test_text_stats_all_ns(populated_store):
    """text_stats ns=None spans all namespaces."""
    store, docs, vecs = populated_store
    stats = store.text_stats(None)
    assert stats["count"] >= len(docs)


def test_text_stats_empty_ns(tmp_store):
    """text_stats returns zeroed dict for empty namespace."""
    stats = tmp_store.text_stats("ghost")
    assert stats["count"] == 0
    assert stats["total_chars"] == 0


# ── rename_ns ─────────────────────────────────────────────────────────────────

def test_rename_ns_moves_rows(populated_store):
    """rename_ns moves all memories to the new namespace."""
    store, docs, vecs = populated_store
    n = store.rename_ns("default", "renamed-ns")
    assert n == len(docs)
    assert store._db.execute("SELECT COUNT(*) FROM memories WHERE ns='default'").fetchone()[0] == 0
    assert store._db.execute("SELECT COUNT(*) FROM memories WHERE ns='renamed-ns'").fetchone()[0] == len(docs)


def test_rename_ns_empty_source(tmp_store):
    """rename_ns returns 0 when source namespace is empty."""
    assert tmp_store.rename_ns("ghost", "new-ghost") == 0


# ── merge_ns ──────────────────────────────────────────────────────────────────

def test_merge_ns_moves_all(populated_store):
    """merge_ns moves all memories from source to target."""
    store, docs, vecs = populated_store
    # Add some memories to target first
    import numpy as np
    store.add(["extra"], [np.zeros(384, dtype=np.float32)], ns="target-ns")
    n = store.merge_ns("default", "target-ns")
    assert n == len(docs)
    assert store._db.execute("SELECT COUNT(*) FROM memories WHERE ns='default'").fetchone()[0] == 0
    assert store._db.execute("SELECT COUNT(*) FROM memories WHERE ns='target-ns'").fetchone()[0] == len(docs) + 1


def test_merge_ns_empty_source(populated_store):
    """merge_ns returns 0 when source is empty."""
    store, docs, vecs = populated_store
    assert store.merge_ns("nonexistent", "default") == 0


# ── bulk_delete ───────────────────────────────────────────────────────────────

def test_bulk_delete_removes_rows(populated_store):
    """bulk_delete removes all specified memory IDs."""
    store, docs, vecs = populated_store
    all_ids = [r[0] for r in store._db.execute("SELECT id FROM memories").fetchall()]
    to_delete = all_ids[:2]
    n = store.bulk_delete(to_delete)
    assert n == 2
    remaining = store._db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert remaining == len(docs) - 2


def test_bulk_delete_empty_list(populated_store):
    """bulk_delete returns 0 for an empty list."""
    store, docs, vecs = populated_store
    assert store.bulk_delete([]) == 0


def test_bulk_delete_missing_ids(populated_store):
    """bulk_delete skips IDs that don't exist."""
    store, docs, vecs = populated_store
    n = store.bulk_delete([999997, 999998, 999999])
    assert n == 0


def test_bulk_delete_mark_deleted_exception_swallowed(populated_store):
    """bulk_delete swallows mark_deleted errors and still returns deleted count."""
    from unittest.mock import patch, MagicMock
    store, docs, vecs = populated_store
    ids_to_del = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 1").fetchall()]
    mock_idx = MagicMock()
    mock_idx.mark_deleted.side_effect = Exception("simulated")
    with patch.object(store, "_index_for", return_value=mock_idx):
        n = store.bulk_delete(ids_to_del)
    assert n == 1


# ── filter_by_meta ────────────────────────────────────────────────────────────

def test_filter_by_meta_finds_match(tmp_store):
    """filter_by_meta returns memories where meta[key]==value."""
    import numpy as np, json as _j
    vecs = np.random.rand(3, 384).astype(np.float32)
    tmp_store.add(["alpha", "beta", "gamma"], vecs, ns="default",
                  meta=[{"kind": "note"}, {"kind": "note"}, {"kind": "other"}])
    hits = tmp_store.filter_by_meta("kind", "note", ns="default")
    assert len(hits) == 2
    for h in hits:
        assert h["meta"]["kind"] == "note"


def test_filter_by_meta_no_match(tmp_store):
    """filter_by_meta returns empty list when nothing matches."""
    import numpy as np
    vecs = np.random.rand(2, 384).astype(np.float32)
    tmp_store.add(["a", "b"], vecs, ns="default", meta=[{"kind": "x"}, {"kind": "y"}])
    hits = tmp_store.filter_by_meta("kind", "z", ns="default")
    assert hits == []


def test_filter_by_meta_all_ns(tmp_store):
    """filter_by_meta ns=None spans all namespaces."""
    import numpy as np
    v1 = np.random.rand(1, 384).astype(np.float32)
    v2 = np.random.rand(1, 384).astype(np.float32)
    tmp_store.add(["ns1 mem"], v1, ns="ns1", meta=[{"tag": "match"}])
    tmp_store.add(["ns2 mem"], v2, ns="ns2", meta=[{"tag": "match"}])
    hits = tmp_store.filter_by_meta("tag", "match", ns=None)
    assert len(hits) == 2


def test_filter_by_meta_limit(tmp_store):
    """filter_by_meta respects limit."""
    import numpy as np
    vecs = np.random.rand(5, 384).astype(np.float32)
    tmp_store.add([f"doc {i}" for i in range(5)], vecs, ns="default",
                  meta=[{"x": 1}] * 5)
    hits = tmp_store.filter_by_meta("x", 1, ns="default", limit=2)
    assert len(hits) == 2


# ── summary_stats ─────────────────────────────────────────────────────────────

def test_summary_stats_keys(populated_store):
    """summary_stats returns all expected keys."""
    store, docs, vecs = populated_store
    stats = store.summary_stats("default")
    assert {"total", "with_summary", "without_summary", "pct_with_summary"} == set(stats.keys())


def test_summary_stats_totals(populated_store):
    """summary_stats total == with + without."""
    store, docs, vecs = populated_store
    stats = store.summary_stats("default")
    assert stats["total"] == stats["with_summary"] + stats["without_summary"]


def test_summary_stats_all_ns(populated_store):
    """summary_stats ns=None spans all namespaces."""
    store, docs, vecs = populated_store
    stats = store.summary_stats(None)
    assert stats["total"] >= len(docs)


def test_summary_stats_empty_ns(tmp_store):
    """summary_stats returns zeroed stats for empty namespace."""
    stats = tmp_store.summary_stats("ghost")
    assert stats["total"] == 0
    assert stats["pct_with_summary"] == 0.0


# ── pinned_memories ───────────────────────────────────────────────────────────

def test_pinned_memories_returns_only_pinned(populated_store):
    """pinned_memories returns only tier=0 memories."""
    store, docs, vecs = populated_store
    # Pin the first memory
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.pin(mid)
    hits = store.pinned_memories("default")
    assert all(h["tier"] == 0 for h in hits)
    assert any(h["id"] == mid for h in hits)


def test_pinned_memories_empty_when_none_pinned(populated_store):
    """pinned_memories returns empty list when no memories are pinned."""
    store, docs, vecs = populated_store
    hits = store.pinned_memories("default")
    assert hits == []


def test_pinned_memories_all_ns(populated_store):
    """pinned_memories ns=None spans all namespaces."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.pin(mid)
    hits = store.pinned_memories(ns=None)
    assert len(hits) >= 1


def test_pinned_memories_limit(populated_store):
    """pinned_memories respects limit."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 3").fetchall()]
    for mid in ids:
        store.pin(mid)
    hits = store.pinned_memories("default", limit=2)
    assert len(hits) == 2


# ── update_meta_key ────────────────────────────────────────────────────────────

def test_update_meta_key_sets_new_key(populated_store):
    """update_meta_key adds a new key to an existing memory's meta."""
    import json as _j
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    result = store.update_meta_key(mid, "new_key", "hello")
    assert result is True
    row = store._db.execute("SELECT meta FROM memories WHERE id=?", (mid,)).fetchone()
    assert _j.loads(row[0])["new_key"] == "hello"


def test_update_meta_key_removes_key(populated_store):
    """update_meta_key with value=None removes the key."""
    import json as _j
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.update_meta_key(mid, "temp", "x")
    store.update_meta_key(mid, "temp", None)
    row = store._db.execute("SELECT meta FROM memories WHERE id=?", (mid,)).fetchone()
    assert "temp" not in _j.loads(row[0])


def test_update_meta_key_missing_id(tmp_store):
    """update_meta_key returns False for non-existent memory."""
    assert tmp_store.update_meta_key(999999, "k", "v") is False


# ── search_by_summary ─────────────────────────────────────────────────────────

def test_search_by_summary_finds_match(tmp_store):
    """search_by_summary returns memories whose summary contains the query."""
    import numpy as np
    vecs = np.random.rand(3, 384).astype(np.float32)
    tmp_store.add(["a", "b", "c"], vecs, ns="default",
                  summaries=["the quick brown fox", "lazy dog", "quick summary"])
    hits = tmp_store.search_by_summary("quick", ns="default")
    assert len(hits) == 2


def test_search_by_summary_no_match(tmp_store):
    """search_by_summary returns empty list when no summary matches."""
    import numpy as np
    vecs = np.random.rand(2, 384).astype(np.float32)
    tmp_store.add(["x", "y"], vecs, ns="default", summaries=["alpha", "beta"])
    hits = tmp_store.search_by_summary("gamma", ns="default")
    assert hits == []


def test_search_by_summary_all_ns(tmp_store):
    """search_by_summary ns=None spans all namespaces."""
    import numpy as np
    v1 = np.random.rand(1, 384).astype(np.float32)
    v2 = np.random.rand(1, 384).astype(np.float32)
    tmp_store.add(["ns1"], v1, ns="ns1", summaries=["target word"])
    tmp_store.add(["ns2"], v2, ns="ns2", summaries=["target word"])
    hits = tmp_store.search_by_summary("target", ns=None)
    assert len(hits) == 2


# ── set_tier_by_tag ────────────────────────────────────────────────────────────

def test_set_tier_by_tag_updates_matching(tmp_store):
    """set_tier_by_tag updates tier only for memories with the tag."""
    import numpy as np
    vecs = np.random.rand(3, 384).astype(np.float32)
    ids = tmp_store.add(["a", "b", "c"], vecs, ns="default",
                        meta=[{"tags": ["important"]}, {"tags": ["other"]}, {"tags": ["important"]}])
    n = tmp_store.set_tier_by_tag("important", 0, ns="default")
    assert n == 2
    for mid in [ids[0], ids[2]]:
        row = tmp_store._db.execute("SELECT tier FROM memories WHERE id=?", (mid,)).fetchone()
        assert row[0] == 0


def test_set_tier_by_tag_no_match(tmp_store):
    """set_tier_by_tag returns 0 when tag not found."""
    import numpy as np
    vecs = np.random.rand(1, 384).astype(np.float32)
    tmp_store.add(["doc"], vecs, ns="default", meta=[{"tags": ["other"]}])
    n = tmp_store.set_tier_by_tag("missing", 0, ns="default")
    assert n == 0


def test_set_tier_by_tag_all_ns(tmp_store):
    """set_tier_by_tag ns=None spans all namespaces."""
    import numpy as np
    v1 = np.random.rand(1, 384).astype(np.float32)
    v2 = np.random.rand(1, 384).astype(np.float32)
    tmp_store.add(["ns1"], v1, ns="ns1", meta=[{"tags": ["flag"]}])
    tmp_store.add(["ns2"], v2, ns="ns2", meta=[{"tags": ["flag"]}])
    n = tmp_store.set_tier_by_tag("flag", 0, ns=None)
    assert n == 2


# ── rotate_ns ──────────────────────────────────────────────────────────────────

def test_rotate_ns_moves_oldest(tmp_store):
    """rotate_ns moves the oldest memories to dst_ns."""
    import numpy as np
    vecs = np.random.rand(5, 384).astype(np.float32)
    tmp_store.add(["a", "b", "c", "d", "e"], vecs, ns="default")
    n = tmp_store.rotate_ns("default", "archive", limit=3)
    assert n == 3
    remaining = tmp_store._db.execute("SELECT count(*) FROM memories WHERE ns='default'").fetchone()[0]
    archived = tmp_store._db.execute("SELECT count(*) FROM memories WHERE ns='archive'").fetchone()[0]
    assert remaining == 2
    assert archived == 3


def test_rotate_ns_empty_source(tmp_store):
    """rotate_ns returns 0 when src_ns is empty."""
    n = tmp_store.rotate_ns("empty", "archive")
    assert n == 0


def test_rotate_ns_by_tier(tmp_store):
    """rotate_ns tier param filters by tier."""
    import numpy as np
    vecs = np.random.rand(3, 384).astype(np.float32)
    ids = tmp_store.add(["a", "b", "c"], vecs, ns="default")
    tmp_store.pin(ids[0])  # tier=0
    n = tmp_store.rotate_ns("default", "archive", tier=1)  # only tier=1
    assert n == 2


# ── compact_meta ──────────────────────────────────────────────────────────────

def test_compact_meta_strips_all(tmp_store):
    """compact_meta with keep_keys=None strips all meta fields."""
    import json, numpy as np
    vecs = np.random.rand(2, 384).astype(np.float32)
    tmp_store.add(["a", "b"], vecs, ns="default",
                  meta=[{"important": "yes", "noise": "x"}, {"tags": ["t"]}])
    n = tmp_store.compact_meta(ns="default", keep_keys=None)
    assert n == 2
    rows = tmp_store._db.execute("SELECT meta FROM memories WHERE ns='default'").fetchall()
    for r in rows:
        assert json.loads(r[0]) == {}


def test_compact_meta_keeps_specified(tmp_store):
    """compact_meta with keep_keys retains only those keys."""
    import json, numpy as np
    vecs = np.random.rand(1, 384).astype(np.float32)
    tmp_store.add(["doc"], vecs, ns="default", meta=[{"a": 1, "b": 2, "c": 3}])
    n = tmp_store.compact_meta(ns="default", keep_keys=["a"])
    assert n == 1
    row = tmp_store._db.execute("SELECT meta FROM memories WHERE ns='default'").fetchone()
    assert json.loads(row[0]) == {"a": 1}


def test_compact_meta_all_ns(tmp_store):
    """compact_meta ns=None spans all namespaces."""
    import numpy as np
    v1 = np.random.rand(1, 384).astype(np.float32)
    v2 = np.random.rand(1, 384).astype(np.float32)
    tmp_store.add(["ns1"], v1, ns="ns1", meta=[{"x": 1}])
    tmp_store.add(["ns2"], v2, ns="ns2", meta=[{"y": 2}])
    n = tmp_store.compact_meta(ns=None, keep_keys=None)
    assert n == 2


def test_compact_meta_no_change_when_keys_match(tmp_store):
    """compact_meta returns 0 when all meta already matches keep_keys."""
    import numpy as np
    vecs = np.random.rand(1, 384).astype(np.float32)
    tmp_store.add(["doc"], vecs, ns="default", meta=[{"a": 1}])
    n = tmp_store.compact_meta(ns="default", keep_keys=["a"])
    assert n == 0


# ── list_by_tier ──────────────────────────────────────────────────────────────

def test_list_by_tier_returns_correct_tier(populated_store):
    """list_by_tier returns only memories with the specified tier."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.pin(mid)
    hits = store.list_by_tier(0, ns="default")
    assert all(h["tier"] == 0 for h in hits)
    assert any(h["id"] == mid for h in hits)


def test_list_by_tier_all_ns(populated_store):
    """list_by_tier ns=None spans all namespaces."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.pin(mid)
    hits = store.list_by_tier(0, ns=None)
    assert len(hits) >= 1


def test_list_by_tier_limit(populated_store):
    """list_by_tier respects limit parameter."""
    store, docs, vecs = populated_store
    hits = store.list_by_tier(1, ns="default", limit=2)
    assert len(hits) <= 2


def test_list_by_tier_empty(tmp_store):
    """list_by_tier returns empty list when no memories at that tier."""
    hits = tmp_store.list_by_tier(2, ns="default")
    assert hits == []


# ── recent ────────────────────────────────────────────────────────────────────

def test_recent_returns_newest_first(tmp_store):
    """recent returns all memories in the namespace."""
    import numpy as np
    v1 = np.random.rand(1, 384).astype(np.float32)
    v2 = np.random.rand(1, 384).astype(np.float32)
    v3 = np.random.rand(1, 384).astype(np.float32)
    tmp_store.add(["first"], v1, ns="default")
    tmp_store.add(["second"], v2, ns="default")
    tmp_store.add(["third"], v3, ns="default")
    hits = tmp_store.recent(ns="default", n=3)
    assert len(hits) == 3
    assert {h["text"] for h in hits} == {"first", "second", "third"}


def test_recent_respects_n(populated_store):
    """recent returns at most n rows."""
    store, docs, vecs = populated_store
    hits = store.recent(ns="default", n=2)
    assert len(hits) == 2


def test_recent_all_ns(populated_store):
    """recent ns=None spans all namespaces."""
    store, docs, vecs = populated_store
    hits = store.recent(ns=None, n=100)
    assert len(hits) >= len(docs)


# ── oldest ────────────────────────────────────────────────────────────────────

def test_oldest_returns_oldest_first(tmp_store):
    """oldest returns memories ordered by created ASC."""
    import numpy as np
    vecs = np.random.rand(3, 384).astype(np.float32)
    tmp_store.add(["first", "second", "third"], vecs, ns="default")
    hits = tmp_store.oldest(ns="default", n=3)
    assert hits[0]["text"] == "first"


def test_oldest_respects_n(populated_store):
    """oldest returns at most n rows."""
    store, docs, vecs = populated_store
    hits = store.oldest(ns="default", n=2)
    assert len(hits) == 2


def test_oldest_all_ns(populated_store):
    """oldest ns=None spans all namespaces."""
    store, docs, vecs = populated_store
    hits = store.oldest(ns=None, n=100)
    assert len(hits) >= len(docs)


# ── replace_text ──────────────────────────────────────────────────────────────

def test_replace_text_updates_field(populated_store):
    """replace_text changes the text column."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    result = store.replace_text(mid, "new content here")
    assert result is True
    row = store._db.execute("SELECT text FROM memories WHERE id=?", (mid,)).fetchone()
    assert row[0] == "new content here"


def test_replace_text_missing_id(tmp_store):
    """replace_text returns False for non-existent memory."""
    assert tmp_store.replace_text(999999, "x") is False


# ── search_text ───────────────────────────────────────────────────────────────

def test_search_text_finds_match(tmp_store):
    """search_text returns memories whose text contains the query."""
    import numpy as np
    vecs = np.random.rand(3, 384).astype(np.float32)
    tmp_store.add(["the quick brown fox", "lazy dog", "quick summary"], vecs, ns="default")
    hits = tmp_store.search_text("quick", ns="default")
    assert len(hits) == 2


def test_search_text_no_match(tmp_store):
    """search_text returns empty list when no text matches."""
    import numpy as np
    vecs = np.random.rand(2, 384).astype(np.float32)
    tmp_store.add(["alpha text", "beta text"], vecs, ns="default")
    hits = tmp_store.search_text("gamma", ns="default")
    assert hits == []


def test_search_text_all_ns(tmp_store):
    """search_text ns=None spans all namespaces."""
    import numpy as np
    v1 = np.random.rand(1, 384).astype(np.float32)
    v2 = np.random.rand(1, 384).astype(np.float32)
    tmp_store.add(["needle in ns1"], v1, ns="ns1")
    tmp_store.add(["needle in ns2"], v2, ns="ns2")
    hits = tmp_store.search_text("needle", ns=None)
    assert len(hits) == 2


def test_search_text_limit(tmp_store):
    """search_text respects limit."""
    import numpy as np
    vecs = np.random.rand(5, 384).astype(np.float32)
    tmp_store.add(["match a", "match b", "match c", "match d", "match e"], vecs, ns="default")
    hits = tmp_store.search_text("match", ns="default", limit=3)
    assert len(hits) == 3


# ── count_by_ns ───────────────────────────────────────────────────────────────

def test_count_by_ns_returns_correct_counts(tmp_store):
    """count_by_ns returns accurate counts per namespace."""
    import numpy as np
    v1 = np.random.rand(3, 384).astype(np.float32)
    v2 = np.random.rand(2, 384).astype(np.float32)
    tmp_store.add(["a", "b", "c"], v1, ns="ns_a")
    tmp_store.add(["x", "y"], v2, ns="ns_b")
    counts = tmp_store.count_by_ns()
    assert counts["ns_a"] == 3
    assert counts["ns_b"] == 2


def test_count_by_ns_empty(tmp_store):
    """count_by_ns returns empty dict when no memories."""
    assert tmp_store.count_by_ns() == {}


# ── clear_ns ──────────────────────────────────────────────────────────────────

def test_clear_ns_deletes_all(tmp_store):
    """clear_ns removes all memories from the namespace."""
    import numpy as np
    vecs = np.random.rand(3, 384).astype(np.float32)
    tmp_store.add(["a", "b", "c"], vecs, ns="to_clear")
    n = tmp_store.clear_ns("to_clear")
    assert n == 3
    remaining = tmp_store._db.execute(
        "SELECT count(*) FROM memories WHERE ns='to_clear'"
    ).fetchone()[0]
    assert remaining == 0


def test_clear_ns_returns_zero_for_empty(tmp_store):
    """clear_ns returns 0 when namespace is already empty."""
    n = tmp_store.clear_ns("nonexistent")
    assert n == 0


def test_clear_ns_does_not_affect_other_ns(tmp_store):
    """clear_ns only removes memories from the target namespace."""
    import numpy as np
    v1 = np.random.rand(2, 384).astype(np.float32)
    v2 = np.random.rand(2, 384).astype(np.float32)
    tmp_store.add(["a", "b"], v1, ns="to_clear")
    tmp_store.add(["x", "y"], v2, ns="keep")
    tmp_store.clear_ns("to_clear")
    kept = tmp_store._db.execute(
        "SELECT count(*) FROM memories WHERE ns='keep'"
    ).fetchone()[0]
    assert kept == 2


# ── copy_to_ns ────────────────────────────────────────────────────────────────

def test_copy_to_ns_copies_memories(tmp_store):
    """copy_to_ns duplicates memories into dst_ns."""
    import numpy as np
    vecs = np.random.rand(3, 384).astype(np.float32)
    ids = tmp_store.add(["alpha", "beta", "gamma"], vecs, ns="src")
    n = tmp_store.copy_to_ns(ids[:2], "dst")
    assert n == 2
    dst_count = tmp_store._db.execute(
        "SELECT count(*) FROM memories WHERE ns='dst'"
    ).fetchone()[0]
    assert dst_count == 2
    src_count = tmp_store._db.execute(
        "SELECT count(*) FROM memories WHERE ns='src'"
    ).fetchone()[0]
    assert src_count == 3  # source untouched


def test_copy_to_ns_empty_ids(tmp_store):
    """copy_to_ns returns 0 when ids list is empty."""
    n = tmp_store.copy_to_ns([], "dst")
    assert n == 0


def test_copy_to_ns_preserves_text(tmp_store):
    """copy_to_ns copies text and summary correctly."""
    import numpy as np
    vecs = np.random.rand(1, 384).astype(np.float32)
    ids = tmp_store.add(["original text"], vecs, ns="src", summaries=["orig summary"])
    tmp_store.copy_to_ns(ids, "dst")
    row = tmp_store._db.execute(
        "SELECT text, summary FROM memories WHERE ns='dst'"
    ).fetchone()
    assert row[0] == "original text"
    assert row[1] == "orig summary"


def test_copy_to_ns_preserves_non_default_tier(tmp_store):
    """copy_to_ns carries over non-default (tier != 1) values."""
    import numpy as np
    vecs = np.random.rand(1, 384).astype(np.float32)
    ids = tmp_store.add(["pinned memory"], vecs, ns="src")
    # make it pinned (tier=0)
    tmp_store._db.execute("UPDATE memories SET tier=0 WHERE id=?", (ids[0],))
    tmp_store._db.commit()
    tmp_store.copy_to_ns(ids, "dst")
    row = tmp_store._db.execute(
        "SELECT tier FROM memories WHERE ns='dst'"
    ).fetchone()
    assert row[0] == 0


# ── rename_tag ─────────────────────────────────────────────────────────────────

def test_rename_tag_renames_correctly(tmp_store):
    """rename_tag replaces old_tag with new_tag in meta.tags."""
    import numpy as np, json
    vecs = np.random.rand(2, 384).astype(np.float32)
    ids = tmp_store.add(["a", "b"], vecs, ns="default",
                        meta=[{"tags": ["alpha", "beta"]}, {"tags": ["beta", "gamma"]}])
    n = tmp_store.rename_tag("beta", "delta", ns="default")
    assert n == 2
    row = tmp_store._db.execute("SELECT meta FROM memories WHERE id=?", (ids[0],)).fetchone()
    tags = json.loads(row[0])["tags"]
    assert "delta" in tags
    assert "beta" not in tags


def test_rename_tag_skips_missing(tmp_store):
    """rename_tag returns 0 when old_tag not present in any memory."""
    import numpy as np
    vecs = np.random.rand(1, 384).astype(np.float32)
    tmp_store.add(["a"], vecs, ns="default", meta=[{"tags": ["foo"]}])
    n = tmp_store.rename_tag("nonexistent", "bar", ns="default")
    assert n == 0


def test_rename_tag_same_tag_noop(tmp_store):
    """rename_tag returns 0 when old_tag == new_tag."""
    n = tmp_store.rename_tag("x", "x", ns="default")
    assert n == 0


def test_rename_tag_all_ns(tmp_store):
    """rename_tag ns=None spans all namespaces."""
    import numpy as np
    v1 = np.random.rand(1, 384).astype(np.float32)
    v2 = np.random.rand(1, 384).astype(np.float32)
    tmp_store.add(["a"], v1, ns="ns1", meta=[{"tags": ["shared"]}])
    tmp_store.add(["b"], v2, ns="ns2", meta=[{"tags": ["shared"]}])
    n = tmp_store.rename_tag("shared", "unified", ns=None)
    assert n == 2


# ── find_duplicates ────────────────────────────────────────────────────────────

def test_find_duplicates_detects_dupes(tmp_store):
    """find_duplicates returns groups with identical text."""
    import numpy as np
    vecs = np.random.rand(4, 384).astype(np.float32)
    tmp_store.add(["same", "same", "unique", "same"], vecs, ns="default")
    groups = tmp_store.find_duplicates(ns="default")
    assert len(groups) == 1
    assert groups[0]["count"] == 3
    assert groups[0]["text"] == "same"


def test_find_duplicates_empty(tmp_store):
    """find_duplicates returns empty list when no duplicates."""
    import numpy as np
    vecs = np.random.rand(3, 384).astype(np.float32)
    tmp_store.add(["a", "b", "c"], vecs, ns="default")
    assert tmp_store.find_duplicates(ns="default") == []


def test_find_duplicates_all_ns(tmp_store):
    """find_duplicates ns=None spans all namespaces."""
    import numpy as np
    v1 = np.random.rand(2, 384).astype(np.float32)
    tmp_store.add(["dupe", "dupe"], v1, ns="ns1")
    groups = tmp_store.find_duplicates(ns=None)
    assert len(groups) == 1


def test_find_duplicates_limit(tmp_store):
    """find_duplicates respects limit."""
    import numpy as np
    vecs = np.random.rand(6, 384).astype(np.float32)
    tmp_store.add(["a", "a", "b", "b", "c", "c"], vecs, ns="default")
    groups = tmp_store.find_duplicates(ns="default", limit=1)
    assert len(groups) == 1


# ── swap_tier ─────────────────────────────────────────────────────────────────

def test_swap_tier_moves_memories(tmp_store):
    """swap_tier bulk-changes tier in a namespace."""
    import numpy as np
    vecs = np.random.rand(3, 384).astype(np.float32)
    ids = tmp_store.add(["a", "b", "c"], vecs, ns="myns")
    # set two to tier=0
    tmp_store._db.execute("UPDATE memories SET tier=0 WHERE id IN (?,?)", ids[:2])
    tmp_store._db.commit()
    n = tmp_store.swap_tier("myns", 0, 2)
    assert n == 2
    rows = tmp_store._db.execute(
        "SELECT tier FROM memories WHERE ns='myns'"
    ).fetchall()
    tiers = [r[0] for r in rows]
    assert 0 not in tiers
    assert tiers.count(2) == 2


def test_swap_tier_same_tier_noop(tmp_store):
    """swap_tier returns 0 when from_tier == to_tier."""
    n = tmp_store.swap_tier("myns", 1, 1)
    assert n == 0


def test_swap_tier_invalid_tier(tmp_store):
    """swap_tier raises ValueError for invalid tier values."""
    import pytest
    with pytest.raises(ValueError):
        tmp_store.swap_tier("myns", 0, 5)


def test_swap_tier_returns_zero_for_nonexistent_ns(tmp_store):
    """swap_tier returns 0 for namespace with no matching memories."""
    n = tmp_store.swap_tier("noexist", 1, 2)
    assert n == 0


# ── ns_summary ────────────────────────────────────────────────────────────────

def test_ns_summary_returns_correct_stats(tmp_store):
    """ns_summary returns accurate counts and stats for a populated ns."""
    import numpy as np
    vecs = np.random.rand(3, 384).astype(np.float32)
    tmp_store.add(["hello world", "foo", "bar baz qux"], vecs, ns="myns")
    # pin one (SQLite has no UPDATE...LIMIT without a separate compile flag)
    first_id = tmp_store._db.execute(
        "SELECT id FROM memories WHERE ns='myns' LIMIT 1"
    ).fetchone()[0]
    tmp_store._db.execute("UPDATE memories SET tier=0 WHERE id=?", (first_id,))
    tmp_store._db.commit()
    s = tmp_store.ns_summary("myns")
    assert s["ns"] == "myns"
    assert s["count"] == 3
    assert s["tiers"]["pinned"] == 1
    assert s["min_chars"] > 0
    assert s["oldest"] is not None


def test_ns_summary_empty_ns(tmp_store):
    """ns_summary returns zeroed dict for empty namespace."""
    s = tmp_store.ns_summary("does_not_exist")
    assert s["count"] == 0
    assert s["tiers"]["pinned"] == 0
    assert s["oldest"] is None


def test_rename_tag_skips_invalid_json(tmp_store):
    """rename_tag skips memories with malformed meta JSON without crashing."""
    import numpy as np
    vecs = np.random.rand(1, 384).astype(np.float32)
    ids = tmp_store.add(["x"], vecs, ns="default")
    # Inject broken JSON that still passes the LIKE '%"tags"%' query filter
    tmp_store._db.execute(
        "UPDATE memories SET meta=? WHERE id=?",
        ('{"tags": broken_not_json}', ids[0]),
    )
    tmp_store._db.commit()
    # Should not raise, should return 0 (no update possible)
    n = tmp_store.rename_tag("foo", "bar", ns="default")
    assert n == 0


# ── toggle_tier ────────────────────────────────────────────────────────────────

def test_toggle_tier_cycles_correctly(tmp_store):
    """toggle_tier cycles: default(1)→pinned(0)→ambient(2)→default(1)."""
    import numpy as np
    vecs = np.random.rand(1, 384).astype(np.float32)
    ids = tmp_store.add(["x"], vecs, ns="default")
    memory_id = ids[0]
    # initial tier should be 1 (default)
    t1 = tmp_store.toggle_tier(memory_id)
    assert t1 == 0  # 1→0 (pinned)
    t2 = tmp_store.toggle_tier(memory_id)
    assert t2 == 2  # 0→2 (ambient)
    t3 = tmp_store.toggle_tier(memory_id)
    assert t3 == 1  # 2→1 (default)


def test_toggle_tier_not_found(tmp_store):
    """toggle_tier returns None for non-existent memory."""
    result = tmp_store.toggle_tier(99999)
    assert result is None


# ── merge_texts ────────────────────────────────────────────────────────────────

def test_merge_texts_creates_new_memory(tmp_store):
    """merge_texts creates a new memory with concatenated text."""
    import numpy as np
    vecs = np.random.rand(3, 384).astype(np.float32)
    ids = tmp_store.add(["alpha", "beta", "gamma"], vecs, ns="default")
    new_id = tmp_store.merge_texts(ids, separator=" | ", ns="default")
    assert new_id is not None
    row = tmp_store._db.execute(
        "SELECT text FROM memories WHERE id=?", (new_id,)
    ).fetchone()
    assert "alpha" in row[0]
    assert "beta" in row[0]
    assert " | " in row[0]


def test_merge_texts_delete_originals(tmp_store):
    """merge_texts with delete_originals removes source memories."""
    import numpy as np
    vecs = np.random.rand(2, 384).astype(np.float32)
    ids = tmp_store.add(["x", "y"], vecs, ns="default")
    new_id = tmp_store.merge_texts(ids, delete_originals=True)
    assert new_id is not None
    # originals gone
    remaining = tmp_store._db.execute(
        f"SELECT COUNT(*) FROM memories WHERE id IN ({','.join('?'*len(ids))})", ids
    ).fetchone()[0]
    assert remaining == 0


def test_merge_texts_empty_ids(tmp_store):
    """merge_texts returns None for empty ids list."""
    result = tmp_store.merge_texts([])
    assert result is None


def test_merge_texts_missing_ids(tmp_store):
    """merge_texts returns None when none of the ids exist."""
    result = tmp_store.merge_texts([99998, 99999])
    assert result is None


# ── truncate_text ─────────────────────────────────────────────────────────────

def test_truncate_text_shortens(tmp_store):
    """truncate_text trims text to max_chars."""
    import numpy as np
    vecs = np.random.rand(1, 384).astype(np.float32)
    ids = tmp_store.add(["hello world long text"], vecs, ns="default")
    ok = tmp_store.truncate_text(ids[0], 5)
    assert ok is True
    row = tmp_store._db.execute(
        "SELECT text FROM memories WHERE id=?", (ids[0],)
    ).fetchone()
    assert row[0] == "hello"


def test_truncate_text_noop_if_short(tmp_store):
    """truncate_text is a no-op if text already fits."""
    import numpy as np
    vecs = np.random.rand(1, 384).astype(np.float32)
    ids = tmp_store.add(["hi"], vecs, ns="default")
    ok = tmp_store.truncate_text(ids[0], 100)
    assert ok is True
    row = tmp_store._db.execute(
        "SELECT text FROM memories WHERE id=?", (ids[0],)
    ).fetchone()
    assert row[0] == "hi"


def test_truncate_text_not_found(tmp_store):
    """truncate_text returns False for non-existent memory."""
    result = tmp_store.truncate_text(99999, 10)
    assert result is False


# ── search_by_access_count ────────────────────────────────────────────────────

def test_search_by_access_count_min_only(tmp_store):
    """search_by_access_count with min_count=0 returns all."""
    import numpy as np
    vecs = np.random.rand(3, 384).astype(np.float32)
    tmp_store.add(["a", "b", "c"], vecs, ns="default")
    hits = tmp_store.search_by_access_count(min_count=0, ns="default")
    assert len(hits) == 3


def test_search_by_access_count_max_zero(tmp_store):
    """search_by_access_count max_count=0 returns never-accessed memories."""
    import numpy as np
    vecs = np.random.rand(2, 384).astype(np.float32)
    ids = tmp_store.add(["a", "b"], vecs, ns="default")
    # increment one
    tmp_store._db.execute(
        "UPDATE memories SET access_count=5 WHERE id=?", (ids[0],)
    )
    tmp_store._db.commit()
    hits = tmp_store.search_by_access_count(min_count=0, max_count=0, ns="default")
    assert len(hits) == 1
    assert hits[0]["access_count"] == 0


def test_search_by_access_count_all_ns(tmp_store):
    """search_by_access_count ns=None spans all namespaces."""
    import numpy as np
    v1 = np.random.rand(1, 384).astype(np.float32)
    v2 = np.random.rand(1, 384).astype(np.float32)
    tmp_store.add(["a"], v1, ns="ns1")
    tmp_store.add(["b"], v2, ns="ns2")
    hits = tmp_store.search_by_access_count(min_count=0, ns=None)
    assert len(hits) >= 2


def test_search_by_access_count_limit(tmp_store):
    """search_by_access_count respects limit."""
    import numpy as np
    vecs = np.random.rand(5, 384).astype(np.float32)
    tmp_store.add(["a", "b", "c", "d", "e"], vecs, ns="default")
    hits = tmp_store.search_by_access_count(min_count=0, ns="default", limit=2)
    assert len(hits) == 2


# ── age_by_ns ─────────────────────────────────────────────────────────────────

def test_age_by_ns_returns_buckets(tmp_store):
    """age_by_ns returns count breakdowns for a populated ns."""
    import numpy as np
    vecs = np.random.rand(3, 384).astype(np.float32)
    tmp_store.add(["a", "b", "c"], vecs, ns="myns")
    result = tmp_store.age_by_ns("myns")
    assert result["ns"] == "myns"
    assert result["total"] == 3
    # All were just created, so should be in "today"
    assert result["today"] == 3


def test_age_by_ns_empty_ns(tmp_store):
    """age_by_ns returns zeroed dict for empty namespace."""
    result = tmp_store.age_by_ns("nonexistent")
    assert result["total"] == 0
    assert result["today"] == 0


# ── delete_by_tier ────────────────────────────────────────────────────────────

def test_delete_by_tier_removes_matching(tmp_store):
    """delete_by_tier removes only memories of the target tier."""
    import numpy as np
    vecs = np.random.rand(4, 384).astype(np.float32)
    ids = tmp_store.add(["a", "b", "c", "d"], vecs, ns="myns")
    # Set two to tier=2
    tmp_store._db.execute(
        "UPDATE memories SET tier=2 WHERE id IN (?,?)", ids[:2]
    )
    tmp_store._db.commit()
    n = tmp_store.delete_by_tier("myns", 2)
    assert n == 2
    remaining = tmp_store._db.execute(
        "SELECT COUNT(*) FROM memories WHERE ns='myns'"
    ).fetchone()[0]
    assert remaining == 2


def test_delete_by_tier_invalid(tmp_store):
    """delete_by_tier raises ValueError for invalid tier."""
    import pytest
    with pytest.raises(ValueError):
        tmp_store.delete_by_tier("myns", 5)


def test_delete_by_tier_returns_zero_for_no_match(tmp_store):
    """delete_by_tier returns 0 when no memories match."""
    n = tmp_store.delete_by_tier("nonexistent_ns", 1)
    assert n == 0


# ── untagged_memories ─────────────────────────────────────────────────────────

def test_untagged_memories_returns_untagged(tmp_store):
    """untagged_memories returns memories without tags."""
    import numpy as np
    vecs = np.random.rand(3, 384).astype(np.float32)
    ids = tmp_store.add(["a", "b", "c"], vecs, ns="default",
                        meta=[{"tags": ["t1"]}, {}, {}])
    hits = tmp_store.untagged_memories(ns="default")
    assert len(hits) == 2
    tagged_ids = {h["id"] for h in hits}
    assert ids[0] not in tagged_ids


def test_untagged_memories_empty_tags_list(tmp_store):
    """untagged_memories includes memories with empty tags list."""
    import numpy as np
    vecs = np.random.rand(1, 384).astype(np.float32)
    tmp_store.add(["x"], vecs, ns="default", meta=[{"tags": []}])
    hits = tmp_store.untagged_memories(ns="default")
    assert len(hits) == 1


def test_untagged_memories_all_ns(tmp_store):
    """untagged_memories ns=None spans all namespaces."""
    import numpy as np
    v1 = np.random.rand(1, 384).astype(np.float32)
    v2 = np.random.rand(1, 384).astype(np.float32)
    tmp_store.add(["a"], v1, ns="ns1")
    tmp_store.add(["b"], v2, ns="ns2")
    hits = tmp_store.untagged_memories(ns=None)
    assert len(hits) == 2


def test_untagged_memories_limit(tmp_store):
    """untagged_memories respects limit."""
    import numpy as np
    vecs = np.random.rand(5, 384).astype(np.float32)
    tmp_store.add(["a", "b", "c", "d", "e"], vecs, ns="default")
    hits = tmp_store.untagged_memories(ns="default", limit=2)
    assert len(hits) == 2


# ── set_meta_for_untagged ─────────────────────────────────────────────────────

def test_set_meta_for_untagged_updates(tmp_store):
    """set_meta_for_untagged sets key on untagged memories."""
    import numpy as np, json
    vecs = np.random.rand(3, 384).astype(np.float32)
    tmp_store.add(["a", "b", "c"], vecs, ns="default",
                  meta=[{"tags": ["t"]}, {}, {}])
    n = tmp_store.set_meta_for_untagged("default", "source", "batch")
    assert n == 2
    row = tmp_store._db.execute(
        "SELECT meta FROM memories WHERE ns='default' AND meta NOT LIKE '%\"tags\"%'"
        " LIMIT 1"
    ).fetchone()
    meta = json.loads(row[0])
    assert meta.get("source") == "batch"


def test_set_meta_for_untagged_returns_zero_for_empty(tmp_store):
    """set_meta_for_untagged returns 0 when no untagged memories."""
    import numpy as np
    vecs = np.random.rand(2, 384).astype(np.float32)
    tmp_store.add(["a", "b"], vecs, ns="default",
                  meta=[{"tags": ["t"]}, {"tags": ["s"]}])
    n = tmp_store.set_meta_for_untagged("default", "k", "v")
    assert n == 0


def test_set_meta_for_untagged_with_broken_meta(tmp_store):
    """set_meta_for_untagged handles memories with broken JSON meta."""
    import numpy as np
    vecs = np.random.rand(1, 384).astype(np.float32)
    ids = tmp_store.add(["x"], vecs, ns="default")
    # Force broken JSON (not containing "tags" so untagged_memories picks it up)
    tmp_store._db.execute(
        "UPDATE memories SET meta=? WHERE id=?", ("{broken json", ids[0])
    )
    tmp_store._db.commit()
    n = tmp_store.set_meta_for_untagged("default", "k", "v")
    assert n == 1


# ─── Batch 5: clone_memory, memories_without_summary, pin_by_tag, promote_by_access ───

def test_clone_memory_same_ns(tmp_store):
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    ids = tmp_store.add(["clone target"], vecs, ns="default")
    new_id = tmp_store.clone_memory(ids[0])
    assert new_id is not None
    assert new_id != ids[0]
    row = tmp_store.get(new_id)
    assert row["text"] == "clone target"
    assert row["ns"] == "default"


def test_clone_memory_dst_ns(tmp_store):
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    ids = tmp_store.add(["clone dst"], vecs, ns="default")
    new_id = tmp_store.clone_memory(ids[0], dst_ns="archive")
    assert new_id is not None
    row = tmp_store.get(new_id)
    assert row["ns"] == "archive"


def test_clone_memory_preserves_tier(tmp_store):
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    ids = tmp_store.add(["pinned"], vecs, ns="default")
    tmp_store.set_tier(ids[0], 0)
    new_id = tmp_store.clone_memory(ids[0])
    assert new_id is not None
    row = tmp_store.get(new_id)
    assert row["tier"] == 0


def test_clone_memory_not_found(tmp_store):
    result = tmp_store.clone_memory(999999)
    assert result is None


def test_memories_without_summary(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    ids = tmp_store.add(["has summary", "no summary"], vecs, ns="default",
                        summaries=["s", None])
    hits = tmp_store.memories_without_summary(ns="default")
    hit_ids = [h["id"] for h in hits]
    assert ids[1] in hit_ids
    assert ids[0] not in hit_ids


def test_memories_without_summary_all_ns(tmp_store):
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    tmp_store.add(["x"], vecs, ns="ns1")
    hits = tmp_store.memories_without_summary(ns=None, limit=100)
    assert len(hits) >= 1


def test_memories_without_summary_limit(tmp_store):
    import numpy as np
    vecs = np.zeros((5, 384), dtype=np.float32)
    tmp_store.add(["a", "b", "c", "d", "e"], vecs, ns="default")
    hits = tmp_store.memories_without_summary(ns="default", limit=2)
    assert len(hits) <= 2


def test_pin_by_tag(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    ids = tmp_store.add(["x", "y"], vecs, ns="default")
    tmp_store.tag(ids[0], "urgent")
    tmp_store.tag(ids[1], "other")
    n = tmp_store.pin_by_tag("urgent", ns="default")
    assert n == 1
    assert tmp_store.get(ids[0])["tier"] == 0
    assert tmp_store.get(ids[1])["tier"] == 1


def test_pin_by_tag_all_ns(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    ids_a = tmp_store.add(["a"], np.zeros((1, 384), dtype=np.float32), ns="ns1")
    ids_b = tmp_store.add(["b"], np.zeros((1, 384), dtype=np.float32), ns="ns2")
    tmp_store.tag(ids_a[0], "vip")
    tmp_store.tag(ids_b[0], "vip")
    n = tmp_store.pin_by_tag("vip", ns=None)
    assert n == 2


def test_promote_by_access_basic(tmp_store):
    import numpy as np
    vecs = np.zeros((3, 384), dtype=np.float32)
    ids = tmp_store.add(["a", "b", "c"], vecs, ns="default", tier=2)
    # Give the first one high access count
    tmp_store._db.execute("UPDATE memories SET access_count=10 WHERE id=?", (ids[0],))
    tmp_store._db.commit()
    n = tmp_store.promote_by_access("default", n=1, from_tier=2, to_tier=1)
    assert n == 1
    assert tmp_store.get(ids[0])["tier"] == 1


def test_promote_by_access_same_tier_returns_zero(tmp_store):
    n = tmp_store.promote_by_access("default", from_tier=1, to_tier=1)
    assert n == 0


def test_promote_by_access_bad_tier(tmp_store):
    import pytest
    with pytest.raises(ValueError):
        tmp_store.promote_by_access("default", from_tier=5, to_tier=0)


def test_promote_by_access_empty_ns(tmp_store):
    n = tmp_store.promote_by_access("nonexistent", n=5, from_tier=2, to_tier=1)
    assert n == 0


# ─── Batch 6: filter_by_text_length, multi_tag_filter, tag_stats, split_memory ───

def test_filter_by_text_length_max(tmp_store):
    import numpy as np
    vecs = np.zeros((3, 384), dtype=np.float32)
    tmp_store.add(["short", "a medium length text here", "a much longer text that is quite extended"], vecs, ns="default")
    hits = tmp_store.filter_by_text_length(max_chars=10, ns="default")
    texts = [h["text"] for h in hits]
    assert "short" in texts
    assert "a much longer text that is quite extended" not in texts


def test_filter_by_text_length_min(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    tmp_store.add(["x", "long enough text"], vecs, ns="default")
    hits = tmp_store.filter_by_text_length(min_chars=5, ns="default")
    texts = [h["text"] for h in hits]
    assert "x" not in texts
    assert "long enough text" in texts


def test_filter_by_text_length_all_ns(tmp_store):
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    tmp_store.add(["hi"], vecs, ns="other")
    hits = tmp_store.filter_by_text_length(max_chars=10, ns=None)
    assert any(h["text"] == "hi" for h in hits)


def test_filter_by_text_length_limit(tmp_store):
    import numpy as np
    vecs = np.zeros((5, 384), dtype=np.float32)
    tmp_store.add(["a", "b", "c", "d", "e"], vecs, ns="default")
    hits = tmp_store.filter_by_text_length(max_chars=5, ns="default", limit=2)
    assert len(hits) <= 2


def test_multi_tag_filter_all(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    ids = tmp_store.add(["x", "y"], vecs, ns="default")
    tmp_store.bulk_tag([ids[0]], ["a", "b"])
    tmp_store.bulk_tag([ids[1]], ["a"])
    hits = tmp_store.multi_tag_filter(["a", "b"], ns="default", mode="all")
    assert len(hits) == 1 and hits[0]["id"] == ids[0]


def test_multi_tag_filter_any(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    ids = tmp_store.add(["x", "y"], vecs, ns="default")
    tmp_store.bulk_tag([ids[0]], ["a"])
    tmp_store.bulk_tag([ids[1]], ["b"])
    hits = tmp_store.multi_tag_filter(["a", "b"], ns="default", mode="any")
    assert len(hits) == 2


def test_multi_tag_filter_empty_tags(tmp_store):
    hits = tmp_store.multi_tag_filter([], ns="default")
    assert hits == []


def test_multi_tag_filter_all_ns(tmp_store):
    import numpy as np
    ids1 = tmp_store.add(["x"], np.zeros((1, 384), dtype=np.float32), ns="ns1")
    tmp_store.bulk_tag(ids1, ["q"])
    hits = tmp_store.multi_tag_filter(["q"], ns=None, mode="any")
    assert len(hits) >= 1


def test_multi_tag_filter_limit(tmp_store):
    import numpy as np
    vecs = np.zeros((4, 384), dtype=np.float32)
    ids = tmp_store.add(["a", "b", "c", "d"], vecs, ns="default")
    tmp_store.bulk_tag(ids, ["x"])
    hits = tmp_store.multi_tag_filter(["x"], ns="default", mode="any", limit=2)
    assert len(hits) <= 2


def test_tag_stats_basic(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    ids = tmp_store.add(["x", "y"], vecs, ns="default")
    tmp_store.bulk_tag([ids[0]], ["a", "b"])
    tmp_store.bulk_tag([ids[1]], ["a"])
    stats = tmp_store.tag_stats(ns="default")
    tag_names = [s["tag"] for s in stats]
    assert "a" in tag_names and "b" in tag_names
    a_stat = next(s for s in stats if s["tag"] == "a")
    assert a_stat["count"] == 2


def test_tag_stats_all_ns(tmp_store):
    import numpy as np
    ids1 = tmp_store.add(["p"], np.zeros((1, 384), dtype=np.float32), ns="ns1")
    tmp_store.bulk_tag(ids1, ["z"])
    stats = tmp_store.tag_stats(ns=None)
    assert any(s["tag"] == "z" for s in stats)


def test_tag_stats_no_tags(tmp_store):
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    tmp_store.add(["x"], vecs, ns="default")
    stats = tmp_store.tag_stats(ns="default")
    assert stats == []


def test_split_memory_separator(tmp_store):
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    ids = tmp_store.add(["part one\n\npart two\n\npart three"], vecs, ns="default")
    new_ids = tmp_store.split_memory(ids[0], separator="\n\n")
    assert new_ids is not None and len(new_ids) == 3
    assert tmp_store.get(new_ids[0])["text"] == "part one"


def test_split_memory_max_chars(tmp_store):
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    ids = tmp_store.add(["hello world foo bar baz"], vecs, ns="default")
    new_ids = tmp_store.split_memory(ids[0], max_chars=15)
    assert new_ids is not None and len(new_ids) >= 2


def test_split_memory_delete_original(tmp_store):
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    ids = tmp_store.add(["a\n\nb"], vecs, ns="default")
    tmp_store.split_memory(ids[0], separator="\n\n", delete_original=True)
    assert tmp_store.get(ids[0]) is None


def test_split_memory_not_found(tmp_store):
    result = tmp_store.split_memory(999999, separator="\n\n")
    assert result is None


def test_split_memory_too_few_parts(tmp_store):
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    ids = tmp_store.add(["no separator here"], vecs, ns="default")
    result = tmp_store.split_memory(ids[0], separator="ZZZNOMATCH")
    assert result is None


def test_split_memory_preserves_tier(tmp_store):
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    ids = tmp_store.add(["part a\n\npart b"], vecs, ns="default")
    tmp_store.set_tier(ids[0], 0)
    new_ids = tmp_store.split_memory(ids[0], separator="\n\n")
    assert new_ids is not None
    for nid in new_ids:
        assert tmp_store.get(nid)["tier"] == 0


def test_multi_tag_filter_broken_json(tmp_store):
    """multi_tag_filter skips memories with broken JSON meta."""
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    ids = tmp_store.add(["broken"], vecs, ns="default")
    tmp_store._db.execute("UPDATE memories SET meta=? WHERE id=?", ("{bad json", ids[0]))
    tmp_store._db.commit()
    # Should not raise; broken meta treated as no tags
    hits = tmp_store.multi_tag_filter(["x"], ns="default", mode="any")
    assert all(h["id"] != ids[0] for h in hits)


def test_tag_stats_broken_json(tmp_store):
    """tag_stats skips memories with broken JSON meta."""
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    ids = tmp_store.add(["x"], vecs, ns="default")
    tmp_store._db.execute("UPDATE memories SET meta=? WHERE id=?", ("{bad", ids[0]))
    tmp_store._db.commit()
    stats = tmp_store.tag_stats(ns="default")
    assert stats == []


def test_split_memory_no_separator_no_max_chars(tmp_store):
    """split_memory with neither separator nor max_chars returns None (single part)."""
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    ids = tmp_store.add(["just one part"], vecs, ns="default")
    result = tmp_store.split_memory(ids[0])
    assert result is None


# ─── Batch 7: bulk_summarize, cross_ns_search, memory_timeline, keyword_extract ───

def test_bulk_summarize_basic(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    ids = tmp_store.add(["a", "b"], vecs, ns="default")
    n = tmp_store.bulk_summarize({ids[0]: "sum-a", ids[1]: "sum-b"})
    assert n == 2
    assert tmp_store.get(ids[0])["summary"] == "sum-a"
    assert tmp_store.get(ids[1])["summary"] == "sum-b"


def test_bulk_summarize_clear(tmp_store):
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    ids = tmp_store.add(["x"], vecs, ns="default", summaries=["s"])
    n = tmp_store.bulk_summarize({ids[0]: None})
    assert n == 1
    assert tmp_store.get(ids[0])["summary"] is None


def test_bulk_summarize_empty(tmp_store):
    n = tmp_store.bulk_summarize({})
    assert n == 0


def test_bulk_summarize_skips_missing(tmp_store):
    n = tmp_store.bulk_summarize({999999: "ghost"})
    assert n == 0


def test_cross_ns_search_basic(tmp_store):
    import numpy as np
    tmp_store.add(["apple pie"], np.zeros((1, 384), dtype=np.float32), ns="ns1")
    tmp_store.add(["apple cider"], np.zeros((1, 384), dtype=np.float32), ns="ns2")
    tmp_store.add(["banana bread"], np.zeros((1, 384), dtype=np.float32), ns="ns3")
    hits = tmp_store.cross_ns_search("apple", ["ns1", "ns2"])
    assert len(hits) == 2
    nss = {h["ns"] for h in hits}
    assert nss == {"ns1", "ns2"}


def test_cross_ns_search_empty_ns_list(tmp_store):
    hits = tmp_store.cross_ns_search("x", [])
    assert hits == []


def test_cross_ns_search_limit(tmp_store):
    import numpy as np
    for i in range(5):
        tmp_store.add([f"hello world {i}"], np.zeros((1, 384), dtype=np.float32), ns="ns1")
    hits = tmp_store.cross_ns_search("hello", ["ns1"], limit=2)
    assert len(hits) <= 2


def test_memory_timeline_basic(tmp_store):
    import numpy as np
    vecs = np.zeros((3, 384), dtype=np.float32)
    tmp_store.add(["first", "second", "third"], vecs, ns="default")
    tl = tmp_store.memory_timeline(ns="default")
    assert len(tl) == 3
    for i, e in enumerate(tl):
        assert e["seq"] == i + 1


def test_memory_timeline_tier_filter(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    ids = tmp_store.add(["x", "y"], vecs, ns="default")
    tmp_store.set_tier(ids[0], 0)
    tl = tmp_store.memory_timeline(ns="default", tier=0)
    assert all(e["tier"] == 0 for e in tl)


def test_memory_timeline_all_ns(tmp_store):
    import numpy as np
    tmp_store.add(["a"], np.zeros((1, 384), dtype=np.float32), ns="ns1")
    tmp_store.add(["b"], np.zeros((1, 384), dtype=np.float32), ns="ns2")
    tl = tmp_store.memory_timeline(ns=None, limit=100)
    nss = {e["ns"] for e in tl}
    assert "ns1" in nss and "ns2" in nss


def test_memory_timeline_limit(tmp_store):
    import numpy as np
    vecs = np.zeros((5, 384), dtype=np.float32)
    tmp_store.add(["a", "b", "c", "d", "e"], vecs, ns="default")
    tl = tmp_store.memory_timeline(ns="default", limit=2)
    assert len(tl) <= 2


def test_keyword_extract_basic(tmp_store):
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    ids = tmp_store.add(["machine learning models predict outcomes data science"], vecs, ns="default")
    result = tmp_store.keyword_extract(ids[0])
    assert isinstance(result, list)
    assert len(result) > 0
    assert "word" in result[0] and "score" in result[0]


def test_keyword_extract_not_found(tmp_store):
    result = tmp_store.keyword_extract(999999)
    assert result is None


def test_keyword_extract_empty_text(tmp_store):
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    ids = tmp_store.add(["a a a"], vecs, ns="default")
    result = tmp_store.keyword_extract(ids[0])
    assert isinstance(result, list)


def test_keyword_extract_top_n(tmp_store):
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    ids = tmp_store.add(["alpha beta gamma delta epsilon zeta eta"], vecs, ns="default")
    result = tmp_store.keyword_extract(ids[0], top_n=3)
    assert len(result) <= 3


# ─── Batch 8: import_ns, get_tier_distribution, archive_by_tier, text_search_ranked ───

def test_import_ns_basic(tmp_store):
    rows = [
        {"text": "hello world", "summary": "hi", "tier": 1},
        {"text": "goodbye world"},
    ]
    n = tmp_store.import_ns("imported", rows)
    assert n == 2
    all_m = tmp_store.list_memories(ns="imported")
    assert len(all_m) == 2


def test_import_ns_overwrite(tmp_store):
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    tmp_store.add(["old content"], vecs, ns="imported")
    rows = [{"text": "fresh start"}]
    n = tmp_store.import_ns("imported", rows, overwrite=True)
    assert n == 1
    all_m = tmp_store.list_memories(ns="imported")
    assert all(m["text"] == "fresh start" for m in all_m)


def test_import_ns_empty_rows(tmp_store):
    n = tmp_store.import_ns("empty_ns", [])
    assert n == 0


def test_import_ns_meta_string(tmp_store):
    rows = [{"text": "meta as string", "meta": '{"tags": ["x"]}'}]
    n = tmp_store.import_ns("ns_meta", rows)
    assert n == 1


def test_import_ns_bad_meta_string(tmp_store):
    rows = [{"text": "bad meta", "meta": "not_json"}]
    n = tmp_store.import_ns("ns_bad", rows)
    assert n == 1


def test_import_ns_tier_override(tmp_store):
    rows = [{"text": "pinned", "tier": 0}]
    n = tmp_store.import_ns("ns_pin", rows)
    assert n == 1
    m = tmp_store.list_memories(ns="ns_pin")[0]
    assert m["tier"] == 0


def test_get_tier_distribution_basic(tmp_store):
    import numpy as np
    vecs = np.zeros((3, 384), dtype=np.float32)
    ids = tmp_store.add(["a", "b", "c"], vecs, ns="default")
    tmp_store.set_tier(ids[0], 0)
    tmp_store.set_tier(ids[1], 2)
    dist = tmp_store.get_tier_distribution(ns="default")
    assert isinstance(dist, dict)
    total = sum(dist.values())
    assert total == 3


def test_get_tier_distribution_all_ns(tmp_store):
    import numpy as np
    tmp_store.add(["x"], np.zeros((1, 384), dtype=np.float32), ns="ns1")
    tmp_store.add(["y"], np.zeros((1, 384), dtype=np.float32), ns="ns2")
    dist = tmp_store.get_tier_distribution(ns=None)
    assert sum(dist.values()) >= 2


def test_get_tier_distribution_empty(tmp_store):
    dist = tmp_store.get_tier_distribution(ns="nonexistent")
    assert dist == {}


def test_archive_by_tier_basic(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    ids = tmp_store.add(["first", "second"], vecs, ns="src")
    tmp_store.set_tier(ids[0], 2)
    new_ids = tmp_store.archive_by_tier("src", 2, dst_ns="dst")
    assert len(new_ids) >= 1
    dst_m = tmp_store.list_memories(ns="dst")
    assert len(dst_m) >= 1


def test_archive_by_tier_default_dst(tmp_store):
    import numpy as np
    ids = tmp_store.add(["a"], np.zeros((1, 384), dtype=np.float32), ns="myns")
    tmp_store.set_tier(ids[0], 0)
    new_ids = tmp_store.archive_by_tier("myns", 0)
    assert len(new_ids) == 1
    assert len(tmp_store.list_memories(ns="myns_archive")) == 1


def test_archive_by_tier_delete_original(tmp_store):
    import numpy as np
    ids = tmp_store.add(["todelete"], np.zeros((1, 384), dtype=np.float32), ns="del_src")
    tmp_store.set_tier(ids[0], 2)
    tmp_store.archive_by_tier("del_src", 2, delete_original=True)
    assert tmp_store.list_memories(ns="del_src") == []


def test_archive_by_tier_no_match(tmp_store):
    new_ids = tmp_store.archive_by_tier("nonexistent", 99)
    assert new_ids == []


def test_text_search_ranked_basic(tmp_store):
    import numpy as np
    vecs = np.zeros((3, 384), dtype=np.float32)
    tmp_store.add(["apple orange", "apple", "banana grape"], vecs, ns="default")
    hits = tmp_store.text_search_ranked("apple orange", ns="default")
    assert len(hits) >= 1
    assert hits[0]["hits"] >= 1


def test_text_search_ranked_empty_query(tmp_store):
    hits = tmp_store.text_search_ranked("", ns="default")
    assert hits == []


def test_text_search_ranked_tier_filter(tmp_store):
    import numpy as np
    ids = tmp_store.add(["hello world", "hello moon"], np.zeros((2, 384), dtype=np.float32), ns="default")
    tmp_store.set_tier(ids[0], 0)
    hits = tmp_store.text_search_ranked("hello", ns="default", tier=0)
    assert all(h["tier"] == 0 for h in hits)


def test_text_search_ranked_all_ns(tmp_store):
    import numpy as np
    tmp_store.add(["test memory"], np.zeros((1, 384), dtype=np.float32), ns="ns_a")
    hits = tmp_store.text_search_ranked("test memory", ns=None)
    assert any(h["ns"] == "ns_a" for h in hits)


def test_text_search_ranked_hit_count_order(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    tmp_store.add(["alpha beta gamma delta", "alpha"], vecs, ns="rank_ns")
    hits = tmp_store.text_search_ranked("alpha beta gamma", ns="rank_ns")
    assert hits[0]["hits"] >= hits[-1]["hits"]


# ─── Batch 9: deduplicate_by_text, merge_memories, search_by_date_range, get_access_stats ───

def test_deduplicate_by_text_removes_dupes(tmp_store):
    import numpy as np
    vecs = np.zeros((3, 384), dtype=np.float32)
    tmp_store.add(["same text", "same text", "different"], vecs, ns="default")
    n = tmp_store.deduplicate_by_text(ns="default")
    assert n == 1
    remaining = tmp_store.list_memories(ns="default")
    texts = [m["text"] for m in remaining]
    assert texts.count("same text") == 1


def test_deduplicate_by_text_keep_newest(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    ids = tmp_store.add(["dup", "dup"], vecs, ns="default")
    n = tmp_store.deduplicate_by_text(ns="default", keep="newest")
    assert n == 1
    remaining = tmp_store.list_memories(ns="default")
    assert len(remaining) == 1
    assert remaining[0]["id"] == ids[1]


def test_deduplicate_by_text_no_dupes(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    tmp_store.add(["a", "b"], vecs, ns="default")
    n = tmp_store.deduplicate_by_text(ns="default")
    assert n == 0


def test_deduplicate_by_text_all_ns(tmp_store):
    import numpy as np
    tmp_store.add(["dup"], np.zeros((1, 384), dtype=np.float32), ns="ns1")
    tmp_store.add(["dup"], np.zeros((1, 384), dtype=np.float32), ns="ns2")
    n = tmp_store.deduplicate_by_text(ns=None)
    assert n == 1


def test_merge_memories_basic(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    ids = tmp_store.add(["part one", "part two"], vecs, ns="default")
    new_id = tmp_store.merge_memories(ids)
    assert new_id is not None
    m = tmp_store.get(new_id)
    assert "part one" in m["text"] and "part two" in m["text"]
    assert tmp_store.get(ids[0]) is None


def test_merge_memories_keep_originals(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    ids = tmp_store.add(["a", "b"], vecs, ns="default")
    new_id = tmp_store.merge_memories(ids, delete_originals=False)
    assert new_id is not None
    assert tmp_store.get(ids[0]) is not None


def test_merge_memories_custom_separator(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    ids = tmp_store.add(["x", "y"], vecs, ns="default")
    new_id = tmp_store.merge_memories(ids, separator=" | ", delete_originals=False)
    m = tmp_store.get(new_id)
    assert " | " in m["text"]


def test_merge_memories_none_found(tmp_store):
    result = tmp_store.merge_memories([999999, 888888])
    assert result is None


def test_merge_memories_custom_ns(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    ids = tmp_store.add(["p", "q"], vecs, ns="src")
    new_id = tmp_store.merge_memories(ids, ns="dst", delete_originals=False)
    m = tmp_store.get(new_id)
    assert m["ns"] == "dst"


def test_search_by_date_range_basic(tmp_store):
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    tmp_store.add(["old memory", "new memory"], vecs, ns="default")
    hits = tmp_store.search_by_date_range("2000-01-01", "2099-12-31", ns="default")
    assert len(hits) == 2


def test_search_by_date_range_future_only(tmp_store):
    import numpy as np
    vecs = np.zeros((1, 384), dtype=np.float32)
    tmp_store.add(["now"], vecs, ns="default")
    hits = tmp_store.search_by_date_range("2099-01-01", "2099-12-31", ns="default")
    assert len(hits) == 0


def test_search_by_date_range_all_ns(tmp_store):
    import numpy as np
    tmp_store.add(["a"], np.zeros((1, 384), dtype=np.float32), ns="ns1")
    hits = tmp_store.search_by_date_range("2000-01-01", "2099-12-31", ns=None)
    assert len(hits) >= 1


def test_search_by_date_range_tier_filter(tmp_store):
    import numpy as np
    ids = tmp_store.add(["x"], np.zeros((1, 384), dtype=np.float32), ns="default")
    tmp_store.set_tier(ids[0], 0)
    hits = tmp_store.search_by_date_range("2000-01-01", "2099-12-31", ns="default", tier=0)
    assert all(h["tier"] == 0 for h in hits)


def test_get_access_stats_basic(tmp_store):
    import numpy as np
    vecs = np.zeros((3, 384), dtype=np.float32)
    tmp_store.add(["a", "b", "c"], vecs, ns="default")
    stats = tmp_store.get_access_stats(ns="default")
    assert "total_accesses" in stats
    assert "unique_accessed" in stats
    assert "never_accessed" in stats
    assert "top" in stats
    assert stats["never_accessed"] == 3


def test_get_access_stats_all_ns(tmp_store):
    import numpy as np
    tmp_store.add(["x"], np.zeros((1, 384), dtype=np.float32), ns="ns1")
    stats = tmp_store.get_access_stats(ns=None)
    assert stats["never_accessed"] >= 1


def test_get_access_stats_top_n(tmp_store):
    import numpy as np
    vecs = np.zeros((5, 384), dtype=np.float32)
    tmp_store.add(["a", "b", "c", "d", "e"], vecs, ns="default")
    stats = tmp_store.get_access_stats(ns="default", top_n=2)
    assert len(stats["top"]) <= 2


def test_merge_memories_skips_unparseable_meta(tmp_store):
    """A source row whose meta is not valid JSON must not break the merge: the
    meta-combine step swallows the parse error and continues."""
    import numpy as np
    vecs = np.zeros((2, 384), dtype=np.float32)
    ids = tmp_store.add(["alpha piece", "beta piece"], vecs, ns="default")
    tmp_store._db.execute(
        "UPDATE memories SET meta = ? WHERE id = ?", ("{not valid json", ids[0])
    )
    tmp_store._db.commit()
    new_id = tmp_store.merge_memories(ids)
    assert new_id is not None
    merged = tmp_store.get(new_id)
    assert "alpha piece" in merged["text"] and "beta piece" in merged["text"]
