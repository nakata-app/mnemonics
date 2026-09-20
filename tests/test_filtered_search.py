import sqlite3

import numpy as np
import pytest

from mnemonics.store import Store


@pytest.fixture
def search_store(tmp_path):
    store = Store(tmp_path, dim=2)
    angles = np.linspace(0, 1, 12)
    vectors = np.column_stack((np.cos(angles), np.sin(angles))).astype("float32")
    ids = store.add([f"memory {i}" for i in range(12)], vectors)
    yield store, ids, vectors[0]
    store._db.close()


def test_superseded_candidates_do_not_consume_result_slots(search_store):
    store, ids, query = search_store
    for mid in ids[:9]:
        store.update_meta_key(mid, "status", "superseded")
    hits = store.search(query, top_k=3)
    assert [hit["id"] for hit in hits] == ids[9:]
    assert store._db.execute("SELECT SUM(access_count) FROM memories").fetchone()[0] == 3
    assert [hit["id"] for hit in store.search(query, top_k=3, exclude_superseded=False)] == ids[:3]


def test_tier_filter_expands_without_touching_candidates(search_store):
    store, ids, query = search_store
    store._db.executemany("UPDATE memories SET tier=0 WHERE id=?", [(mid,) for mid in ids[8:]])
    store._db.commit()
    hits = store.search(query, top_k=3, min_tier=0, max_tier=0, touch=False)
    assert [hit["id"] for hit in hits] == ids[8:11]
    assert store._db.execute("SELECT SUM(access_count) FROM memories").fetchone()[0] == 0
    assert store.search(query, top_k=3, min_tier=2) == []
    assert store.search(query, top_k=0) == []


def test_deleted_labels_do_not_erase_surviving_filtered_results(search_store):
    store, ids, query = search_store
    for mid in ids[:5]:
        store.delete(mid)
    for mid in ids[5:9]:
        store.update_meta_key(mid, "status", "superseded")
    hits = store.search(query, top_k=5)
    assert [hit["id"] for hit in hits] == ids[9:]
    assert len(store.search(query, top_k=20, exclude_superseded=False)) == 7


def test_fully_deleted_index_returns_no_results(search_store):
    store, ids, query = search_store
    store.delete_many(ids)
    assert store.search(query, top_k=3) == []


def test_candidate_expansion_respects_sqlite_bind_limit(tmp_path):
    store = Store(tmp_path, dim=2)
    db = store._db
    limit = 8 if hasattr(db, "setlimit") else 900
    count = limit + 3
    angles = np.linspace(0, 1, count)
    vectors = np.column_stack((np.cos(angles), np.sin(angles))).astype("float32")
    ids = store.add([f"memory {i}" for i in range(count)], vectors)
    db.executemany("UPDATE memories SET tier=0 WHERE id=?", [(mid,) for mid in ids[-3:]])
    db.commit()
    if hasattr(db, "setlimit"):
        db.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, limit)

    class LimitedConnection:
        def __init__(self):
            self.candidate_bind_counts = []

        def __getattr__(self, name):
            return getattr(db, name)

        def execute(self, sql, params=()):
            # Python 3.10 lacks setlimit; enforce the fallback budget while
            # still executing every accepted query against the real database.
            if len(params) > limit:
                raise sqlite3.OperationalError("too many SQL variables")
            if sql.startswith("SELECT id, text, summary"):
                self.candidate_bind_counts.append(len(params))
            return db.execute(sql, params)

    store._db = limited = LimitedConnection()
    try:
        hits = store.search(vectors[0], top_k=3, min_tier=0, max_tier=0)
        assert [hit["id"] for hit in hits] == ids[-3:]
        assert max(limited.candidate_bind_counts) == limit
        assert db.execute("SELECT SUM(access_count) FROM memories").fetchone()[0] == 3
    finally:
        db.close()
