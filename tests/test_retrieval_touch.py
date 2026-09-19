from __future__ import annotations

import numpy as np

from mnemonics.retrieve import retrieve
from mnemonics.store import Store


def test_search_touch_false_does_not_reinforce_candidates(tmp_path):
    store = Store(path=tmp_path, dim=3)
    ids = store.add(
        ["alpha", "beta"],
        np.asarray([[1, 0, 0], [0.9, 0.1, 0]], dtype="float32"),
        ns="t",
    )

    rows = store.search(
        np.asarray([1, 0, 0], dtype="float32"),
        ns="t",
        top_k=2,
        touch=False,
    )
    assert {row["id"] for row in rows} == set(ids)

    counts = dict(
        store._db.execute(
            "SELECT id, access_count FROM memories WHERE id IN (?, ?)",
            (ids[0], ids[1]),
        ).fetchall()
    )
    assert counts == {ids[0]: 0, ids[1]: 0}


def test_touch_ids_reinforces_each_final_row_once(tmp_path):
    store = Store(path=tmp_path, dim=3)
    ids = store.add(
        ["alpha", "beta"],
        np.asarray([[1, 0, 0], [0.9, 0.1, 0]], dtype="float32"),
        ns="t",
    )

    assert store.touch_ids([ids[0], ids[0], ids[1]]) == 2
    counts = dict(
        store._db.execute(
            "SELECT id, access_count FROM memories WHERE id IN (?, ?)",
            (ids[0], ids[1]),
        ).fetchall()
    )
    assert counts == {ids[0]: 1, ids[1]: 1}


def test_record_retrieval_feedback_tracks_success_and_failure(tmp_path):
    import json

    store = Store(path=tmp_path, dim=3)
    ids = store.add(
        ["alpha", "beta"],
        np.asarray([[1, 0, 0], [0, 1, 0]], dtype="float32"),
        ns="t",
    )

    assert store.record_retrieval_feedback(
        [ids[0], ids[0], ids[1], 999999],
        success=True,
    ) == {"updated": 2, "missing": 1}
    assert store.record_retrieval_feedback([ids[0]], success=False) == {
        "updated": 1,
        "missing": 0,
    }

    rows = store._db.execute(
        "SELECT id, meta FROM memories WHERE id IN (?, ?) ORDER BY id",
        (ids[0], ids[1]),
    ).fetchall()
    metas = {row_id: json.loads(meta) for row_id, meta in rows}
    assert metas[ids[0]]["retrieval_success_count"] == 1
    assert metas[ids[0]]["retrieval_failure_count"] == 1
    assert metas[ids[0]]["retrieval_feedback_last_outcome"] == "failure"
    assert metas[ids[0]]["retrieval_feedback_last_at"]
    assert metas[ids[1]]["retrieval_success_count"] == 1
    assert "retrieval_failure_count" not in metas[ids[1]]


def test_record_retrieval_feedback_empty_is_noop(tmp_path):
    store = Store(path=tmp_path, dim=3)
    assert store.record_retrieval_feedback([], success=True) == {
        "updated": 0,
        "missing": 0,
    }


def test_retrieve_touches_only_final_rows_once(tmp_path):
    store = Store(path=tmp_path, dim=3)
    ids = store.add(
        ["alpha one", "alpha two", "alpha three"],
        np.asarray(
            [[1.0, 0.0, 0.0], [0.99, 0.1, 0.0], [0.98, 0.2, 0.0]],
            dtype="float32",
        ),
        ns="t",
    )

    result = retrieve(
        "alpha",
        store,
        ns="t",
        top_k=1,
        candidate_k=3,
        hybrid=True,
        decay=False,
        query_vector=np.asarray([1.0, 0.0, 0.0], dtype="float32"),
        touch=True,
    )

    assert len(result["results"]) == 1
    winner = result["results"][0]["id"]
    counts = dict(
        store._db.execute(
            "SELECT id, access_count FROM memories WHERE id IN (?, ?, ?)",
            tuple(ids),
        ).fetchall()
    )
    assert counts[winner] == 1
    assert sum(counts.values()) == 1


def test_retrieve_touch_false_leaves_final_rows_untouched(tmp_path):
    store = Store(path=tmp_path, dim=3)
    ids = store.add(
        ["alpha one", "alpha two"],
        np.asarray([[1.0, 0.0, 0.0], [0.99, 0.1, 0.0]], dtype="float32"),
        ns="t",
    )

    retrieve(
        "alpha",
        store,
        ns="t",
        top_k=1,
        candidate_k=2,
        hybrid=True,
        decay=False,
        query_vector=np.asarray([1.0, 0.0, 0.0], dtype="float32"),
        touch=False,
    )

    counts = dict(
        store._db.execute(
            "SELECT id, access_count FROM memories WHERE id IN (?, ?)",
            tuple(ids),
        ).fetchall()
    )
    assert counts == {ids[0]: 0, ids[1]: 0}