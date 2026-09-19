from __future__ import annotations

import numpy as np

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