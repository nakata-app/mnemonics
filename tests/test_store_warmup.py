from __future__ import annotations

import numpy as np

from mnemonics.store import Store


def test_warm_namespace_loads_index_without_touching_access_counters(tmp_path):
    store = Store(path=tmp_path, dim=3)
    store.add(
        ["alpha memory"],
        np.asarray([[1.0, 0.0, 0.0]], dtype="float32"),
        ns="warm",
    )
    row_id = store._db.execute(
        "SELECT id FROM memories WHERE ns = 'warm' LIMIT 1"
    ).fetchone()[0]
    before = store._db.execute(
        "SELECT access_count, last_accessed FROM memories WHERE id = ?",
        (row_id,),
    ).fetchone()

    assert store.warm_namespace("warm") == 1

    after = store._db.execute(
        "SELECT access_count, last_accessed FROM memories WHERE id = ?",
        (row_id,),
    ).fetchone()
    assert after == before
