from __future__ import annotations

import json

import numpy as np

from mnemonics.store import Store


def vec(x: float, y: float, z: float) -> np.ndarray:
    return np.asarray([x, y, z], dtype="float32")


def test_canonical_upsert_archives_prior_value(tmp_path):
    store = Store(path=tmp_path, dim=3)
    first = store.add_canonical(
        "branch head is abc",
        vec(1, 0, 0),
        ns="project",
        canonical_key="repo:branch-head",
    )
    second = store.add_canonical(
        "branch head is def",
        vec(0.99, 0.01, 0),
        ns="project",
        canonical_key="repo:branch-head",
    )

    assert first["action"] == "add"
    assert second["action"] == "update"
    assert second["superseded"] == [first["id"]]

    old_meta = json.loads(store._db.execute("SELECT meta FROM memories WHERE id=?", (first["id"],)).fetchone()[0])
    new_meta = json.loads(store._db.execute("SELECT meta FROM memories WHERE id=?", (second["id"],)).fetchone()[0])
    assert old_meta["status"] == "superseded"
    assert old_meta["superseded_by"] == second["id"]
    assert old_meta["valid_until"]
    assert new_meta["status"] == "active"
    assert new_meta["supersedes"] == [first["id"]]

    visible = store.search(vec(1, 0, 0), ns="project", top_k=10)
    assert first["id"] not in {row["id"] for row in visible}
    assert second["id"] in {row["id"] for row in visible}


def test_canonical_exact_reingest_is_idempotent(tmp_path):
    store = Store(path=tmp_path, dim=3)
    first = store.add_canonical(
        "current config is v2",
        vec(1, 0, 0),
        ns="project",
        canonical_key="config:version",
    )
    second = store.add_canonical(
        "current config is v2",
        vec(1, 0, 0),
        ns="project",
        canonical_key="config:version",
    )

    assert second == {"action": "noop", "id": first["id"], "superseded": []}
    count = store._db.execute(
        "SELECT COUNT(*) FROM memories WHERE ns='project'"
    ).fetchone()[0]
    assert count == 1


def test_canonical_reingest_repairs_duplicate_active_rows(tmp_path):
    store = Store(path=tmp_path, dim=3)
    ids = store.add(
        ["same fact", "same fact"],
        np.asarray([[1, 0, 0], [1, 0, 0]], dtype="float32"),
        ns="project",
        meta=[
            {"canonical_key": "fact:x", "status": "active"},
            {"canonical_key": "fact:x", "status": "active"},
        ],
    )
    result = store.add_canonical(
        "same fact",
        vec(1, 0, 0),
        ns="project",
        canonical_key="fact:x",
    )

    assert result["action"] == "consolidate"
    assert result["id"] == max(ids)
    assert result["superseded"] == [min(ids)]
    loser_meta = json.loads(store._db.execute("SELECT meta FROM memories WHERE id=?", (min(ids),)).fetchone()[0])
    assert loser_meta["status"] == "superseded"
    assert loser_meta["superseded_by"] == max(ids)


def test_canonical_key_validation_and_vector_dimension(tmp_path):
    store = Store(path=tmp_path, dim=3)
    try:
        store.add_canonical("x", vec(1, 0, 0), ns="p", canonical_key="")
    except ValueError as exc:
        assert "canonical_key" in str(exc)
    else:
        raise AssertionError("empty canonical key must fail")

    try:
        store.add_canonical(
            "x",
            np.asarray([1, 0], dtype="float32"),
            ns="p",
            canonical_key="fact:x",
        )
    except ValueError as exc:
        assert "vector dim" in str(exc)
    else:
        raise AssertionError("wrong vector dimension must fail")
