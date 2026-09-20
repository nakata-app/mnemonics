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


def test_server_warmup_primes_realistic_batch(monkeypatch):
    from mnemonics import server

    seen: list[list[str]] = []
    searches: list[tuple[str, dict]] = []

    class FakeEncoder:
        def encode(self, texts, **kwargs):
            batch = list(texts)
            seen.append(batch)
            assert kwargs["batch_size"] == 4
            return np.zeros((len(batch), 1024), dtype="float32")

    class FakeStore:
        root = "/tmp/fake"

        def warm_namespace(self, ns):
            assert ns == "sessions"
            return 42

        def search(self, vector, **kwargs):
            searches.append(("vector", kwargs))
            assert kwargs["touch"] is False
            return []

        def search_bm25(self, query, **kwargs):
            searches.append(("bm25", kwargs))
            assert query == "provider timeout cleanup"
            return []

    monkeypatch.setattr(server, "_get_store", lambda: FakeStore())
    monkeypatch.setattr(
        server,
        "_resolve_model_for_store",
        lambda model, store: "/model",
    )
    monkeypatch.setattr(server, "_get_encoder", lambda model: FakeEncoder())

    result = server._warm_store("sessions")

    assert len(seen) == 1
    assert len(seen[0]) == 4
    assert "src/agent/provider-attempt.ts" in seen[0]
    assert [kind for kind, _ in searches] == ["vector", "bm25"]
    assert result["dim"] == 1024
    assert result["count"] == 42
