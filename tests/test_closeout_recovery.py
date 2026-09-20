import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mnemonics.store import Store
from tests.test_server import http_call

ingest = importlib.import_module("mnemonics.ingest")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def fake_fastembed(monkeypatch, sources, registry=()):
    embedding = SimpleNamespace(
        list_supported_models=lambda: [{"model": MODEL, "sources": sources}],
        EMBEDDINGS_REGISTRY=registry,
    )
    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=embedding))
    return embedding


@pytest.mark.parametrize("sources", [{}, {"hf": "bad repo"}])
def test_invalid_cache_sources_never_choose_a_deletion_path(monkeypatch, tmp_path, sources):
    fake_fastembed(monkeypatch, sources)
    assert ingest._fastembed_hf_cache_dir(MODEL, str(tmp_path)) is None


def test_registry_errors_fail_closed(monkeypatch, tmp_path):
    embedding = fake_fastembed(monkeypatch, {})

    def broken():
        raise RuntimeError("registry unavailable")

    embedding.list_supported_models = broken
    assert ingest._fastembed_hf_cache_dir(MODEL, str(tmp_path)) is None
    assert ingest._fastembed_gcs_cache_dir(MODEL, str(tmp_path)) is None


def test_invalid_gcs_leaf_and_missing_purge_target(monkeypatch, tmp_path):
    embedding = fake_fastembed(monkeypatch, {})
    embedding.list_supported_models = lambda: [
        {"model": "owner/unsafe leaf", "sources": {"url": "https://example.invalid/model"}}
    ]
    monkeypatch.setattr(ingest, "_fastembed_model_name", lambda _: "owner/unsafe leaf")
    assert ingest._fastembed_gcs_cache_dir(MODEL, str(tmp_path)) is None
    assert not ingest._purge_fastembed_model_cache(MODEL, str(tmp_path))
    fake_fastembed(monkeypatch, {"url": "https://example.invalid/model"})
    monkeypatch.setattr(ingest, "_fastembed_model_name", lambda _: "owner/unsafe leaf")
    assert ingest._fastembed_gcs_cache_dir(MODEL, str(tmp_path)) is None


def test_cache_probe_and_purge_permission_failures(monkeypatch, tmp_path):
    candidate = tmp_path / "model-cache"
    candidate.mkdir()
    monkeypatch.setattr(ingest, "_fastembed_hf_cache_dir", lambda *_: candidate)

    def denied(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "rglob", denied)
    assert not ingest._fastembed_cache_looks_corrupt(
        MODEL, str(tmp_path), RuntimeError("unrelated")
    )
    assert ingest._fastembed_cache_looks_corrupt(
        MODEL, str(tmp_path), RuntimeError("model.onnx failed")
    )
    monkeypatch.setattr(ingest.shutil, "rmtree", denied)
    assert not ingest._purge_fastembed_model_cache(MODEL, str(tmp_path))
    assert candidate.exists()


def test_gcs_registry_fallbacks_are_bounded(monkeypatch, tmp_path):
    def unsupported(_):
        raise ValueError("unsupported")

    missing = SimpleNamespace(_get_model_description=unsupported)
    fake_fastembed(monkeypatch, {}, [missing])
    assert ingest._recover_fastembed_from_gcs(MODEL, str(tmp_path)) is None
    no_url = SimpleNamespace(
        _get_model_description=lambda _: SimpleNamespace(sources=SimpleNamespace(url=None))
    )
    fake_fastembed(monkeypatch, {}, [missing, no_url])
    assert ingest._recover_fastembed_from_gcs(MODEL, str(tmp_path)) is None

    def failed(_):
        raise OSError("failed")

    fake_fastembed(monkeypatch, {}, [SimpleNamespace(_get_model_description=failed)])
    assert ingest._recover_fastembed_from_gcs(MODEL, str(tmp_path)) is None


@pytest.mark.parametrize("recover", [True, False])
def test_fastembed_repairs_failure_created_during_load(monkeypatch, tmp_path, recover):
    candidate = tmp_path / "models--owner--model"
    recovered = tmp_path / "recovered"
    calls = []

    class Embedding:
        @staticmethod
        def list_supported_models():
            return [{"model": MODEL, "sources": {"hf": "owner/model"}, "dim": 384}]

        def __init__(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                candidate.mkdir()
                raise RuntimeError("model.onnx failed")
            assert not candidate.exists()
            assert kwargs["specific_model_path"] == str(recovered)

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=Embedding))
    monkeypatch.setenv("MNEMONICS_FASTEMBED_CACHE", str(tmp_path))
    monkeypatch.setenv("MNEMONICS_FASTEMBED_REPAIR", "1")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(
        ingest, "_recover_fastembed_from_gcs", lambda *_: recovered if recover else None
    )
    if recover:
        assert ingest._FastEmbedEncoder(MODEL).get_sentence_embedding_dimension() == 384
        assert len(calls) == 2
    else:
        with pytest.raises(RuntimeError, match="model.onnx failed"):
            ingest._FastEmbedEncoder(MODEL)
        assert len(calls) == 1


@pytest.mark.parametrize(
    "keys", [[1], [""], ["x" * 257], ["x\nx"], [f"key-{i}" for i in range(17)]]
)
def test_invalid_retire_keys_leave_store_unchanged(tmp_path, keys):
    store = Store(tmp_path, dim=2)
    with pytest.raises(ValueError, match="retire_canonical_keys"):
        store.add_canonical(
            "fact",
            np.ones(2, dtype="float32"),
            ns="default",
            canonical_key="fact",
            retire_canonical_keys=keys,
        )
    assert store.export_ns("default") == []


def test_canonical_http_rejects_nonstring_summary(tmp_store):
    status, response = http_call(
        tmp_store, "POST", "/ingest", {"texts": ["fact"], "canonical_key": "fact", "summary": 42}
    )
    assert status == 400
    assert "summary must be a string" in response["error"]


def test_feedback_repairs_invalid_metadata(tmp_path):
    store = Store(tmp_path, dim=2)
    ids = store.add(["fact"], np.ones((1, 2), dtype="float32"))
    store._db.execute("UPDATE memories SET meta='broken'")
    store._db.commit()
    assert store.record_retrieval_feedback(ids, success=True) == {"updated": 1, "missing": 0}
    assert store.export_ns("default")[0]["meta"]["retrieval_success_count"] == 1


def test_search_honors_reported_connection_variable_limit(monkeypatch, tmp_path):
    store_mod = importlib.import_module("mnemonics.store")
    store = Store(tmp_path, dim=2)
    vectors = np.array([[1, value] for value in range(12)], dtype="float32")
    ids = store.add([str(i) for i in range(12)], vectors)
    for mid in ids[:9]:
        store.update_meta_key(mid, "status", "superseded")
    database = store._db

    class LimitedConnection:
        def __getattr__(self, name):
            return getattr(database, name)

        def getlimit(self, category):
            assert category == store_mod.sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER
            return 8

        def execute(self, sql, params=()):
            assert len(params) <= 8
            return database.execute(sql, params)

    monkeypatch.setattr(store_mod.sqlite3, "SQLITE_LIMIT_VARIABLE_NUMBER", 9, raising=False)
    store._db = LimitedConnection()
    assert len(store.search(vectors[0], top_k=3, min_tier=1, max_tier=1)) == 3


def test_canonical_retirement_is_idempotent_under_duplicate_cursor_rows(tmp_path):
    store = Store(tmp_path, dim=2)
    vector = np.ones(2, dtype="float32")
    prior = store.add_canonical("old", vector, ns="default", canonical_key="slot")
    database = store._db

    class DuplicateCursorConnection:
        def __getattr__(self, name):
            return getattr(database, name)

        def execute(self, sql, params=()):
            cursor = database.execute(sql, params)
            if sql.startswith("SELECT id, text, meta FROM memories WHERE ns=?"):
                rows = cursor.fetchall()
                return SimpleNamespace(fetchall=lambda: rows + rows)
            return cursor

    store._db = DuplicateCursorConnection()
    result = store.add_canonical("new", vector, ns="default", canonical_key="slot")
    assert result["superseded"] == [prior["id"]]
    rows = store.export_ns("default")
    assert len(rows) == 2
    assert sum(row["meta"]["status"] == "active" for row in rows) == 1
