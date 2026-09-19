from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import mnemonics.dedup as dedup_mod
import mnemonics.ingest as ingest_mod
import mnemonics.retrieve as retrieve_mod
from mnemonics.lifecycle import canonical_ingest
from mnemonics.query_plan import PlannedQuery, _dedupe, _scope_factor, retrieve_planned
from mnemonics.store import Store


def test_resolve_model_for_store_survives_manifest_read_error(tmp_path, monkeypatch):
    from mnemonics import embed_manifest
    monkeypatch.delenv("MNEMONICS_ENCODER_MODEL", raising=False)
    monkeypatch.delenv("MNEMONICS_ADAPTMEM_PATH", raising=False)
    monkeypatch.setattr(embed_manifest, "read", lambda root: (_ for _ in ()).throw(OSError("boom")))
    store = SimpleNamespace(root=tmp_path)
    assert ingest_mod._resolve_model_for_store("all-MiniLM-L6-v2", store) == "all-MiniLM-L6-v2"


def test_onnx_encoder_init_dimension_and_encode(monkeypatch):
    class FakeSession:
        def __init__(self, path, providers):
            assert path.endswith("model.onnx")
            assert providers == ["CPUExecutionProvider"]
        def get_outputs(self):
            return [SimpleNamespace(shape=[None, 3])]
        def run(self, _, feeds):
            n = feeds["input_ids"].shape[0]
            return [np.ones((n, 3), dtype="float32")]

    class FakeTokenizer:
        @classmethod
        def from_file(cls, path):
            assert path.endswith("model/tokenizer.json")
            return cls()
        def enable_truncation(self, max_length):
            assert max_length == 512
        def token_to_id(self, token):
            assert token == "[PAD]"
            return 0
        def enable_padding(self, pad_id, pad_token):
            assert (pad_id, pad_token) == (0, "[PAD]")
        def encode_batch(self, texts):
            return [
                SimpleNamespace(ids=[1, 2], attention_mask=[1, 1])
                for _ in texts
            ]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=FakeSession))
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=FakeTokenizer))

    enc = ingest_mod._OnnxEmbeddingEncoder("/tmp/model")
    assert enc.get_sentence_embedding_dimension() == 3
    assert enc.encode([]).shape == (0, 384)
    out = enc.encode(["a", "b"], batch_size=1)
    assert out.shape == (2, 3)
    assert out.dtype == np.float32


def test_fastembed_encoder_encode_path(monkeypatch, tmp_path):
    class FakeTextEmbedding:
        def __init__(self, model_name, cache_dir):
            self.model_name = model_name
        @classmethod
        def list_supported_models(cls):
            return [{"model": "custom", "dim": 2}]
        def embed(self, texts, batch_size=256):
            assert list(texts) == ["a", "b"]
            assert batch_size == 4
            return [np.array([1.0, 0.0]), np.array([0.0, 1.0])]

    monkeypatch.setenv("MNEMONICS_FASTEMBED_CACHE", str(tmp_path))
    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    enc = ingest_mod._FastEmbedEncoder("custom")
    assert enc.get_sentence_embedding_dimension() == 2
    out = enc.encode(["a", "b"], batch_size=4)
    assert out.shape == (2, 2)
    assert out.dtype == np.float32


def test_build_encoder_local_onnx_and_local_torch_fallback(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"x")
    fake_onnx = object()
    monkeypatch.setattr(ingest_mod, "_OnnxEmbeddingEncoder", lambda path: fake_onnx)
    assert ingest_mod._build_encoder(str(model)) is fake_onnx

    onnx.unlink()
    class FakeST:
        def __init__(self, path):
            self.path = path
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeST))
    built = ingest_mod._build_encoder(str(model))
    assert isinstance(built, FakeST)
    assert built.path == str(model)


def test_canonical_ingest_happy_path_and_empty(tmp_path, monkeypatch):
    store = Store(path=tmp_path, dim=3)
    fake_encoder = SimpleNamespace(
        encode=lambda texts, **kwargs: np.asarray([[1.0, 0.0, 0.0]], dtype="float32")
    )
    monkeypatch.setattr("mnemonics.lifecycle._resolve_model_for_store", lambda model, store: "resolved")
    monkeypatch.setattr("mnemonics.lifecycle._get_encoder", lambda model: fake_encoder)

    result = canonical_ingest(
        "branch head is abc",
        store,
        canonical_key="repo:head",
        ns="project",
        summary="head",
        meta={"kind": "fact"},
        tier=0,
    )
    assert result["action"] == "add"
    assert result["canonical_key"] == "repo:head"
    assert result["encoder"] == "resolved"
    with pytest.raises(ValueError, match="text must not be empty"):
        canonical_ingest("   ", store, canonical_key="x")


def test_query_plan_uncovered_edges(monkeypatch):
    assert _dedupe([
        PlannedQuery("", 1.0, "x"),
        PlannedQuery("Same", 1.0, "x"),
        PlannedQuery(" same ", 0.5, "y"),
    ], 4) == [PlannedQuery("Same", 1.0, "x")]
    assert _scope_factor({}, None) == 1.0
    assert _scope_factor({}, ["", "  "]) == 1.0
    assert _scope_factor({"meta": "bad"}, ["repo"]) == 1.0
    assert _scope_factor({"meta": {"project": "/x/repo"}}, ["/other/repo"]) == 1.25
    assert retrieve_planned("", SimpleNamespace(), max_queries=0) == {"results": [], "queries": []}

    class Enc:
        def encode(self, texts, **kwargs):
            return np.asarray([[1.0, 0.0, 0.0]], dtype="float32")
    class S:
        def touch_ids(self, ids):
            return len(ids)
    monkeypatch.setattr("mnemonics.query_plan._resolve_model_for_store", lambda model, store: "m")
    monkeypatch.setattr("mnemonics.query_plan._get_encoder", lambda model: Enc())
    monkeypatch.setattr(
        "mnemonics.query_plan.retrieve",
        lambda **kwargs: {"results": [{"id": 1, "score": 1.0, "text": "x", "tier": 1, "meta": {}}]},
    )
    monkeypatch.setattr(
        "mnemonics.query_plan._ce_rerank",
        lambda query, rows, top_k: [{**rows[0], "ce_score": 9.0}],
    )
    result = retrieve_planned("hello world", S(), top_k=1, rerank=True, project_hints=["repo"])
    assert result["results"][0]["ce_score"] == 9.0
    assert result["results"][0]["scope_boost"] == 1.0


def test_retrieve_knobs_query_vector_and_full_band(monkeypatch):
    class CE:
        def predict(self, pairs, **kwargs):
            assert kwargs["batch_size"] == 2
            return [0.2, 0.8]
    monkeypatch.setattr(retrieve_mod, "_rerank_ce", CE())
    monkeypatch.setattr(retrieve_mod, "_rerank_model_name", "ce")
    monkeypatch.setenv("MNEMONICS_RERANK_MODEL", "ce")
    monkeypatch.setenv("MNEMONICS_RERANK_BATCH_SIZE", "2")
    rows = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]
    ranked = retrieve_mod._ce_rerank("q", rows, top_k=1)
    assert ranked[0]["id"] == 2

    class FakeStore:
        root = "/tmp/fake"
        dim = 3
        def search(self, vector, **kwargs):
            return [
                {"id": 1, "text": "one", "created": "2026-09-19 00:00:00", "tier": 1, "access_count": 0, "score": 0.5},
                {"id": 2, "text": "two", "created": "2026-09-19 00:00:00", "tier": 1, "access_count": 0, "score": 0.9},
            ]
        def search_bm25(self, *args, **kwargs):
            return []

    monkeypatch.setattr(retrieve_mod, "_resolve_model_for_store", lambda model, store: "m")
    monkeypatch.setattr("mnemonics.embed_manifest.verify", lambda root, live: ("ok", ""))
    monkeypatch.setattr("mnemonics.embed_manifest.encoder_fingerprint", lambda model, dim: {"dim": dim})
    retrieve_mod.retrieve.__dict__.pop("_enc_checked", None)
    monkeypatch.setenv("MNEMONICS_SCORE_FULL_BAND", "1")
    out = retrieve_mod.retrieve(
        "q",
        FakeStore(),
        top_k=1,
        candidate_k=2,
        hybrid=False,
        decay=False,
        query_vector=np.asarray([1.0, 0.0, 0.0], dtype="float32"),
    )
    assert len(out["results"]) == 1
    assert out["results"][0]["id"] == 2


def test_rerank_max_length_is_forwarded(monkeypatch):
    seen = {}
    class FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            seen["name"] = name
            seen["kwargs"] = kwargs
    monkeypatch.setattr(retrieve_mod, "_rerank_ce", None)
    monkeypatch.setattr(retrieve_mod, "_rerank_model_name", None)
    monkeypatch.setenv("MNEMONICS_RERANK_MODEL", "ce")
    monkeypatch.setenv("MNEMONICS_RERANK_MAX_LENGTH", "256")
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    retrieve_mod._get_rerank_ce()
    assert seen == {"name": "ce", "kwargs": {"max_length": 256}}


def test_store_edge_branches(tmp_path, monkeypatch):
    from mnemonics import embed_manifest
    monkeypatch.setattr(embed_manifest, "read", lambda root: (_ for _ in ()).throw(OSError("bad")))
    monkeypatch.delenv("MNEMONICS_DIM", raising=False)
    assert Store._resolve_dim(None, tmp_path) == 384

    store = Store(path=tmp_path / "s", dim=3)
    assert store.touch_ids([]) == 0
    with pytest.raises(ValueError, match="tier"):
        store.add_canonical("x", np.asarray([1, 0, 0], dtype="float32"), ns="n", canonical_key="k", tier=9)


def test_store_corrupt_reload_and_canonical_resize(tmp_path, monkeypatch):
    store = Store(path=tmp_path, dim=3)
    idx_path = tmp_path / "index_bad.bin"
    idx_path.write_bytes(b"corrupt")

    class BadIndex:
        def load_index(self, path):
            raise RuntimeError("corrupt")
    monkeypatch.setattr("mnemonics.store.hnswlib.Index", lambda **kwargs: BadIndex())
    store._reload_if_stale("bad")
    assert not idx_path.exists()

    store2 = Store(path=tmp_path / "resize", dim=3)
    class TinyIndex:
        def __init__(self):
            self.resized = False
        def get_current_count(self):
            return 1
        def get_max_elements(self):
            return 1
        def resize_index(self, size):
            self.resized = True
            assert size >= 4
        def add_items(self, vec, ids):
            pass
        def save_index(self, path):
            open(path, "wb").write(b"x")
    tiny = TinyIndex()
    monkeypatch.setattr(store2, "_index_for", lambda ns: tiny)
    result = store2.add_canonical(
        "new canonical",
        np.asarray([1, 0, 0], dtype="float32"),
        ns="n",
        canonical_key="key",
    )
    assert result["action"] == "add"
    assert tiny.resized is True

def test_reconcile_ingest_tolerates_empty_ingest_result(monkeypatch):
    store = SimpleNamespace()
    monkeypatch.setattr(dedup_mod, "find_similar", lambda *args, **kwargs: [])
    monkeypatch.setattr(dedup_mod, "ingest", lambda *args, **kwargs: [])
    result = dedup_mod.reconcile_ingest(["x"], store)
    assert result == {
        "added": [],
        "noop_skipped": [],
        "superseded": [],
        "supersede_failed": [],
    }

