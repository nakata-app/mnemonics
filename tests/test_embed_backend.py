from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import mnemonics.ingest as ingest_mod


class FakeSentenceTransformer:
    def __init__(self, model_name: str):
        self.model_name = model_name


def test_sentence_transformers_backend_can_be_pinned(monkeypatch):
    monkeypatch.setenv("MNEMONICS_EMBED_BACKEND", "sentence-transformers")
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    encoder = ingest_mod._build_encoder("all-MiniLM-L6-v2")
    assert isinstance(encoder, FakeSentenceTransformer)
    assert encoder.model_name == "all-MiniLM-L6-v2"


def test_invalid_embed_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("MNEMONICS_EMBED_BACKEND", "mystery")
    with pytest.raises(ValueError, match="MNEMONICS_EMBED_BACKEND"):
        ingest_mod._build_encoder("all-MiniLM-L6-v2")


def test_get_encoder_initialization_is_thread_safe(monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor

    built = object()
    build_calls = 0

    def fake_build(_resolved: str):
        nonlocal build_calls
        build_calls += 1
        time.sleep(0.03)
        return built

    monkeypatch.setattr(ingest_mod, "_encoder", None)
    monkeypatch.setattr(ingest_mod, "_encoder_name", "all-MiniLM-L6-v2")
    monkeypatch.setattr(ingest_mod, "_build_encoder", fake_build)
    monkeypatch.delenv("MNEMONICS_EMBED_BACKEND", raising=False)

    with ThreadPoolExecutor(max_workers=8) as pool:
        encoders = list(
            pool.map(
                lambda _i: ingest_mod._get_encoder("thread-safe-model"),
                range(8),
            )
        )

    assert build_calls == 1
    assert all(encoder is built for encoder in encoders)


def test_fastembed_corrupt_cache_is_purged_and_retried(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls = 0
    cache_dir = tmp_path / "fastembed"
    broken = cache_dir / "models--qdrant--all-MiniLM-L6-v2-onnx"
    (broken / "blobs").mkdir(parents=True)
    (broken / "blobs" / "model.incomplete").write_text("partial")

    class FakeTextEmbedding:
        def __init__(
            self,
            model_name: str,
            cache_dir: str,
            specific_model_path: str | None = None,
        ):
            nonlocal calls
            calls += 1
            self.model_name = model_name
            self.cache_dir = cache_dir
            self.specific_model_path = specific_model_path
            if calls == 1:
                raise RuntimeError(
                    "model.onnx failed: File doesn't exist"
                )

        @staticmethod
        def list_supported_models():
            return [{
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "sources": {"hf": "qdrant/all-MiniLM-L6-v2-onnx"},
                "dim": 384,
            }]

    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        SimpleNamespace(TextEmbedding=FakeTextEmbedding),
    )
    recovered = tmp_path / "gcs-model"
    recovered.mkdir()
    monkeypatch.setattr(
        ingest_mod,
        "_recover_fastembed_from_gcs",
        lambda _model, _cache: recovered,
    )
    monkeypatch.setenv("MNEMONICS_FASTEMBED_CACHE", str(cache_dir))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("MNEMONICS_FASTEMBED_REPAIR", raising=False)

    encoder = ingest_mod._FastEmbedEncoder("all-MiniLM-L6-v2")

    assert calls == 2
    assert encoder._emb.specific_model_path == str(recovered)
    assert encoder.get_sentence_embedding_dimension() == 384
    assert not broken.exists()


def test_fastembed_repair_is_disabled_offline(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls = 0
    cache_dir = tmp_path / "fastembed"
    broken = cache_dir / "models--qdrant--all-MiniLM-L6-v2-onnx"
    (broken / "blobs").mkdir(parents=True)
    (broken / "blobs" / "model.incomplete").write_text("partial")

    class FakeTextEmbedding:
        def __init__(self, model_name: str, cache_dir: str):
            nonlocal calls
            calls += 1
            raise RuntimeError("model.onnx failed: File doesn't exist")

        @staticmethod
        def list_supported_models():
            return [{
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "sources": {"hf": "qdrant/all-MiniLM-L6-v2-onnx"},
                "dim": 384,
            }]

    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        SimpleNamespace(TextEmbedding=FakeTextEmbedding),
    )
    monkeypatch.setenv("MNEMONICS_FASTEMBED_CACHE", str(cache_dir))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    with pytest.raises(RuntimeError, match="model.onnx failed"):
        ingest_mod._FastEmbedEncoder("all-MiniLM-L6-v2")

    assert calls == 1
    assert broken.exists()


def test_fastembed_cache_purge_unlinks_symlink_without_touching_target(
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    cache_dir = tmp_path / "fastembed"
    target = tmp_path / "outside"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    candidate = cache_dir / "models--qdrant--all-MiniLM-L6-v2-onnx"
    cache_dir.mkdir()
    candidate.symlink_to(target, target_is_directory=True)

    class FakeTextEmbedding:
        @staticmethod
        def list_supported_models():
            return [{
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "sources": {"hf": "qdrant/all-MiniLM-L6-v2-onnx"},
                "dim": 384,
            }]

    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        SimpleNamespace(TextEmbedding=FakeTextEmbedding),
    )

    assert ingest_mod._purge_fastembed_model_cache(
        "all-MiniLM-L6-v2",
        str(cache_dir),
    )
    assert not candidate.exists()
    assert (target / "keep.txt").read_text() == "keep"


def test_fastembed_gcs_recovery_uses_registry_source(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls: list[tuple] = []
    recovered = tmp_path / "recovered-model"
    recovered.mkdir()

    class FakeEmbeddingType:
        @classmethod
        def _get_model_description(cls, model_name: str):
            if model_name != "sentence-transformers/all-MiniLM-L6-v2":
                raise ValueError(model_name)
            return SimpleNamespace(
                model=model_name,
                sources=SimpleNamespace(
                    url="https://storage.example/model.tar.gz",
                    deprecated_tar_struct=True,
                ),
            )

        @classmethod
        def retrieve_model_gcs(
            cls,
            model_name,
            source_url,
            cache_dir,
            *,
            deprecated_tar_struct,
            local_files_only,
        ):
            calls.append(
                (
                    model_name,
                    source_url,
                    cache_dir,
                    deprecated_tar_struct,
                    local_files_only,
                )
            )
            return recovered

    class FakeTextEmbedding:
        EMBEDDINGS_REGISTRY = [FakeEmbeddingType]

    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        SimpleNamespace(TextEmbedding=FakeTextEmbedding),
    )

    result = ingest_mod._recover_fastembed_from_gcs(
        "all-MiniLM-L6-v2",
        str(tmp_path / "cache"),
    )

    assert result == recovered
    assert calls == [(
        "sentence-transformers/all-MiniLM-L6-v2",
        "https://storage.example/model.tar.gz",
        str(tmp_path / "cache"),
        True,
        False,
    )]


def test_fastembed_ready_gcs_cache_bypasses_network_resolution(tmp_path, monkeypatch):
    from types import SimpleNamespace

    cache_dir = tmp_path / "fastembed"
    local_model = cache_dir / "fast-all-MiniLM-L6-v2"
    local_model.mkdir(parents=True)
    (local_model / "model.onnx").write_bytes(b"onnx")
    seen_paths: list[str | None] = []

    class FakeTextEmbedding:
        def __init__(
            self,
            model_name: str,
            cache_dir: str,
            specific_model_path: str | None = None,
        ):
            seen_paths.append(specific_model_path)

        @staticmethod
        def list_supported_models():
            return [{
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "sources": {
                    "hf": "qdrant/all-MiniLM-L6-v2-onnx",
                    "url": "https://storage.example/model.tar.gz",
                    "_deprecated_tar_struct": True,
                },
                "dim": 384,
            }]

    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        SimpleNamespace(TextEmbedding=FakeTextEmbedding),
    )
    monkeypatch.setenv("MNEMONICS_FASTEMBED_CACHE", str(cache_dir))

    encoder = ingest_mod._FastEmbedEncoder("all-MiniLM-L6-v2")

    assert seen_paths == [str(local_model)]
    assert encoder.get_sentence_embedding_dimension() == 384
