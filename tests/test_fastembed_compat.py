from __future__ import annotations

import sys
from types import SimpleNamespace

from mnemonics.ingest import _fastembed_model_name, _FastEmbedEncoder


def test_minilm_shorthand_maps_to_fastembed_registry_id():
    assert (
        _fastembed_model_name("all-MiniLM-L6-v2")
        == "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert _fastembed_model_name("BAAI/bge-base-en-v1.5") == "BAAI/bge-base-en-v1.5"


def test_fastembed_wrapper_exposes_sentence_transformer_dimension(monkeypatch, tmp_path):
    seen: list[tuple[str, str | None]] = []
    monkeypatch.setenv("MNEMONICS_FASTEMBED_CACHE", str(tmp_path))

    class FakeTextEmbedding:
        def __init__(self, model_name: str, cache_dir: str | None = None):
            seen.append((model_name, cache_dir))

        @classmethod
        def list_supported_models(cls):
            return [
                {
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "dim": 384,
                }
            ]

        def embed(self, texts, batch_size=256):
            return []

    monkeypatch.setitem(
        sys.modules,
        "fastembed",
        SimpleNamespace(TextEmbedding=FakeTextEmbedding),
    )

    encoder = _FastEmbedEncoder("all-MiniLM-L6-v2")
    assert seen == [
        ("sentence-transformers/all-MiniLM-L6-v2", str(tmp_path))
    ]
    assert encoder.get_sentence_embedding_dimension() == 384

def test_stock_encoder_falls_back_when_fastembed_init_fails(monkeypatch):
    import mnemonics.ingest as ingest_mod

    class Boom:
        def __init__(self, model_name: str):
            raise RuntimeError("broken fastembed cache")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str):
            self.model_name = model_name

    monkeypatch.setattr(ingest_mod, "_FastEmbedEncoder", Boom)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    encoder = ingest_mod._build_encoder("all-MiniLM-L6-v2")
    assert isinstance(encoder, FakeSentenceTransformer)
    assert encoder.model_name == "all-MiniLM-L6-v2"
