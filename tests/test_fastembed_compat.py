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


def test_fastembed_wrapper_exposes_sentence_transformer_dimension(monkeypatch):
    seen: list[str] = []

    class FakeTextEmbedding:
        def __init__(self, model_name: str):
            seen.append(model_name)

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
    assert seen == ["sentence-transformers/all-MiniLM-L6-v2"]
    assert encoder.get_sentence_embedding_dimension() == 384
