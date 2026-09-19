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
