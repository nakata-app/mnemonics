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
