from __future__ import annotations

from mnemonics import embed_manifest
from mnemonics.store import Store


def test_store_dim_comes_from_embed_manifest_when_unconfigured(tmp_path, monkeypatch):
    monkeypatch.delenv("MNEMONICS_DIM", raising=False)
    embed_manifest.write(
        tmp_path,
        {"encoder": "model-x", "dim": 1024, "fingerprint": "x", "kind": "hub"},
    )
    assert Store._resolve_dim(None, tmp_path) == 1024


def test_store_dim_env_overrides_manifest(tmp_path, monkeypatch):
    embed_manifest.write(
        tmp_path,
        {"encoder": "model-x", "dim": 1024, "fingerprint": "x", "kind": "hub"},
    )
    monkeypatch.setenv("MNEMONICS_DIM", "768")
    assert Store._resolve_dim(None, tmp_path) == 768


def test_store_explicit_dim_overrides_everything(tmp_path, monkeypatch):
    embed_manifest.write(
        tmp_path,
        {"encoder": "model-x", "dim": 1024, "fingerprint": "x", "kind": "hub"},
    )
    monkeypatch.setenv("MNEMONICS_DIM", "768")
    assert Store._resolve_dim(384, tmp_path) == 384
