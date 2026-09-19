from __future__ import annotations

from types import SimpleNamespace

from mnemonics import embed_manifest
from mnemonics.ingest import _resolve_model_for_store


def test_store_manifest_encoder_wins_over_unconfigured_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MNEMONICS_ENCODER_MODEL", raising=False)
    monkeypatch.delenv("MNEMONICS_ADAPTMEM_PATH", raising=False)
    model_dir = tmp_path / "adaptmem-model"
    model_dir.mkdir()
    embed_manifest.write(
        tmp_path,
        {
            "encoder": str(model_dir),
            "dim": 1024,
            "fingerprint": "test",
            "kind": "local",
        },
    )

    store = SimpleNamespace(root=tmp_path)
    assert _resolve_model_for_store("all-MiniLM-L6-v2", store) == str(model_dir)


def test_explicit_encoder_env_overrides_store_manifest(tmp_path, monkeypatch):
    embed_manifest.write(
        tmp_path,
        {"encoder": "old-model", "dim": 384, "fingerprint": "old", "kind": "hub"},
    )
    monkeypatch.setenv("MNEMONICS_ENCODER_MODEL", "new-model")
    store = SimpleNamespace(root=tmp_path)

    assert _resolve_model_for_store("all-MiniLM-L6-v2", store) == "new-model"


def test_explicit_model_argument_is_not_replaced_by_manifest(tmp_path, monkeypatch):
    monkeypatch.delenv("MNEMONICS_ENCODER_MODEL", raising=False)
    monkeypatch.delenv("MNEMONICS_ADAPTMEM_PATH", raising=False)
    embed_manifest.write(
        tmp_path,
        {"encoder": "old-model", "dim": 384, "fingerprint": "old", "kind": "hub"},
    )
    store = SimpleNamespace(root=tmp_path)

    assert _resolve_model_for_store("custom-model", store) == "custom-model"
