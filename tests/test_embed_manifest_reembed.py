"""Tests for the encoder-manifest + reembed safety system.

Covers: embed_manifest (fingerprint/read/write/verify), store.reembed_ns,
ingest.reembed_all + ingest version-stamp, retrieve encoder-drift check, and
the `reembed` CLI command.
"""
import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mnemonics import embed_manifest as em
from mnemonics.cli import main
from mnemonics.ingest import ingest, reembed_all
from mnemonics.retrieve import retrieve
from mnemonics.store import DIM, Store


def _vecs(n):
    """n unit-ish vectors of the store dimension."""
    return np.ones((n, DIM), dtype="float32")


def _enc_mock():
    """Encoder whose encode() returns the right (n, DIM) shape for any input."""
    enc = MagicMock()
    enc.encode.side_effect = lambda texts, **kw: _vecs(len(texts))
    return enc


def _clear_retrieve_cache():
    retrieve.__dict__.pop("_enc_checked", None)


# ── embed_manifest.encoder_fingerprint ────────────────────────────────────────

def test_fingerprint_hub_model():
    fp = em.encoder_fingerprint("org/some-nonexistent-hub-model-xyz", 384)
    assert fp["kind"] == "hub"
    assert fp["fingerprint"] == "org/some-nonexistent-hub-model-xyz"
    assert fp["dim"] == 384


def test_fingerprint_local_with_weights(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    fp = em.encoder_fingerprint(str(tmp_path), 384)
    assert fp["kind"] == "local"
    # weights-based fingerprint is a blake2b hex digest, not the path
    assert fp["fingerprint"] != str(tmp_path)
    assert len(fp["fingerprint"]) == 32


def test_fingerprint_local_no_weights(tmp_path):
    enc_dir = tmp_path / "enc"
    enc_dir.mkdir()
    fp = em.encoder_fingerprint(str(enc_dir), 384)
    assert fp["kind"] == "local"
    assert len(fp["fingerprint"]) == 32  # blake2b of the path string


# ── embed_manifest.read / write ───────────────────────────────────────────────

def test_read_missing_returns_none(tmp_path):
    assert em.read(tmp_path) is None


def test_write_then_read_roundtrip(tmp_path):
    em.write(tmp_path, {"a": 1, "encoder": "m"})
    assert em.read(tmp_path) == {"a": 1, "encoder": "m"}


def test_read_corrupt_returns_none(tmp_path):
    (tmp_path / em.MANIFEST_NAME).write_text("{not valid json")
    assert em.read(tmp_path) is None


def test_write_cleans_up_tmp_on_error(tmp_path):
    # json.dump on a non-serializable value raises; the finally branch must
    # unlink the temp file so no .tmp turd is left behind.
    with pytest.raises(TypeError):
        em.write(tmp_path, {"bad": object()})
    assert list(tmp_path.glob("*.tmp")) == []


# ── embed_manifest.verify ─────────────────────────────────────────────────────

def test_verify_missing(tmp_path):
    status, why = em.verify(tmp_path, {"dim": 384, "fingerprint": "x"})
    assert status == "missing"
    assert why


def test_verify_dim_mismatch(tmp_path):
    em.write(tmp_path, {"dim": 384, "fingerprint": "x", "encoder": "m"})
    status, why = em.verify(tmp_path, {"dim": 1024, "fingerprint": "x"})
    assert status == "mismatch"
    assert "dim" in why


def test_verify_fingerprint_mismatch(tmp_path):
    em.write(tmp_path, {"dim": 384, "fingerprint": "A", "encoder": "old"})
    status, why = em.verify(tmp_path, {"dim": 384, "fingerprint": "B", "encoder": "new"})
    assert status == "mismatch"
    assert "fingerprint" in why


def test_verify_ok(tmp_path):
    em.write(tmp_path, {"dim": 384, "fingerprint": "A", "encoder": "m"})
    status, why = em.verify(tmp_path, {"dim": 384, "fingerprint": "A", "encoder": "m"})
    assert status == "ok"
    assert why == ""


# ── store.reembed_ns ──────────────────────────────────────────────────────────

def test_reembed_ns_recomputes(tmp_store):
    tmp_store.add(["alpha", "beta", "gamma"], _vecs(3))
    n = tmp_store.reembed_ns("default", lambda texts: _vecs(len(texts)))
    assert n == 3
    # index still usable after rebuild
    hits = tmp_store.search(_vecs(1)[0], ns="default", top_k=3)
    assert len(hits) == 3


def test_reembed_ns_empty_returns_zero(tmp_store):
    assert tmp_store.reembed_ns("ghost", lambda texts: _vecs(len(texts))) == 0


def test_reembed_ns_count_mismatch_raises(tmp_store):
    tmp_store.add(["a", "b"], _vecs(2))
    with pytest.raises(ValueError, match="vectors for"):
        tmp_store.reembed_ns("default", lambda texts: _vecs(len(texts) + 1))


def test_reembed_ns_dim_mismatch_raises(tmp_store):
    tmp_store.add(["a", "b"], _vecs(2))
    with pytest.raises(ValueError, match="store dim"):
        tmp_store.reembed_ns("default", lambda texts: np.ones((len(texts), 5), "float32"))


# ── ingest.reembed_all + version stamp ────────────────────────────────────────

def test_reembed_all_rebuilds_and_stamps(tmp_path):
    store = Store(tmp_path)
    store.add(["one", "two", "three"], _vecs(3))
    with patch("mnemonics.ingest._get_encoder", return_value=_enc_mock()):
        results = reembed_all(store)
    assert {"ns": "default", "n": 3} in results
    assert em.read(store.root) is not None  # manifest re-stamped


def test_reembed_all_survives_manifest_write_failure(tmp_path):
    store = Store(tmp_path)
    store.add(["x"], _vecs(1))
    with (
        patch("mnemonics.ingest._get_encoder", return_value=_enc_mock()),
        patch("mnemonics.embed_manifest.write", side_effect=OSError("disk full")),
    ):
        results = reembed_all(store)  # best-effort stamp must not break reembed
    assert results == [{"ns": "default", "n": 1}]


def test_ingest_survives_manifest_write_failure(tmp_path):
    store = Store(tmp_path)
    with (
        patch("mnemonics.ingest._get_encoder", return_value=_enc_mock()),
        patch("mnemonics.embed_manifest.write", side_effect=OSError("disk full")),
    ):
        n = ingest("hello world", store=store)
    assert n >= 1


# ── retrieve encoder-drift check ──────────────────────────────────────────────

def test_retrieve_warns_on_encoder_drift(populated_store, caplog):
    store, docs, vecs = populated_store
    # Same dim as the store (so verify gets past the dim check) but a fingerprint
    # the live encoder can never produce, so verify() always reports a mismatch.
    em.write(store.root, {"encoder": "old", "dim": store.dim,
                          "fingerprint": "DEFINITELY_OLD_FP", "kind": "hub"})
    _clear_retrieve_cache()
    with patch("mnemonics.retrieve._get_encoder") as m:
        m.return_value.encode.return_value = vecs[0].reshape(1, -1)
        with caplog.at_level(logging.WARNING, logger="mnemonics"):
            retrieve("q", store)
    assert "ENCODER DRIFT" in caplog.text


def test_retrieve_drift_check_never_breaks_retrieval(populated_store):
    store, docs, vecs = populated_store
    _clear_retrieve_cache()
    with (
        patch("mnemonics.retrieve._get_encoder") as m,
        patch("mnemonics.embed_manifest.verify", side_effect=RuntimeError("boom")),
    ):
        m.return_value.encode.return_value = vecs[0].reshape(1, -1)
        result = retrieve("q", store)  # drift check raises internally, swallowed
    assert "results" in result


# ── cli reembed ───────────────────────────────────────────────────────────────

def test_cli_reembed_reports_counts(tmp_path, capsys):
    with (
        patch("mnemonics.store.Store", return_value=MagicMock()),
        patch("mnemonics.ingest.reembed_all", return_value=[{"ns": "default", "n": 3}]),
        patch("sys.argv", ["mnemonics", "reembed", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "re-embedded" in out
    assert "manifest updated" in out


def test_cli_reembed_no_namespaces(tmp_path, capsys):
    with (
        patch("mnemonics.store.Store", return_value=MagicMock()),
        patch("mnemonics.ingest.reembed_all", return_value=[]),
        patch("sys.argv", ["mnemonics", "reembed", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "No namespaces found" in out
