"""Tests for mnemonics.retrieve."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from mnemonics.retrieve import retrieve
from mnemonics.store import DIM


def _enc_returning(vec: np.ndarray) -> MagicMock:
    enc = MagicMock()
    enc.encode.return_value = vec.reshape(1, -1)
    return enc


@pytest.fixture
def mock_enc():
    with patch("mnemonics.retrieve._get_encoder") as m:
        yield m


# ── basic retrieval ──────────────────────────────────────────────────────────

def test_retrieve_returns_results(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    result = retrieve("query", store, verify=False)
    assert len(result["results"]) > 0


def test_retrieve_closest_is_first(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[2])
    result = retrieve("query", store, top_k=1, verify=False)
    assert result["results"][0]["text"] == docs[2]


def test_retrieve_top_k_respected(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    result = retrieve("q", store, top_k=3, verify=False)
    assert len(result["results"]) == 3


def test_retrieve_empty_store(tmp_store, mock_enc):
    rng = np.random.default_rng(1)
    v = rng.random((1, DIM)).astype("float32")
    mock_enc.return_value = _enc_returning(v[0])
    result = retrieve("q", tmp_store, verify=False)
    assert result["results"] == []
    assert result["trust_score"] == 1.0
    assert result["flagged_count"] == 0


# ── result shape ─────────────────────────────────────────────────────────────

def test_result_keys_present(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    result = retrieve("q", store, verify=False)
    assert set(result.keys()) == {"results", "verified", "trust_score", "flagged_count"}


def test_each_result_has_flagged_key(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    result = retrieve("q", store, verify=False)
    for r in result["results"]:
        assert "flagged" in r


def test_no_verify_trust_score_default(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    result = retrieve("q", store, verify=False)
    assert result["trust_score"] == 1.0
    assert result["flagged_count"] == 0
    assert result["verified"] is True


# ── halluguard integration ───────────────────────────────────────────────────

def test_verify_calls_halluguard(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])

    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {
        "ok": True,
        "trust_score": 0.9,
        "n_flagged": 0,
        "flagged": [],
    }

    with patch("mnemonics.retrieve.requests") as mock_requests:
        mock_requests.post.return_value = mock_resp
        result = retrieve("q", store, verify=True, top_k=3)

    mock_requests.post.assert_called_once()
    call_url = mock_requests.post.call_args[0][0]
    assert "7801" in call_url
    assert result["trust_score"] == 0.9


def test_verify_marks_flagged_results(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])

    target_text = docs[0]
    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {
        "ok": False,
        "trust_score": 0.4,
        "n_flagged": 1,
        "flagged": [{"text": target_text}],
    }

    with patch("mnemonics.retrieve.requests") as mock_requests:
        mock_requests.post.return_value = mock_resp
        result = retrieve("q", store, verify=True, top_k=5)

    flagged = [r for r in result["results"] if r.get("flagged")]
    assert result["flagged_count"] == 1
    assert any(r["text"] == target_text for r in flagged)


def test_verify_graceful_when_daemon_down(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])

    with patch("mnemonics.retrieve.requests") as mock_requests:
        mock_requests.post.side_effect = ConnectionError("daemon down")
        result = retrieve("q", store, verify=True)

    # Should not raise; defaults preserved
    assert result["trust_score"] == 1.0
    assert result["flagged_count"] == 0
    for r in result["results"]:
        assert r["flagged"] is False


def test_verify_graceful_on_bad_response(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])

    mock_resp = MagicMock()
    mock_resp.ok = False
    mock_resp.json.return_value = {}

    with patch("mnemonics.retrieve.requests") as mock_requests:
        mock_requests.post.return_value = mock_resp
        result = retrieve("q", store, verify=True)

    assert result["trust_score"] == 1.0


# ── namespace isolation ───────────────────────────────────────────────────────

def test_retrieve_namespace_isolation(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    result = retrieve("q", store, ns="nonexistent", verify=False)
    assert result["results"] == []


# ── score ordering ────────────────────────────────────────────────────────────

def test_results_ordered_by_score_desc(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    result = retrieve("q", store, top_k=5, verify=False)
    scores = [r["score"] for r in result["results"]]
    assert scores == sorted(scores, reverse=True)
