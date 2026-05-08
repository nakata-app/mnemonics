"""Tests for mnemonics.retrieve (V2: tier + decay + reinforcement)."""
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mnemonics.retrieve import (
    _age_days,
    _decay_factor,
    _reinforcement_boost,
    retrieve,
)
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
    result = retrieve("query", store)
    assert len(result["results"]) > 0


def test_retrieve_closest_is_first(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[2])
    result = retrieve("query", store, top_k=1)
    assert result["results"][0]["text"] == docs[2]


def test_retrieve_top_k_respected(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    result = retrieve("q", store, top_k=3)
    assert len(result["results"]) == 3


def test_retrieve_empty_store(tmp_store, mock_enc):
    rng = np.random.default_rng(1)
    v = rng.random((1, DIM)).astype("float32")
    mock_enc.return_value = _enc_returning(v[0])
    result = retrieve("q", tmp_store)
    assert result["results"] == []


def test_retrieve_namespace_isolation(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    result = retrieve("q", store, ns="nonexistent")
    assert result["results"] == []


# ── result shape ─────────────────────────────────────────────────────────────

def test_result_keys_top_level(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    result = retrieve("q", store)
    assert set(result.keys()) == {"results"}


def test_per_row_v2_keys(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    result = retrieve("q", store, decay=True)
    for r in result["results"]:
        assert {"id", "text", "score", "raw_score", "decay_factor",
                "boost", "age_days", "tier"}.issubset(r.keys())


# ── decay & boost helpers ────────────────────────────────────────────────────

def test_decay_pinned_never_fades():
    assert _decay_factor(tier=0, age_days=10_000) == 1.0


def test_decay_default_half_life_90d():
    # 90 days should be exactly half
    assert math.isclose(_decay_factor(tier=1, age_days=90), 0.5, rel_tol=1e-3)


def test_decay_ambient_half_life_14d():
    assert math.isclose(_decay_factor(tier=2, age_days=14), 0.5, rel_tol=1e-3)


def test_boost_zero_access_is_unity():
    assert _reinforcement_boost(0) == 1.0


def test_boost_grows_then_caps_at_2():
    assert _reinforcement_boost(10) > 1.0
    assert _reinforcement_boost(10**8) <= 2.0


def test_age_days_handles_garbage():
    assert _age_days("not a date") == 0.0
    assert _age_days(None) == 0.0  # type: ignore[arg-type]


# ── decay applied vs not ─────────────────────────────────────────────────────

def test_no_decay_score_equals_raw(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    result = retrieve("q", store, decay=False)
    for r in result["results"]:
        assert r["score"] == r["raw_score"]
        assert r["boost"] == 1.0
        assert r["decay_factor"] == 1.0


def test_decay_lowers_score_for_old_default_row(populated_store, mock_enc, monkeypatch):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])

    # Backdate one row by 90 days so its decay = 0.5
    target_id = list(store._db.execute(
        "SELECT id FROM memories ORDER BY id LIMIT 1"
    ).fetchall())[0][0]
    old = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S")
    store._db.execute("UPDATE memories SET created=? WHERE id=?", (old, target_id))
    store._db.commit()

    result = retrieve("q", store, top_k=5, decay=True)
    target = next(r for r in result["results"] if r["id"] == target_id)
    assert target["decay_factor"] < 0.6  # ≈ 0.5 with a tiny float wiggle
    assert target["score"] < target["raw_score"]


# ── score ordering ────────────────────────────────────────────────────────────

def test_results_ordered_by_score_desc(populated_store, mock_enc):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    result = retrieve("q", store, top_k=5)
    scores = [r["score"] for r in result["results"]]
    assert scores == sorted(scores, reverse=True)
