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


# ── CE rerank (adaptmem) ──────────────────────────────────────────────────────


class _StubAM:
    """Stub AdaptMem.rerank() — returns canned (index, score) tuples."""

    def __init__(self, scores_for_text: dict[str, float]):
        self._scores = scores_for_text
        self.last_query: str | None = None
        self.last_texts: list[str] | None = None

    def rerank(self, query, candidates, top_k=None):
        self.last_query = query
        self.last_texts = list(candidates)
        ranked = sorted(
            enumerate(self._scores.get(t, 0.0) for t in candidates),
            key=lambda x: -float(x[1]),
        )
        if top_k is not None:
            ranked = ranked[:top_k]
        return [(i, float(s)) for i, s in ranked]


def test_rerank_reorders_and_attaches_ce_score(populated_store, mock_enc, monkeypatch):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    # Stub the cached AdaptMem instance — CE strongly prefers the photosynthesis doc.
    stub = _StubAM({
        "The Eiffel Tower is located in Paris, France.": 0.10,
        "Python is a high-level programming language created by Guido van Rossum.": 0.20,
        "The speed of light is approximately 299,792 km/s in vacuum.": 0.30,
        "Photosynthesis converts sunlight into chemical energy in plants.": 0.95,
        "Machine learning is a subset of artificial intelligence.": 0.40,
    })
    monkeypatch.setattr("mnemonics.retrieve._get_rerank_ce", lambda model=None: stub)

    result = retrieve("q", store, top_k=3, rerank=True)
    # CE pushes photosynthesis to position 0 regardless of vector order.
    assert result["results"][0]["text"] == "Photosynthesis converts sunlight into chemical energy in plants."
    assert result["results"][0]["ce_score"] == 0.95
    # `score` is replaced with ce_score under rerank.
    assert result["results"][0]["score"] == 0.95
    # raw_score (bi-encoder/RRF) is preserved as observation.
    assert "raw_score" in result["results"][0]
    # top_k is honored.
    assert len(result["results"]) == 3


def test_rerank_off_does_not_call_ce(populated_store, mock_enc, monkeypatch):
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    sentinel = {"called": False}

    def _should_not_be_called(model=None):
        sentinel["called"] = True
        raise AssertionError("CE must not load when rerank=False")

    monkeypatch.setattr("mnemonics.retrieve._get_rerank_ce", _should_not_be_called)
    retrieve("q", store, top_k=3, rerank=False)
    assert sentinel["called"] is False


def test_rerank_widens_candidate_band_before_truncate(populated_store, mock_enc, monkeypatch):
    """rerank=True keeps the full candidate_k band so CE can rescue buried hits."""
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    stub = _StubAM({})  # all 0.0 — order irrelevant
    monkeypatch.setattr("mnemonics.retrieve._get_rerank_ce", lambda model=None: stub)
    retrieve("q", store, top_k=2, candidate_k=20, rerank=True)
    # All 5 docs (or fewer if store has less) must reach CE — not just top_k=2.
    assert stub.last_texts is not None
    assert len(stub.last_texts) >= min(5, 20)


def test_rerank_falls_back_to_sentence_transformers_when_adaptmem_missing(
    populated_store, mock_enc, monkeypatch
):
    """If adaptmem is unavailable, rerank falls back to bare CrossEncoder."""
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    monkeypatch.setattr("mnemonics.retrieve._rerank_ce", None)
    monkeypatch.setattr("mnemonics.retrieve._rerank_model_name", None)

    import builtins
    real_import = builtins.__import__

    def _block_adaptmem(name, *a, **kw):
        if name == "adaptmem":
            raise ImportError("simulated missing adaptmem")
        return real_import(name, *a, **kw)

    class _StubCE:
        def __init__(self, model_name):
            self.model_name = model_name
        def predict(self, pairs, show_progress_bar=False):
            return [0.5 - i * 0.01 for i in range(len(pairs))]

    import sentence_transformers
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", _StubCE)
    monkeypatch.setattr(builtins, "__import__", _block_adaptmem)
    out = retrieve("q", store, top_k=3, rerank=True)
    results = out["results"] if isinstance(out, dict) else out
    assert len(results) <= 3
    assert all("ce_score" in r for r in results)


def test_rerank_empty_results_returns_empty(tmp_store, mock_enc, monkeypatch):
    """rerank over an empty store returns [] without touching CE."""
    import numpy as np
    mock_enc.return_value = _enc_returning(np.zeros(DIM, dtype="float32"))
    stub = _StubAM({})
    monkeypatch.setattr("mnemonics.retrieve._get_rerank_ce", lambda model=None: stub)
    result = retrieve("q", tmp_store, top_k=5, rerank=True)
    assert result["results"] == []
    assert stub.last_texts is None  # CE never invoked


# ── signal boost (quoted phrase + person name) ────────────────────────────────


def test_extract_quoted_phrases_single_and_double():
    from mnemonics.retrieve import _extract_quoted_phrases
    q = "what did you say about 'sexual compulsions' and \"binge eating\" earlier"
    out = _extract_quoted_phrases(q)
    assert "sexual compulsions" in out
    assert "binge eating" in out


def test_extract_quoted_phrases_too_short_skipped():
    from mnemonics.retrieve import _extract_quoted_phrases
    # Single-char and 2-char quoted content must not match (min length is 3)
    assert _extract_quoted_phrases("only 'a' here") == []
    assert _extract_quoted_phrases("only 'no'") == []


def test_extract_person_names_basic():
    from mnemonics.retrieve import _extract_person_names
    out = _extract_person_names("What did I do with Rachel on Wednesday?")
    assert "rachel" in out
    assert "wednesday" not in out  # filtered by _NOT_NAMES


def test_extract_person_names_question_words_filtered():
    from mnemonics.retrieve import _extract_person_names
    assert _extract_person_names("What is the answer?") == []


def test_signal_boost_quoted_match_lifts():
    from mnemonics.retrieve import _signal_boost
    score = _signal_boost("session mentions sexual compulsions clearly", ["sexual compulsions"], [])
    assert abs(score - (1.0 + 0.60)) < 1e-6


def test_signal_boost_name_match_smaller_lift():
    from mnemonics.retrieve import _signal_boost
    score = _signal_boost("I went to dinner with rachel", [], ["rachel"])
    assert abs(score - (1.0 + 0.25)) < 1e-6


def test_signal_boost_no_signals_is_noop():
    from mnemonics.retrieve import _signal_boost
    assert _signal_boost("any text", [], []) == 1.0


def test_retrieve_boost_signals_attaches_field_on_match(populated_store, mock_enc):
    """signal_boost field gets a >1.0 value for any doc matching quoted phrase."""
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    q = "where did you mention 'Photosynthesis converts sunlight'?"
    # boost_signals=True is default
    result = retrieve(q, store, top_k=5, hybrid=False, decay=False)
    photo = next(r for r in result["results"] if "Photosynthesis" in r["text"])
    assert photo["signal_boost"] > 1.0
    # Non-matching docs keep signal_boost = 1.0
    other = next(r for r in result["results"] if "Eiffel" in r["text"])
    assert other["signal_boost"] == 1.0


def test_retrieve_boost_signals_off_is_baseline(populated_store, mock_enc):
    """Setting boost_signals=False should suppress the boost."""
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    q = "where did you mention 'Photosynthesis converts sunlight'?"
    result = retrieve(q, store, top_k=5, hybrid=False, decay=False, boost_signals=False)
    for r in result["results"]:
        assert r["signal_boost"] == 1.0


def test_retrieve_boost_signals_no_match_is_noop(populated_store, mock_enc):
    """Quoted phrase that no doc contains should leave order unchanged."""
    store, docs, vecs = populated_store
    mock_enc.return_value = _enc_returning(vecs[0])
    base = retrieve("nothing matches here", store, top_k=5, hybrid=False, decay=False, boost_signals=False)
    boosted = retrieve("nothing 'xyz' matches", store, top_k=5, hybrid=False, decay=False, boost_signals=True)
    assert [r["id"] for r in base["results"]] == [r["id"] for r in boosted["results"]]


def test_get_rerank_ce_cache_hit(monkeypatch):
    """_get_rerank_ce returns cached instance without re-creating when model matches."""
    from mnemonics.retrieve import _get_rerank_ce

    sentinel = object()
    monkeypatch.setattr("mnemonics.retrieve._rerank_ce", sentinel)
    monkeypatch.setattr("mnemonics.retrieve._rerank_model_name", "test-model")

    result = _get_rerank_ce("test-model")
    assert result is sentinel


# ── _get_rerank_ce adaptmem and import-error paths ────────────────────────────

def test_get_rerank_ce_adaptmem_with_rerank(monkeypatch):
    """adaptmem available + has rerank → return AdaptMem instance."""
    import sys
    from unittest.mock import MagicMock
    from mnemonics import retrieve as _ret

    mock_am = MagicMock()
    mock_am.rerank = lambda q, t: []
    mock_adaptmem_mod = MagicMock()
    mock_adaptmem_mod.AdaptMem.return_value = mock_am

    monkeypatch.setitem(sys.modules, "adaptmem", mock_adaptmem_mod)
    _ret._rerank_ce = None
    _ret._rerank_model_name = None

    result = _ret._get_rerank_ce("dummy-model")
    assert result is mock_am
    _ret._rerank_ce = None
    _ret._rerank_model_name = None


def test_get_rerank_ce_missing_sentence_transformers(monkeypatch):
    """If sentence_transformers is missing and adaptmem fails → RuntimeError."""
    import sys
    from unittest.mock import MagicMock
    from mnemonics import retrieve as _ret

    # Make adaptmem import fail
    monkeypatch.setitem(sys.modules, "adaptmem", None)
    # Make sentence_transformers import fail
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    _ret._rerank_ce = None
    _ret._rerank_model_name = None

    import pytest
    with pytest.raises(RuntimeError, match="sentence-transformers"):
        _ret._get_rerank_ce("no-model")

    _ret._rerank_ce = None
    _ret._rerank_model_name = None
