"""Characterization of final ordering: decay, reinforcement and signal boost.

These tests DOCUMENT what retrieve() does today. They are not a statement that
this is the intended ranking semantics. The headline fact they lock:

    With rerank=True the cross-encoder score replaces `score` and fully decides
    the order. Tier decay, reinforcement (access_count) and the quoted-phrase /
    name signal boost are still computed and still appear on each returned row
    (`decay_factor`, `boost`, `signal_boost`), but they no longer influence the
    final ordering. Only `project_hints` is applied after the cross-encoder.

With rerank=False the same signals DO decide the order, within the rows that
survived the top_k cut taken before they are applied.

If a later change makes the signals count after the cross-encoder, these tests
must fail on purpose and be updated together with a measured A/B, not silently.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mnemonics.retrieve import retrieve
from mnemonics.store import DIM, Store


class StubCE:
    """Reranker returning fixed scores by document text (default 0.0)."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.calls = 0

    def rerank(self, query, documents, max_length=None, batch_size=None):
        self.calls += 1
        ranked = sorted(
            enumerate(self.scores.get(d, 0.0) for d in documents), key=lambda x: -x[1]
        )
        return [(i, float(s)) for i, s in ranked]


@pytest.fixture
def query_vec():
    v = np.zeros(DIM, dtype="float32")
    v[0] = 1.0
    return v


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("MNEMONICS_SCORE_FULL_BAND", raising=False)
    with patch("mnemonics.retrieve._get_encoder") as enc:
        yield enc


def _use_query(enc, vec):
    m = MagicMock()
    m.encode.return_value = vec.reshape(1, -1)
    enc.return_value = m


def _store(tmp_path, rows):
    """rows: dicts with text, cos (to the query), tier, age_days, access, meta."""
    store = Store(tmp_path)
    vecs = []
    for k, r in enumerate(rows):
        v = np.zeros(DIM, dtype="float32")
        v[0] = r["cos"]
        v[k + 1] = float(np.sqrt(1.0 - r["cos"] ** 2))
        vecs.append(v)
    ids = store.add(
        [r["text"] for r in rows],
        np.stack(vecs),
        meta=[r.get("meta", {}) for r in rows],
    )
    now = datetime.now(timezone.utc)
    for mid, r in zip(ids, rows):
        created = (now - timedelta(days=r.get("age_days", 0))).strftime("%Y-%m-%d %H:%M:%S")
        store._db.execute(
            "UPDATE memories SET created=?, tier=?, access_count=? WHERE id=?",
            (created, r.get("tier", 1), r.get("access", 0), mid),
        )
    store._db.commit()
    return store, dict(zip([r["text"] for r in rows], ids))


def _texts(out):
    return [r["text"] for r in out["results"]]


def _run(store, **kw):
    kw.setdefault("hybrid", False)
    kw.setdefault("top_k", 3)
    kw.setdefault("candidate_k", 10)
    return retrieve("find 'needle phrase'" if kw.pop("quoted", False) else "q", store, **kw)


# Each scenario isolates ONE signal. The vector order is always alpha > bravo >
# charlie; the cross-encoder order is always alpha > charlie > bravo.
CE_ORDER = ["alpha", "charlie", "bravo"]
CE_SCORES = {"alpha": 0.9, "charlie": 0.7, "bravo": 0.6}

SCENARIOS = {
    # alpha is ambient (tier 2) and 400 days old: its decay factor is ~0.
    "decay": dict(
        rows=[
            dict(text="alpha", cos=0.99, tier=2, age_days=400),
            dict(text="bravo", cos=0.90, tier=1),
            dict(text="charlie", cos=0.80, tier=1),
        ],
        quoted=False,
        off_order=["bravo", "charlie", "alpha"],
        field="decay_factor",
    ),
    # bravo has been read 50 times (boost 1.39); everything else is untouched.
    "reinforcement": dict(
        rows=[
            dict(text="alpha", cos=0.99, tier=0),
            dict(text="bravo", cos=0.90, tier=0, access=50),
            dict(text="charlie", cos=0.80, tier=0),
        ],
        quoted=False,
        off_order=["bravo", "alpha", "charlie"],
        field="boost",
    ),
    # bravo's text contains the quoted phrase in the question (boost 1.6).
    "signal": dict(
        rows=[
            dict(text="alpha", cos=0.99, tier=0),
            dict(text="bravo needle phrase", cos=0.90, tier=0),
            dict(text="charlie", cos=0.80, tier=0),
        ],
        quoted=True,
        off_order=["bravo needle phrase", "alpha", "charlie"],
        field="signal_boost",
    ),
}


def _ce_for(name):
    if name == "signal":
        return StubCE({"alpha": 0.9, "charlie": 0.7, "bravo needle phrase": 0.5})
    return StubCE(CE_SCORES)


def _ce_order(name):
    return ["alpha", "charlie", "bravo needle phrase"] if name == "signal" else CE_ORDER


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_without_rerank_the_signal_decides_the_order(tmp_path, _env, query_vec, name):
    sc = SCENARIOS[name]
    _use_query(_env, query_vec)
    store, _ = _store(tmp_path, sc["rows"])
    on = _run(store, rerank=False, decay=True, quoted=sc["quoted"], boost_signals=True)
    assert _texts(on) == sc["off_order"]


@pytest.mark.parametrize("name", ["decay", "reinforcement"])
def test_without_rerank_decay_false_restores_vector_order(tmp_path, _env, query_vec, name):
    sc = SCENARIOS[name]
    _use_query(_env, query_vec)
    store, _ = _store(tmp_path, sc["rows"])
    out = _run(store, rerank=False, decay=False)
    assert _texts(out) == [r["text"] for r in sc["rows"]]  # alpha > bravo > charlie


def test_without_rerank_signal_boost_off_restores_vector_order(tmp_path, _env, query_vec):
    sc = SCENARIOS["signal"]
    _use_query(_env, query_vec)
    store, _ = _store(tmp_path, sc["rows"])
    out = _run(store, rerank=False, decay=True, quoted=True, boost_signals=False)
    assert _texts(out) == ["alpha", "bravo needle phrase", "charlie"]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_with_rerank_the_cross_encoder_alone_decides_the_order(
    tmp_path, _env, query_vec, monkeypatch, name
):
    """Same rows, same signals; rerank=True ignores them for ordering."""
    sc = SCENARIOS[name]
    _use_query(_env, query_vec)
    store, _ = _store(tmp_path, sc["rows"])
    ce = _ce_for(name)
    monkeypatch.setattr("mnemonics.retrieve._get_rerank_ce", lambda model=None: ce)

    results = {}
    for decay in (True, False):
        for boost in (True, False):
            out = _run(
                store, rerank=True, decay=decay, boost_signals=boost, quoted=sc["quoted"]
            )
            results[(decay, boost)] = out
            assert _texts(out) == _ce_order(name)

    # `score` is the CE score, exactly, whatever the signals say.
    for out in results.values():
        assert [r["score"] for r in out["results"]] == [r["ce_score"] for r in out["results"]]


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_with_rerank_the_signal_is_still_computed_and_exposed(
    tmp_path, _env, query_vec, monkeypatch, name
):
    """The factor is on the row (so it looks live) but never reaches `score`."""
    sc = SCENARIOS[name]
    _use_query(_env, query_vec)
    store, _ = _store(tmp_path, sc["rows"])
    ce = _ce_for(name)
    monkeypatch.setattr("mnemonics.retrieve._get_rerank_ce", lambda model=None: ce)
    out = _run(store, rerank=True, decay=True, boost_signals=True, quoted=sc["quoted"])
    by_text = {r["text"]: r for r in out["results"]}
    field = sc["field"]

    if name == "decay":
        assert by_text["alpha"]["decay_factor"] < 1e-6  # effectively erased...
        assert _texts(out)[0] == "alpha"  # ...and still ranked first by the CE
    elif name == "reinforcement":
        assert by_text["bravo"]["boost"] > 1.3  # strongly reinforced...
        assert _texts(out)[-1] == "bravo"  # ...and still ranked last by the CE
    else:
        assert by_text["bravo needle phrase"]["signal_boost"] == pytest.approx(1.6)
        assert _texts(out)[-1] == "bravo needle phrase"
    assert field in next(iter(by_text.values()))


def test_with_rerank_decay_false_still_reports_neutral_factors(
    tmp_path, _env, query_vec, monkeypatch
):
    sc = SCENARIOS["decay"]
    _use_query(_env, query_vec)
    store, _ = _store(tmp_path, sc["rows"])
    ce = _ce_for("decay")
    monkeypatch.setattr("mnemonics.retrieve._get_rerank_ce", lambda model=None: ce)
    out = _run(store, rerank=True, decay=False)
    assert {r["decay_factor"] for r in out["results"]} == {1.0}
    assert {r["boost"] for r in out["results"]} == {1.0}


def test_reinforcement_accumulates_under_rerank_but_never_ranks(
    tmp_path, _env, query_vec, monkeypatch
):
    """Retrieval keeps bumping access_count on the returned rows, yet the count
    cannot change a rerank=True order, so the reinforcement loop is write-only."""
    sc = SCENARIOS["reinforcement"]
    _use_query(_env, query_vec)
    store, ids = _store(tmp_path, sc["rows"])
    ce = _ce_for("reinforcement")
    monkeypatch.setattr("mnemonics.retrieve._get_rerank_ce", lambda model=None: ce)
    before = dict(store._db.execute("SELECT id, access_count FROM memories").fetchall())
    for _ in range(3):
        assert _texts(_run(store, rerank=True)) == CE_ORDER
    after = dict(store._db.execute("SELECT id, access_count FROM memories").fetchall())
    for mid in ids.values():
        assert after[mid] == before[mid] + 3


def test_project_hints_is_the_only_adjustment_applied_after_the_cross_encoder(
    tmp_path, _env, query_vec, monkeypatch
):
    rows = [
        dict(text="alpha", cos=0.99, tier=0, meta={"project": "other"}),
        dict(text="bravo", cos=0.90, tier=0, meta={"project": "repo"}),
        dict(text="charlie", cos=0.80, tier=0),
    ]
    _use_query(_env, query_vec)
    store, _ = _store(tmp_path, rows)
    ce = StubCE({"alpha": 0.9, "bravo": 0.8, "charlie": 0.1})
    monkeypatch.setattr("mnemonics.retrieve._get_rerank_ce", lambda model=None: ce)

    plain = _run(store, rerank=True)
    assert _texts(plain) == ["alpha", "bravo", "charlie"]

    scoped = _run(store, rerank=True, project_hints=["repo"])
    assert _texts(scoped) == ["bravo", "alpha", "charlie"]  # 0.8*1.35 > 0.9*0.90
    by_text = {r["text"]: r for r in scoped["results"]}
    assert by_text["bravo"]["scope_boost"] == 1.35
    assert by_text["alpha"]["scope_boost"] == 0.90
    assert by_text["bravo"]["score"] == by_text["bravo"]["ce_score"]  # score untouched
    assert by_text["bravo"]["scope_score"] == pytest.approx(0.8 * 1.35)


def test_without_rerank_signals_only_reshuffle_rows_that_survived_the_top_k_cut(
    tmp_path, _env, query_vec, monkeypatch
):
    """The fused list is cut to top_k BEFORE decay / boost / signal run, so a
    quoted-phrase match at rank 2 cannot be promoted into a top_k=1 answer.
    MNEMONICS_SCORE_FULL_BAND=1 scores the whole band first and fixes that."""
    sc = SCENARIOS["signal"]
    _use_query(_env, query_vec)
    store, _ = _store(tmp_path, sc["rows"])

    cut = _run(store, rerank=False, top_k=1, quoted=True, boost_signals=True)
    assert _texts(cut) == ["alpha"]

    monkeypatch.setenv("MNEMONICS_SCORE_FULL_BAND", "1")
    wide = _run(store, rerank=False, top_k=1, quoted=True, boost_signals=True)
    assert _texts(wide) == ["bravo needle phrase"]
