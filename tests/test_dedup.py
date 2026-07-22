"""Tests for mnemonics.dedup — find_similar() and check_before_ingest().

These use the real encoder because the whole point of dedup is that the
similarity signal is genuine. Test ns is isolated per-test via tmp_path.
"""
from __future__ import annotations

import pytest

from mnemonics.dedup import check_before_ingest, find_similar
from mnemonics.ingest import ingest
from mnemonics.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(path=tmp_path)
    # Seed three loosely-related memories.
    ingest(
        texts=[
            "Zeus tests must run DeepSeek V4 Pro via NIM, not deepseek-chat v3.",
            "Sienna packshot pipeline is modelless; 1353-manifest uploads to R2.",
            "Pistachio devtools (F12) are forbidden in production UI checks.",
        ],
        store=s,
        ns="ddtest",
    )
    return s


def test_find_similar_returns_high_match_for_paraphrase(store):
    matches = find_similar(
        "Zeus tests should use DeepSeek V4 Pro through NIM instead of V3.",
        store=store,
        ns="ddtest",
        threshold=0.7,
    )
    assert len(matches) >= 1
    assert matches[0]["similarity"] >= 0.7
    # Best match must be the Zeus row, not Sienna or Pistachio.
    assert "Zeus" in matches[0]["text"] or "DeepSeek" in matches[0]["text"]


def test_find_similar_returns_empty_for_unrelated_topic(store):
    matches = find_similar(
        "Mac firmware updates require a SIP disable workaround.",
        store=store,
        ns="ddtest",
        threshold=0.92,
    )
    assert matches == []


def test_find_similar_respects_threshold(store):
    # A weakly-related query: with a low threshold something comes back, with
    # a strict one it shouldn't.
    weak = find_similar("Zeus eval harness pipeline notes", store=store, ns="ddtest", threshold=0.3)
    strict = find_similar("Zeus eval harness pipeline notes", store=store, ns="ddtest", threshold=0.99)
    assert len(weak) >= 1
    assert strict == []


def test_find_similar_respects_namespace(store):
    # Same text exists in ddtest but not in a fresh ns.
    matches = find_similar(
        "Zeus tests must run DeepSeek V4 Pro via NIM, not deepseek-chat v3.",
        store=store,
        ns="empty-ns",
        threshold=0.5,
    )
    assert matches == []


def test_check_before_ingest_per_text_shape(store):
    out = check_before_ingest(
        texts=[
            "Zeus tests use DeepSeek V4 Pro via NIM.",  # near-dupe of seed row
            "Completely unrelated note about coffee grinder calibration.",
        ],
        store=store,
        ns="ddtest",
        threshold=0.7,
    )
    assert len(out) == 2
    assert out[0]["suggestions"], "near-paraphrase should produce a suggestion"
    assert out[1]["suggestions"] == [], "unrelated text should produce zero suggestions"


def test_find_similar_top_k_cap(store):
    # Even if many rows clear the threshold, top_k should bound the output.
    out = find_similar(
        "Zeus tests DeepSeek pipeline",
        store=store,
        ns="ddtest",
        threshold=0.0,
        top_k=2,
    )
    assert len(out) <= 2


# --- reconcile_ingest: NOOP dedup + archive-not-delete supersede ---------------

from mnemonics.dedup import reconcile_ingest
from mnemonics.retrieve import retrieve


def test_reconcile_noop_skips_restatement(store):
    """A near-identical restatement is skipped, not stored twice."""
    before = check_before_ingest(["Pistachio devtools (F12) are forbidden in production UI checks."],
                                 store, ns="ddtest")
    assert before[0]["suggestions"], "seed should be a near-duplicate of itself"
    res = reconcile_ingest(
        ["Pistachio devtools (F12) are forbidden in production UI checks."],
        store, ns="ddtest",
    )
    assert res["added"] == []
    assert len(res["noop_skipped"]) == 1
    assert res["noop_skipped"][0]["similarity"] >= 0.98


def test_reconcile_adds_genuinely_new(store):
    res = reconcile_ingest(["Doriva billing runs on a wholly unrelated Stripe webhook."],
                           store, ns="ddtest")
    assert len(res["added"]) == 1
    assert res["noop_skipped"] == []


def test_reconcile_supersede_archives_not_deletes(tmp_path):
    s = Store(path=tmp_path)
    r0 = reconcile_ingest(["Atakan lives in Istanbul."], s, ns="t")
    old_id = r0["added"][0]

    r1 = reconcile_ingest(["Atakan moved to Ankara and now lives in Ankara."],
                          s, ns="t", supersede_map={0: [old_id]})
    new_id = r1["added"][0]
    assert r1["superseded"] == [{"old_id": old_id, "new_id": new_id}]
    assert r1["supersede_failed"] == []

    # Superseded row is hidden from normal retrieval...
    res = retrieve("where does Atakan live", s, ns="t", top_k=10)
    ids = [h["id"] for h in res["results"]]
    assert old_id not in ids
    assert new_id in ids

    # ...but still present in the DB for audit.
    assert s.get(old_id) is not None
    from mnemonics.ingest import _get_encoder
    qv = _get_encoder().encode(["Atakan Istanbul"], normalize_embeddings=True,
                               convert_to_numpy=True)[0]
    audit = s.search(qv, ns="t", top_k=10, exclude_superseded=False)
    old_row = next(a for a in audit if a["id"] == old_id)
    assert old_row["meta"]["status"] == "superseded"
    assert old_row["meta"]["superseded_by"] == new_id


def test_reconcile_supersede_bypasses_noop(tmp_path):
    """A supersede-flagged text is ingested even if it looks like a duplicate."""
    s = Store(path=tmp_path)
    r0 = reconcile_ingest(["The API rate limit is 100 requests per minute."], s, ns="t")
    old_id = r0["added"][0]
    # Same sentence, but the caller flags it as a correction of old_id.
    r1 = reconcile_ingest(["The API rate limit is 100 requests per minute."],
                          s, ns="t", supersede_map={0: [old_id]})
    assert len(r1["added"]) == 1           # NOT skipped as NOOP
    assert r1["superseded"] == [{"old_id": old_id, "new_id": r1["added"][0]}]


def test_reconcile_supersede_missing_id(tmp_path):
    s = Store(path=tmp_path)
    r = reconcile_ingest(["fresh fact"], s, ns="t", supersede_map={0: [99999]})
    assert len(r["added"]) == 1
    assert r["supersede_failed"] == [99999]
    assert r["superseded"] == []
