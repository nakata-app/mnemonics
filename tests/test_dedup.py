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
