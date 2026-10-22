"""Reranker scores must survive the call that produced them.

Before the cache existed every rerank call scored all documents and dropped
the numbers on the floor, so a second identical call paid the model cost again
and a re-run of an evaluation could not reuse anything.  These tests pin the
two properties that make the scores usable as experience: they are written
once, and the second call does not touch the model at all while returning the
identical ranking.
"""

from __future__ import annotations

import mnemonics.retrieve as retrieve
from mnemonics import rerank_cache


class _CountingReranker:
    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query, documents, max_length=None, batch_size=None):
        self.calls += 1
        # deterministic scoring that does not depend on call order
        return sorted(
            ((i, float(len(doc))) for i, doc in enumerate(documents)),
            key=lambda kv: (-kv[1], kv[0]),
        )


def _results():
    return [
        {"id": "a", "text": "short"},
        {"id": "b", "text": "a much longer document"},
        {"id": "c", "text": "medium length"},
    ]


def test_second_call_is_served_from_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMONICS_RERANK_CACHE", str(tmp_path / "scores.db"))
    fake = _CountingReranker()
    monkeypatch.setattr(retrieve, "_get_rerank_ce", lambda model=None: fake)
    monkeypatch.setattr(retrieve, "_rerank_model_name", "fake-model", raising=False)

    first = retrieve._ce_rerank("q", _results(), top_k=10)
    assert fake.calls == 1
    assert [r["id"] for r in first] == ["b", "c", "a"]

    second = retrieve._ce_rerank("q", _results(), top_k=10)
    assert fake.calls == 1, "a fully cached query must not load the model"
    assert [r["id"] for r in second] == [r["id"] for r in first]

    stats = rerank_cache.stats()
    assert stats["rows"] == 3
    assert stats["queries"] == 1


def test_changed_document_is_rescored(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMONICS_RERANK_CACHE", str(tmp_path / "scores.db"))
    fake = _CountingReranker()
    monkeypatch.setattr(retrieve, "_get_rerank_ce", lambda model=None: fake)
    monkeypatch.setattr(retrieve, "_rerank_model_name", "fake-model", raising=False)

    retrieve._ce_rerank("q", _results(), top_k=10)
    assert fake.calls == 1

    changed = _results()
    changed[2] = {"id": "c", "text": "medium length but edited"}
    retrieve._ce_rerank("q", changed, top_k=10)
    assert fake.calls == 2, "content hashes must invalidate stale scores"


def test_cache_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMONICS_RERANK_CACHE", "off")
    fake = _CountingReranker()
    monkeypatch.setattr(retrieve, "_get_rerank_ce", lambda model=None: fake)
    monkeypatch.setattr(retrieve, "_rerank_model_name", "fake-model", raising=False)

    retrieve._ce_rerank("q", _results(), top_k=10)
    retrieve._ce_rerank("q", _results(), top_k=10)
    assert fake.calls == 2
    assert not (tmp_path / "scores.db").exists()
