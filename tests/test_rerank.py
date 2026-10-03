"""Reranker interface: backend selection, limits, typed failures, and parity.

Parity: the unified Reranker must rank exactly like the pre-interface
``_ce_rerank`` when no limit env var is set, on every backend. The legacy
implementation is reproduced below verbatim as the reference.
"""
from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import pytest

from mnemonics import retrieve as retrieve_mod
from mnemonics.rerank import (
    AdaptMemReranker,
    RerankerError,
    RerankerIncompatible,
    RerankerUnavailable,
    SentenceTransformersReranker,
    env_int,
    make_reranker,
)


class FakeCrossEncoder:
    """Deterministic CE: score depends only on (query, text); includes ties."""

    built: list[FakeCrossEncoder] = []

    def __init__(self, name, **kwargs):
        self.name = name
        self.init_kwargs = kwargs
        self.predict_kwargs: dict | None = None
        FakeCrossEncoder.built.append(self)

    def predict(self, pairs, **kwargs):
        self.predict_kwargs = kwargs
        return [float(len(t) % 3) + (0.5 if "x" in t else 0.0) for _, t in pairs]


class FakeAdaptMem:
    """Mirrors adaptmem.AdaptMem.rerank, built on the same fake CE."""

    last: FakeAdaptMem | None = None

    def __init__(self, rerank_model="m"):
        self.ce = FakeCrossEncoder(rerank_model)
        FakeAdaptMem.last = self

    def rerank(self, query, candidates, top_k=None, *, max_length=None, batch_size=None):
        if max_length is not None:
            self.ce.init_kwargs["max_length"] = max_length
        kwargs = {"show_progress_bar": False}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        scores = self.ce.predict([(query, t) for t in candidates], **kwargs)
        ranked = sorted(enumerate(scores), key=lambda x: -float(x[1]))
        return [(i, float(s)) for i, s in ranked]


class OldAdaptMem:
    """An adaptmem release that predates max_length / batch_size."""

    def __init__(self, rerank_model="m"):
        pass

    def rerank(self, query, candidates, top_k=None):
        return []


def _legacy_ce_rerank(ce_cls, name, query, results, top_k):
    """The pre-interface bare-CrossEncoder path of retrieve._ce_rerank, verbatim."""
    ce = ce_cls(name)
    texts = [r["text"] for r in results]
    pairs = [(query, t) for t in texts]
    scores = ce.predict(pairs, show_progress_bar=False)
    ranked = sorted(enumerate(scores), key=lambda x: -float(x[1]))
    out = []
    for idx, ce_score in ranked:
        item = dict(results[idx])
        item["ce_score"] = round(float(ce_score), 4)
        item["score"] = item["ce_score"]
        out.append(item)
        if len(out) >= top_k:
            break
    return out


ROWS = [{"id": i, "text": t} for i, t in enumerate(
    ["alpha", "bravo x", "charlie", "delta x", "echo", "foxtrot", "golf x", "hotel", "india"]
)]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    FakeCrossEncoder.built = []
    FakeAdaptMem.last = None
    for var in (
        "MNEMONICS_RERANK_BACKEND",
        "MNEMONICS_RERANK_MAX_LENGTH",
        "MNEMONICS_RERANK_BATCH_SIZE",
        "MNEMONICS_RERANK_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(retrieve_mod, "_rerank_ce", None)
    monkeypatch.setattr(retrieve_mod, "_rerank_model_name", None)


def _install(monkeypatch, *, adaptmem):
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeCrossEncoder)
    )
    monkeypatch.setitem(
        sys.modules,
        "adaptmem",
        None if adaptmem is None else SimpleNamespace(AdaptMem=adaptmem, __version__="9.9"),
    )


# ── parity with the pre-interface path (env unset) ────────────────────────────


@pytest.mark.parametrize("backend_cls", [None, FakeAdaptMem], ids=["sentence-transformers", "adaptmem"])
@pytest.mark.parametrize("top_k", [1, 3, 5, 9, 20])
def test_rerank_stage_parity_with_legacy_path(monkeypatch, backend_cls, top_k):
    _install(monkeypatch, adaptmem=backend_cls)
    legacy = _legacy_ce_rerank(FakeCrossEncoder, "ce", "q", ROWS, top_k)
    monkeypatch.setenv("MNEMONICS_RERANK_MODEL", "ce")
    now = retrieve_mod._ce_rerank("q", ROWS, top_k)
    assert [r["id"] for r in now] == [r["id"] for r in legacy]
    assert [r["ce_score"] for r in now] == [r["ce_score"] for r in legacy]
    assert now == legacy  # every field, including the replaced `score`


def test_parity_calls_the_model_exactly_as_before(monkeypatch):
    """Env unset: CrossEncoder gets no kwargs and predict only show_progress_bar."""
    _install(monkeypatch, adaptmem=None)
    retrieve_mod._ce_rerank("q", ROWS, 3)
    ce = FakeCrossEncoder.built[-1]
    assert ce.init_kwargs == {}
    assert ce.predict_kwargs == {"show_progress_bar": False}


# ── limits apply on every backend ─────────────────────────────────────────────


@pytest.mark.parametrize("backend_cls", [None, FakeAdaptMem], ids=["sentence-transformers", "adaptmem"])
def test_limits_env_applies_on_every_backend(monkeypatch, backend_cls):
    _install(monkeypatch, adaptmem=backend_cls)
    monkeypatch.setenv("MNEMONICS_RERANK_MAX_LENGTH", "512")
    monkeypatch.setenv("MNEMONICS_RERANK_BATCH_SIZE", "16")
    retrieve_mod._ce_rerank("q", ROWS, 3)
    ce = FakeAdaptMem.last.ce if backend_cls else FakeCrossEncoder.built[-1]
    assert ce.init_kwargs == {"max_length": 512}
    assert ce.predict_kwargs == {"show_progress_bar": False, "batch_size": 16}


def test_sentence_transformers_reloads_only_when_max_length_changes(monkeypatch):
    _install(monkeypatch, adaptmem=None)
    r = SentenceTransformersReranker("ce")
    r.rerank("q", ["a"], max_length=128)
    r.rerank("q", ["a"], max_length=128, batch_size=2)
    assert len(FakeCrossEncoder.built) == 1
    r.rerank("q", ["a"], max_length=256)
    assert len(FakeCrossEncoder.built) == 2


def test_empty_documents_return_empty_without_loading(monkeypatch):
    _install(monkeypatch, adaptmem=FakeAdaptMem)
    assert SentenceTransformersReranker("ce").rerank("q", []) == []
    assert AdaptMemReranker("ce").rerank("q", []) == []
    assert FakeCrossEncoder.built[0].predict_kwargs is None


# ── backend selection is explicit ─────────────────────────────────────────────


def test_auto_prefers_adaptmem_when_it_supports_the_limits(monkeypatch):
    _install(monkeypatch, adaptmem=FakeAdaptMem)
    assert make_reranker("ce").backend == "adaptmem"


def test_auto_falls_back_with_info_when_adaptmem_missing(monkeypatch, caplog):
    _install(monkeypatch, adaptmem=None)
    with caplog.at_level(logging.INFO, logger="mnemonics"):
        r = make_reranker("ce")
    assert r.backend == "sentence-transformers"
    assert any("adaptmem is not installed" in m for m in caplog.messages)


def test_auto_falls_back_with_warning_when_adaptmem_is_too_old(monkeypatch, caplog):
    _install(monkeypatch, adaptmem=OldAdaptMem)
    with caplog.at_level(logging.WARNING, logger="mnemonics"):
        r = make_reranker("ce")
    assert r.backend == "sentence-transformers"
    warned = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert warned and "max_length, batch_size" in warned[0].getMessage()
    assert "9.9" in warned[0].getMessage()


def test_explicit_adaptmem_backend_never_falls_back(monkeypatch):
    _install(monkeypatch, adaptmem=OldAdaptMem)
    monkeypatch.setenv("MNEMONICS_RERANK_BACKEND", "adaptmem")
    with pytest.raises(RerankerIncompatible, match="upgrade adaptmem"):
        make_reranker("ce")
    _install(monkeypatch, adaptmem=None)
    with pytest.raises(RerankerUnavailable, match="not installed"):
        make_reranker("ce")


def test_explicit_sentence_transformers_backend_ignores_adaptmem(monkeypatch):
    _install(monkeypatch, adaptmem=FakeAdaptMem)
    assert make_reranker("ce", backend="sentence-transformers").backend == "sentence-transformers"


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("MNEMONICS_RERANK_BACKEND", "magic")
    with pytest.raises(ValueError, match="MNEMONICS_RERANK_BACKEND"):
        make_reranker("ce")


def test_incompatible_is_an_unavailable_is_a_runtime_error():
    assert issubclass(RerankerIncompatible, RerankerUnavailable)
    assert issubclass(RerankerUnavailable, RerankerError)
    assert issubclass(RerankerError, RuntimeError)


# ── failures are typed and visible ────────────────────────────────────────────


def test_load_failure_is_typed_and_chained(monkeypatch):
    class Boom:
        def __init__(self, name, **kwargs):
            raise OSError("no such model")

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=Boom))
    with pytest.raises(RerankerError, match="could not load") as ei:
        SentenceTransformersReranker("ce").rerank("q", ["a"])
    assert isinstance(ei.value.__cause__, OSError)


def test_missing_sentence_transformers_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RerankerUnavailable, match="sentence-transformers"):
        SentenceTransformersReranker("ce").rerank("q", ["a"])


def test_scoring_failure_is_typed_on_both_backends(monkeypatch):
    class BadCE(FakeCrossEncoder):
        def predict(self, pairs, **kwargs):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=BadCE))
    with pytest.raises(RerankerError, match="failed to score") as ei:
        SentenceTransformersReranker("ce").rerank("q", ["a"])
    assert isinstance(ei.value.__cause__, RuntimeError)

    class BadAM(FakeAdaptMem):
        def rerank(self, query, candidates, top_k=None, *, max_length=None, batch_size=None):
            raise RuntimeError("boom")

    monkeypatch.setitem(sys.modules, "adaptmem", SimpleNamespace(AdaptMem=BadAM))
    with pytest.raises(RerankerError, match="adaptmem reranker"):
        AdaptMemReranker("ce").rerank("q", ["a"])


# ── env parsing ───────────────────────────────────────────────────────────────


def test_env_int(monkeypatch):
    assert env_int("MNEMONICS_X_UNSET") is None
    monkeypatch.setenv("MNEMONICS_X", "")
    assert env_int("MNEMONICS_X") is None
    monkeypatch.setenv("MNEMONICS_X", "32")
    assert env_int("MNEMONICS_X") == 32
    monkeypatch.setenv("MNEMONICS_X", "lots")
    with pytest.raises(ValueError, match="MNEMONICS_X must be an integer"):
        env_int("MNEMONICS_X")
