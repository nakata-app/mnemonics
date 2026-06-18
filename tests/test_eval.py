"""Unit tests for mnemonics.eval — metrics and aggregation only.

run_eval() needs sentence-transformers + a downloaded model, so we don't
load it here; that path is exercised manually with the real eval set
(see ~/Projects/adaptmem-mnemonics-eval/data/).
"""
from __future__ import annotations

import math

import pytest

from mnemonics.eval import (
    aggregate,
    compare_table,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b", "c"], "a") == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["a", "b", "c"], "c") == pytest.approx(1 / 3)


def test_reciprocal_rank_miss():
    assert reciprocal_rank(["a", "b", "c"], "z") == 0.0


def test_recall_at_k_hit_inside_window():
    assert recall_at_k(["a", "b", "c", "d", "e"], "c", k=5) == 1.0


def test_recall_at_k_hit_outside_window():
    assert recall_at_k(["a", "b", "c", "d", "e", "z"], "z", k=5) == 0.0


def test_ndcg_at_k_rank_1_equals_one():
    assert ndcg_at_k(["a", "b", "c"], "a", k=10) == 1.0


def test_ndcg_at_k_rank_2_known_value():
    # 1 / log2(3) ≈ 0.6309
    assert ndcg_at_k(["x", "a", "y"], "a", k=10) == pytest.approx(1.0 / math.log2(3))


def test_ndcg_at_k_miss_returns_zero():
    assert ndcg_at_k(["a", "b"], "z", k=10) == 0.0


def test_ndcg_at_k_outside_window_returns_zero():
    hits = [str(i) for i in range(12)]
    assert ndcg_at_k(hits, "11", k=10) == 0.0


def test_aggregate_empty_input():
    agg = aggregate([])
    assert agg == {"n": 0, "mrr": 0.0, "r@5": 0.0, "r@10": 0.0, "ndcg@10": 0.0}


def test_aggregate_averages():
    pq = [
        {"rr": 1.0, "r5": 1.0, "r10": 1.0, "ndcg10": 1.0},
        {"rr": 0.5, "r5": 1.0, "r10": 1.0, "ndcg10": 0.6309},
        {"rr": 0.0, "r5": 0.0, "r10": 0.0, "ndcg10": 0.0},
    ]
    agg = aggregate(pq)
    assert agg["n"] == 3
    assert agg["mrr"] == pytest.approx(0.5)
    assert agg["r@5"] == pytest.approx(2 / 3)
    assert agg["r@10"] == pytest.approx(2 / 3)
    assert agg["ndcg@10"] == pytest.approx((1.0 + 0.6309 + 0.0) / 3)


def test_compare_table_single_encoder_no_delta_column():
    results = {
        "minilm": {
            "agg": {"mrr": 0.8, "r@5": 0.9, "r@10": 0.95, "ndcg@10": 0.85}
        }
    }
    out = compare_table(results)
    assert "minilm" in out
    assert "mrr" in out
    # single encoder → no delta annotation
    assert "(+" not in out and "(-" not in out


def test_compare_table_multi_encoder_renders_deltas():
    results = {
        "minilm":   {"agg": {"mrr": 0.50, "r@5": 0.80, "r@10": 0.90, "ndcg@10": 0.70}},
        "adaptmem": {"agg": {"mrr": 0.83, "r@5": 0.93, "r@10": 0.93, "ndcg@10": 0.86}},
    }
    out = compare_table(results)
    # second column should carry signed deltas vs the first (baseline)
    assert "+0.33" in out  # mrr delta
    assert "+0.13" in out  # r@5 delta


# ── run_eval with mocked encoder ─────────────────────────────────────────────

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


def _write_jsonl(path: Path, items: list):
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def _fake_encoder():
    enc = MagicMock()
    # Always return normalized unit vectors
    def encode(texts, **kw):
        n = len(texts)
        vecs = np.eye(n, 4, dtype="float32")
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs
    enc.encode.side_effect = encode
    return enc


def test_run_eval_vector_method(tmp_path):
    from mnemonics.eval import run_eval

    corpus = [{"id": f"c{i}", "text": f"chunk {i}"} for i in range(4)]
    queries = [{"query": "chunk 0", "relevant_id": "c0"}]
    _write_jsonl(tmp_path / "corpus.jsonl", corpus)
    _write_jsonl(tmp_path / "queries.jsonl", queries)

    with patch("mnemonics.eval._build_encoder", return_value=_fake_encoder()):
        result = run_eval(
            corpus_path=tmp_path / "corpus.jsonl",
            queries_path=tmp_path / "queries.jsonl",
            encoder="minilm",
            top_k=4,
            method="vector",
        )
    assert result["method"] == "vector"
    assert "agg" in result
    assert result["agg"]["n"] == 1


def test_run_eval_hybrid_method(tmp_path):
    from mnemonics.eval import run_eval

    corpus = [{"id": f"c{i}", "text": f"word{i} unique"} for i in range(4)]
    queries = [{"query": "word0", "relevant_id": "c0"}]
    _write_jsonl(tmp_path / "corpus.jsonl", corpus)
    _write_jsonl(tmp_path / "queries.jsonl", queries)

    with patch("mnemonics.eval._build_encoder", return_value=_fake_encoder()):
        result = run_eval(
            corpus_path=tmp_path / "corpus.jsonl",
            queries_path=tmp_path / "queries.jsonl",
            encoder="minilm",
            top_k=4,
            method="hybrid",
        )
    assert "+hybrid" in result["encoder"]
    assert result["agg"]["n"] == 1


def test_run_eval_unknown_method_raises(tmp_path):
    from mnemonics.eval import run_eval

    corpus = [{"id": "c0", "text": "text"}]
    queries = [{"query": "q", "relevant_id": "c0"}]
    _write_jsonl(tmp_path / "corpus.jsonl", corpus)
    _write_jsonl(tmp_path / "queries.jsonl", queries)

    with pytest.raises(ValueError, match="unknown method"):
        run_eval(
            corpus_path=tmp_path / "corpus.jsonl",
            queries_path=tmp_path / "queries.jsonl",
            method="invalid",
        )


def test_load_jsonl(tmp_path):
    from mnemonics.eval import _load_jsonl
    p = tmp_path / "data.jsonl"
    _write_jsonl(p, [{"a": 1}, {"b": 2}])
    rows = _load_jsonl(p)
    assert rows == [{"a": 1}, {"b": 2}]


def test_build_encoder_minilm():
    from mnemonics.eval import _build_encoder
    with patch("sentence_transformers.SentenceTransformer") as MockST:
        _build_encoder("minilm", None)
    MockST.assert_called_once_with("all-MiniLM-L6-v2")


def test_build_encoder_custom():
    from mnemonics.eval import _build_encoder
    with patch("sentence_transformers.SentenceTransformer") as MockST:
        _build_encoder("some/model", None)
    MockST.assert_called_once_with("some/model")


def test_build_encoder_adaptmem_no_path_raises():
    from mnemonics.eval import _build_encoder
    import os
    env = {k: v for k, v in os.environ.items() if k != "MNEMONICS_ADAPTMEM_PATH"}
    with patch.dict("os.environ", env, clear=True):
        with pytest.raises(ValueError, match="adaptmem encoder"):
            _build_encoder("adaptmem", model_path=None)


def test_build_encoder_adaptmem_with_path(tmp_path):
    from mnemonics.eval import _build_encoder
    (tmp_path / "model").mkdir()  # model subdir exists → use it
    with patch("sentence_transformers.SentenceTransformer") as MockST:
        _build_encoder("adaptmem", model_path=str(tmp_path))
    MockST.assert_called_once_with(str(tmp_path / "model"))


def test_bm25_rank_fts_operational_error(tmp_path):
    from mnemonics.eval import _build_bm25_index, _bm25_rank
    conn = _build_bm25_index(["some text"])
    chunk_ids = ["c0"]
    # Unmatched paren triggers OperationalError → returns []
    from unittest.mock import MagicMock as MM
    from mnemonics.store import Store
    with patch.object(Store, "_fts_sanitize", staticmethod(lambda q: "(((")):
        result = _bm25_rank(conn, "anything", chunk_ids, top_k=5)
    assert result == []
