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
