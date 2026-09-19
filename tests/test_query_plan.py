from __future__ import annotations

from mnemonics.query_plan import PlannedQuery, _weighted_rrf, plan_queries, retrieve_planned


def test_plan_queries_preserves_original_and_extracts_code_signals():
    q = "Why does src/agent/provider-attempt.ts stall after runProviderAttempt?"
    plans = plan_queries(q, max_queries=4)

    assert plans[0] == PlannedQuery(q, 1.0, "original")
    assert any(p.kind == "paths" and "src/agent/provider-attempt.ts" in p.text for p in plans)
    assert any(p.kind == "symbols" and "runProviderAttempt" in p.text for p in plans)
    assert len(plans) <= 4


def test_plan_queries_is_bounded_and_deduplicated():
    plans = plan_queries("alpha alpha alpha beta gamma", max_queries=2)
    assert len(plans) <= 2
    assert len({p.text.lower() for p in plans}) == len(plans)


def test_weighted_rrf_rewards_cross_query_agreement():
    original = PlannedQuery("full query", 1.0, "original")
    symbols = PlannedQuery("FooBar", 0.9, "symbols")
    rows = _weighted_rrf(
        [
            (original, [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.8}]),
            (symbols, [{"id": 2, "score": 0.7}, {"id": 3, "score": 0.6}]),
        ],
        top_k=3,
    )
    assert rows[0]["id"] == 2
    assert rows[0]["matched_queries"] == ["original", "symbols"]


def test_retrieve_planned_runs_bounded_subqueries_and_fuses(monkeypatch):
    calls: list[str] = []

    def fake_retrieve(*, query, **kwargs):
        calls.append(query)
        if "provider-attempt" in query:
            return {"results": [{"id": 7, "score": 0.8, "text": "provider attempt", "tier": 1}]}
        return {"results": [{"id": 8, "score": 0.7, "text": "other", "tier": 1}]}

    monkeypatch.setattr("mnemonics.query_plan.retrieve", fake_retrieve)
    result = retrieve_planned(
        "Inspect src/agent/provider-attempt.ts and runProviderAttempt",
        store=object(),  # fake_retrieve ignores the store
        top_k=2,
        max_queries=3,
    )

    assert 1 <= len(calls) <= 3
    assert result["queries"][0]["kind"] == "original"
    assert result["results"][0]["id"] == 7