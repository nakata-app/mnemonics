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


def test_retrieve_planned_batches_embeddings_and_touches_final_rows_once(monkeypatch):
    calls: list[dict] = []
    encoded_batches: list[list[str]] = []
    touched: list[list[int]] = []

    class FakeEncoder:
        def encode(self, texts, **kwargs):
            batch = list(texts)
            encoded_batches.append(batch)
            return [[float(i), 0.0, 0.0] for i, _ in enumerate(batch, start=1)]

    class FakeStore:
        def touch_ids(self, ids):
            touched.append(list(ids))
            return len(ids)

    def fake_retrieve(*, query, **kwargs):
        calls.append({"query": query, **kwargs})
        if "provider-attempt" in query:
            return {
                "results": [
                    {"id": 7, "score": 0.8, "text": "provider attempt", "tier": 1},
                    {"id": 9, "score": 0.4, "text": "shared", "tier": 1},
                ]
            }
        return {
            "results": [
                {"id": 8, "score": 0.7, "text": "other", "tier": 1},
                {"id": 9, "score": 0.6, "text": "shared", "tier": 1},
            ]
        }

    monkeypatch.setattr(
        "mnemonics.query_plan._resolve_model_for_store",
        lambda model, store: "resolved-model",
    )
    monkeypatch.setattr("mnemonics.query_plan._get_encoder", lambda model: FakeEncoder())
    monkeypatch.setattr("mnemonics.query_plan.retrieve", fake_retrieve)

    result = retrieve_planned(
        "Inspect src/agent/provider-attempt.ts and runProviderAttempt",
        store=FakeStore(),
        top_k=2,
        max_queries=3,
    )

    assert len(encoded_batches) == 1
    assert encoded_batches[0] == [q["text"] for q in result["queries"]]
    assert len(calls) == len(result["queries"])
    assert all(call["touch"] is False for call in calls)
    assert all(call["model"] == "resolved-model" for call in calls)
    assert all(call["query_vector"] is not None for call in calls)
    assert touched == [[row["id"] for row in result["results"]]]
    assert result["queries"][0]["kind"] == "original"
    assert result["results"][0]["id"] in {7, 9}