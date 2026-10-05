from benchmarks.longmemeval_v2.mnemonics_memory import trajectory_chunks


def test_v2_trajectory_adapter_is_schema_only_and_preserves_provenance():
    trajectory = {
        "id": "traj-1",
        "domain": "web",
        "environment": "shop",
        "goal": "Update a shipping address",
        "outcome": "success",
        "start_url": "https://example.test/account",
        "states": [
            {
                "state_index": 0,
                "step": 1,
                "url": "https://example.test/account/addresses",
                "action": "click Edit",
                "thought": "Open the saved address",
                "accessibility_tree": "Shipping Addresses Edit 123 Main Street Save",
                "screenshot": "screenshots/traj-1/1.png",
            }
        ],
        # Deliberate benchmark-like forbidden fields: adapter must ignore them.
        "question_id": "SHOULD_NOT_APPEAR",
        "answer": "SECRET_GOLD",
        "answer_session_ids": ["SECRET_EVIDENCE"],
        "eval_function": "SECRET_EVAL",
    }
    texts, meta = trajectory_chunks(trajectory)
    joined = "\n".join(texts)
    assert texts
    assert "TRAJECTORY=traj-1" in joined
    assert "Shipping Addresses" in joined
    assert "SHOULD_NOT_APPEAR" not in joined
    assert "SECRET_GOLD" not in joined
    assert "SECRET_EVIDENCE" not in joined
    assert "SECRET_EVAL" not in joined
    assert all(m.get("trajectory_id") == "traj-1" for m in meta)
