"""E2E: preference augmentation across every production ingest surface.

Proves the P1 runtime gap is closed: extract_preferences → canonical ingest()
→ CLI / REST / MCP → storage → later retrieval. Also proves the opt-in default
(off), provenance tagging, dedup/supersede compatibility, and that assistant-fact
extraction is a SEPARATE opt-in that never fires implicitly.
"""

from unittest.mock import patch

import pytest

from mnemonics.ingest import ingest
from mnemonics.store import Store

# Reuse the server test harness helpers.
from tests.test_server import _mcp, http_call  # type: ignore

PREF_TEXT = "I prefer dark mode over light mode and I usually drink tea in the morning."
FACT_TURN = "[assistant] Your total is $240 and you saved 20% on the trip."


def _pref_rows(store: Store, ns: str) -> list[dict]:
    return [r for r in store.export_ns(ns) if r.get("meta", {}).get("kind") == "preference"]


def _fact_rows(store: Store, ns: str) -> list[dict]:
    return [r for r in store.export_ns(ns) if r.get("meta", {}).get("kind") == "fact"]


# ── 1. Canonical library API ────────────────────────────────────────────────


def test_library_augment_on_creates_provenance_tagged_pref(tmp_store):
    ingest([PREF_TEXT], tmp_store, ns="lib", augment_preferences=True)
    prefs = _pref_rows(tmp_store, "lib")
    assert len(prefs) == 1
    # Provenance: derived docs are distinguishable, not verbatim user text.
    assert prefs[0]["text"].startswith("User has mentioned:")
    assert prefs[0]["meta"]["kind"] == "preference"


def test_library_augment_off_by_default(tmp_store):
    ingest([PREF_TEXT], tmp_store, ns="lib")
    assert _pref_rows(tmp_store, "lib") == []


def test_library_augment_off_creates_no_fact(tmp_store):
    # Even with preferences on, assistant facts must NOT be extracted implicitly.
    ingest([FACT_TURN], tmp_store, ns="lib", augment_preferences=True)
    assert _fact_rows(tmp_store, "lib") == []


# ── 2. CLI surface ──────────────────────────────────────────────────────────


def _run_cli(*args):
    from mnemonics.cli import main

    with patch("sys.argv", ["mnemonics", *args]):
        try:
            main()
        except SystemExit:
            pass


def test_cli_augment_preferences_flag_flows_to_ingest(tmp_path):
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=2) as mock,
        patch(
            "sys.argv",
            ["mnemonics", "ingest", PREF_TEXT, "--augment-preferences", "--path", str(tmp_path)],
        ),
    ):
        from mnemonics.cli import main

        main()
    kw = mock.call_args[1]
    assert kw["augment_preferences"] is True
    assert kw["augment_assistant_facts"] is False  # separate, stays off


def test_cli_default_is_opt_out(tmp_path):
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=1) as mock,
        patch("sys.argv", ["mnemonics", "ingest", PREF_TEXT, "--path", str(tmp_path)]),
    ):
        from mnemonics.cli import main

        main()
    kw = mock.call_args[1]
    assert kw["augment_preferences"] is False
    assert kw["augment_assistant_facts"] is False


def test_cli_end_to_end_stores_and_retrieves(tmp_path):
    path = str(tmp_path / "db")
    _run_cli("ingest", PREF_TEXT, "--augment-preferences", "--ns", "cli", "--path", path)
    store = Store(path)
    prefs = _pref_rows(store, "cli")
    assert len(prefs) == 1
    # Retrieval recovers the derived preference by keyword (BM25 surface).
    hits = store.search_bm25("dark mode", ns="cli", top_k=5)
    assert any(h["meta"].get("kind") == "preference" for h in hits)


# ── 3. REST surface ─────────────────────────────────────────────────────────


def test_rest_augment_on(tmp_store):
    status, resp = http_call(
        tmp_store,
        "POST",
        "/ingest",
        {"texts": [PREF_TEXT], "ns": "rest", "augment_preferences": True},
    )
    assert status == 200
    assert len(_pref_rows(tmp_store, "rest")) == 1


def test_rest_default_off(tmp_store):
    status, _ = http_call(tmp_store, "POST", "/ingest", {"texts": [PREF_TEXT], "ns": "rest"})
    assert status == 200
    assert _pref_rows(tmp_store, "rest") == []


def test_rest_no_implicit_assistant_facts(tmp_store):
    http_call(
        tmp_store,
        "POST",
        "/ingest",
        {"texts": [FACT_TURN], "ns": "rest", "augment_preferences": True},
    )
    assert _fact_rows(tmp_store, "rest") == []


# ── 4. MCP surface ──────────────────────────────────────────────────────────


def _mcp_ingest(store, args: dict) -> list[dict]:
    return _mcp(
        store,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "mnemonics_ingest", "arguments": args},
        },
    )


def test_mcp_augment_on(tmp_store):
    _mcp_ingest(tmp_store, {"texts": [PREF_TEXT], "ns": "mcp", "augment_preferences": True})
    assert len(_pref_rows(tmp_store, "mcp")) == 1


def test_mcp_default_off(tmp_store):
    _mcp_ingest(tmp_store, {"texts": [PREF_TEXT], "ns": "mcp"})
    assert _pref_rows(tmp_store, "mcp") == []


def test_mcp_schema_exposes_augment_flags(tmp_store):
    resp = _mcp(tmp_store, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    tools = resp[0]["result"]["tools"]
    ingest_tool = next(t for t in tools if t["name"] == "mnemonics_ingest")
    props = ingest_tool["inputSchema"]["properties"]
    assert "augment_preferences" in props
    assert "augment_assistant_facts" in props


def test_mcp_no_implicit_assistant_facts(tmp_store):
    _mcp_ingest(tmp_store, {"texts": [FACT_TURN], "ns": "mcp", "augment_preferences": True})
    assert _fact_rows(tmp_store, "mcp") == []


# ── 5. Malformed / no-preference text ───────────────────────────────────────


def test_no_preference_text_creates_no_derived_row(tmp_store):
    ingest(["today is sunny. that is all."], tmp_store, ns="np", augment_preferences=True)
    assert _pref_rows(tmp_store, "np") == []


# ── 6. Dedup / supersession compatibility ───────────────────────────────────


def test_repeated_reconcile_ingest_does_not_duplicate_pref(tmp_store):
    from mnemonics.dedup import reconcile_ingest

    reconcile_ingest([PREF_TEXT], tmp_store, ns="dd", augment_preferences=True)
    first = len(_pref_rows(tmp_store, "dd"))
    # Re-ingesting the identical text via the conflict-aware path is a NOOP.
    res = reconcile_ingest([PREF_TEXT], tmp_store, ns="dd", augment_preferences=True)
    assert res["noop_skipped"], "identical re-ingest should NOOP"
    assert len(_pref_rows(tmp_store, "dd")) == first


def test_superseded_preference_drops_out_of_retrieval(tmp_store):
    ids = ingest([PREF_TEXT], tmp_store, ns="sup", augment_preferences=True, return_ids=True)
    # Mark every derived preference row superseded.
    for r in _pref_rows(tmp_store, "sup"):
        tmp_store.update_meta_key(r["id"], "status", "superseded")
    hits = tmp_store.search_bm25("dark mode", ns="sup", top_k=5)
    assert all(h["meta"].get("kind") != "preference" for h in hits)
    assert ids  # sanity: ingest returned ids


# ── 7. Retrieval-band fix interaction (vector path) ─────────────────────────


def test_pref_retrievable_via_vector_search(tmp_store):
    ingest([PREF_TEXT], tmp_store, ns="vec", augment_preferences=True)
    # Encode the preference gist and confirm the derived row is reachable.
    from mnemonics.ingest import _get_encoder

    qv = _get_encoder().encode(
        ["dark mode preference"], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    hits = tmp_store.search(qv, ns="vec", top_k=5)
    assert any(h["meta"].get("kind") == "preference" for h in hits)


@pytest.mark.parametrize("key", ["augment_preferences", "augment_assistant_facts"])
@pytest.mark.parametrize("value", ["false", 1, None])
def test_augmentation_requires_boolean(tmp_store, key, value):
    status, response = http_call(tmp_store, "POST", "/ingest", {"texts": [PREF_TEXT], key: value})
    assert status == 400
    assert "booleans" in response["error"]
    response = _mcp_ingest(tmp_store, {"texts": [PREF_TEXT], key: value})
    assert "booleans" in str(response)
    assert tmp_store.export_ns("default") == []


def test_canonical_augmentation_rejected_before_writes(tmp_store, tmp_path, capsys):
    args = {"texts": [PREF_TEXT], "canonical_key": "theme", "augment_preferences": True}
    status, response = http_call(tmp_store, "POST", "/ingest", args)
    assert status == 400
    assert "cannot be combined" in response["error"]
    assert "cannot be combined" in str(_mcp_ingest(tmp_store, args))
    assert tmp_store.export_ns("default") == []
    _run_cli(
        "ingest",
        PREF_TEXT,
        "--canonical-key",
        "theme",
        "--augment-preferences",
        "--path",
        str(tmp_path),
    )
    assert "cannot be combined" in capsys.readouterr().err
    assert Store(tmp_path).export_ns("default") == []


def test_mcp_reconcile_keeps_augmentation(tmp_store):
    args = {"texts": [PREF_TEXT], "ns": "reconcile", "reconcile": True, "augment_preferences": True}
    _mcp_ingest(tmp_store, args)
    assert len(_pref_rows(tmp_store, "reconcile")) == 1
    _mcp_ingest(tmp_store, args)
    assert len(_pref_rows(tmp_store, "reconcile")) == 1


def test_assistant_fact_opt_in_across_surfaces(tmp_store, tmp_path):
    args = {"texts": [FACT_TURN], "ns": "facts", "augment_assistant_facts": True}
    status, _ = http_call(tmp_store, "POST", "/ingest", args)
    assert status == 200
    assert len(_fact_rows(tmp_store, "facts")) == 1
    assert _pref_rows(tmp_store, "facts") == []
    args["ns"] = "mcp-facts"
    _mcp_ingest(tmp_store, args)
    assert len(_fact_rows(tmp_store, "mcp-facts")) == 1
    _run_cli("ingest", FACT_TURN, "--augment-facts", "--path", str(tmp_path))
    assert len(_fact_rows(Store(tmp_path), "default")) == 1
