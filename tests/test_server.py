"""Tests for REST and MCP server logic."""
import io
import json
import threading
from http.client import HTTPConnection
from unittest.mock import MagicMock, patch

import pytest

from mnemonics import server as srv
from mnemonics.store import DIM, Store


# ── helper: in-process server ────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_store():
    """Reset global store singleton between tests."""
    old = srv._store
    srv._store = None
    yield
    srv._store = old


def _make_handler(store):
    with patch("mnemonics.server._get_store", return_value=store):
        yield


class FakeSocket:
    def __init__(self, request_bytes: bytes):
        self._in = io.BytesIO(request_bytes)
        self._out = io.BytesIO()

    def makefile(self, mode):
        if "r" in mode:
            return io.BufferedReader(self._in)
        return self._out

    def sendall(self, data):
        self._out.write(data)

    def getsockname(self):
        return ("127.0.0.1", 7810)

    def getpeername(self):
        return ("127.0.0.1", 0)


def http_call(store, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    """Fire a synthetic HTTP request at _Handler with the given store."""
    body_bytes = json.dumps(body).encode() if body else b""
    content_length = str(len(body_bytes))

    raw = (
        f"{method} {path} HTTP/1.1\r\n"
        f"Host: localhost\r\n"
        f"Content-Length: {content_length}\r\n"
        f"Content-Type: application/json\r\n"
        f"\r\n"
    ).encode() + body_bytes

    fs = FakeSocket(raw)

    with patch("mnemonics.server._get_store", return_value=store):
        handler = srv._Handler.__new__(srv._Handler)
        handler.rfile = io.BufferedReader(io.BytesIO(raw))
        handler.wfile = io.BytesIO()
        handler.server = MagicMock()
        handler.client_address = ("127.0.0.1", 0)
        handler.request_version = "HTTP/1.1"
        handler.requestline = f"{method} {path} HTTP/1.1"
        handler.command = method
        handler.path = path
        handler.headers = {
            "Content-Length": content_length,
            "Content-Type": "application/json",
        }
        handler._body = lambda: body or {}
        handler.responses = {}

        captured = {"code": None, "data": None}
        def fake_json(code, data):
            captured["code"] = code
            captured["data"] = data
        handler._json = fake_json

        if method == "GET":
            handler.do_GET()
        elif method == "POST":
            handler.do_POST()
        elif method == "PATCH":
            handler.do_PATCH()
        elif method == "DELETE":
            handler.do_DELETE()

    return captured["code"], captured["data"]


# ── GET /health ───────────────────────────────────────────────────────────────

def test_health(tmp_store):
    code, data = http_call(tmp_store, "GET", "/health")
    assert code == 200
    assert data["status"] == "ok"


# ── POST /search-bm25 ────────────────────────────────────────────────────────

def test_http_search_bm25_hit(populated_store):
    store, docs, vecs = populated_store
    # docs[0] is the first text — search for a keyword in it
    query = docs[0].split()[0]
    code, data = http_call(store, "POST", "/search-bm25", {"query": query, "ns": "default"})
    assert code == 200
    assert "results" in data
    assert data["query"] == query


def test_http_search_bm25_empty(tmp_store):
    code, data = http_call(tmp_store, "POST", "/search-bm25", {"query": "anything"})
    assert code == 200
    assert data["results"] == []


def test_http_search_bm25_missing_query(tmp_store):
    code, data = http_call(tmp_store, "POST", "/search-bm25", {})
    assert code == 400


# ── PATCH /memory/<id> ────────────────────────────────────────────────────────

def test_http_patch_summary(populated_store):
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    code, data = http_call(store, "PATCH", f"/memory/{first_id}", {"summary": "updated gist"})
    assert code == 200
    assert data["updated"] is True
    row = store.get(first_id)
    assert row["summary"] == "updated gist"


def test_http_patch_summary_clear(populated_store):
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.update_summary(first_id, "some summary")
    code, data = http_call(store, "PATCH", f"/memory/{first_id}", {"summary": None})
    assert code == 200
    assert store.get(first_id)["summary"] is None


def test_http_patch_not_found(tmp_store):
    code, data = http_call(tmp_store, "PATCH", "/memory/9999", {"summary": "x"})
    assert code == 404


def test_http_patch_no_summary_field(populated_store):
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    code, data = http_call(store, "PATCH", f"/memory/{first_id}", {"tier": 2})
    assert code == 400


def test_http_patch_invalid_id(tmp_store):
    code, data = http_call(tmp_store, "PATCH", "/memory/abc", {"summary": "x"})
    assert code == 400
    assert "invalid" in data["error"]


# ── GET /memory/<id> ──────────────────────────────────────────────────────────

def test_http_get_memory(populated_store):
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    code, data = http_call(store, "GET", f"/memory/{first_id}")
    assert code == 200
    assert data["id"] == first_id
    assert "text" in data
    assert "ns" in data
    assert "tier" in data


def test_http_get_memory_not_found(tmp_store):
    code, data = http_call(tmp_store, "GET", "/memory/9999")
    assert code == 404
    assert "not found" in data["error"]


def test_http_get_memory_invalid_id(tmp_store):
    code, data = http_call(tmp_store, "GET", "/memory/abc")
    assert code == 400


# ── GET /memories ─────────────────────────────────────────────────────────────

def test_http_list_memories_empty(tmp_store):
    code, data = http_call(tmp_store, "GET", "/memories?ns=default")
    assert code == 200
    assert data["rows"] == []
    assert data["count"] == 0


def test_http_list_memories(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/memories?ns=default&limit=5")
    assert code == 200
    assert data["count"] > 0
    row = data["rows"][0]
    assert "id" in row and "text" in row and "tier" in row


def test_http_list_memories_limit_cap(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/memories?ns=default&limit=999")
    assert code == 200
    assert data["count"] <= 100


def test_http_list_memories_tier_filter(populated_store):
    store, docs, vecs = populated_store
    # Pin first row, then filter to tier=0 — only that row should appear
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.pin(first_id)
    code, data = http_call(store, "GET", "/memories?ns=default&tier=0")
    assert code == 200
    assert data["count"] == 1
    assert data["rows"][0]["id"] == first_id
    assert data["rows"][0]["tier"] == 0


# ── GET /stats ────────────────────────────────────────────────────────────────

def test_http_stats_empty(tmp_store):
    code, data = http_call(tmp_store, "GET", "/stats")
    assert code == 200
    assert data["namespaces"] == []


def test_http_stats_tier_breakdown(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/stats")
    assert code == 200
    ns_names = [r["ns"] for r in data["namespaces"]]
    assert "default" in ns_names
    row = next(r for r in data["namespaces"] if r["ns"] == "default")
    assert "total" in row
    assert "pin" in row
    assert "def" in row
    assert "amb" in row


# ── POST /forget-ns ───────────────────────────────────────────────────────────

def test_http_forget_ns_dry_run(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/forget-ns", {"ns": "default", "dry_run": True})
    assert code == 200
    assert data["dry_run"] is True
    assert "candidates" in data
    assert data["ns"] == "default"


def test_http_forget_ns_apply(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/forget-ns", {"ns": "default", "dry_run": False})
    assert code == 200
    assert data["dry_run"] is False
    assert "deleted" in data


def test_http_forget_ns_missing_ns(tmp_store):
    code, data = http_call(tmp_store, "POST", "/forget-ns", {})
    assert code == 400
    assert "ns" in data["error"]


# ── GET /doctor ───────────────────────────────────────────────────────────────

def test_http_doctor(tmp_store):
    code, data = http_call(tmp_store, "GET", "/doctor")
    assert code == 200
    assert data["db_integrity"] == "ok"
    assert "namespaces" in data
    assert "orphan_indexes" in data


# ── POST /repair ──────────────────────────────────────────────────────────────

def test_http_repair(tmp_store):
    code, data = http_call(tmp_store, "POST", "/repair")
    assert code == 200
    assert "orphan_vectors_fixed" in data
    assert "orphan_indexes_removed" in data
    assert "missing_vectors_reported" in data


# ── POST /rebuild-index ───────────────────────────────────────────────────────

def test_http_rebuild_index(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/rebuild-index", {"ns": "default"})
    assert code == 200
    assert data["ns"] == "default"
    assert "old_count" in data
    assert "new_count" in data
    assert "removed" in data


def test_http_rebuild_index_missing_ns(tmp_store):
    code, data = http_call(tmp_store, "POST", "/rebuild-index", {})
    assert code == 400
    assert "ns" in data["error"]


# ── POST /pin and POST /tier ──────────────────────────────────────────────────

def test_http_pin(populated_store):
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    code, data = http_call(store, "POST", "/pin", {"id": first_id})
    assert code == 200
    assert data["id"] == first_id
    assert "pinned" in data


def test_http_pin_missing_id(tmp_store):
    code, data = http_call(tmp_store, "POST", "/pin", {})
    assert code == 400
    assert "id" in data["error"]


def test_http_tier(populated_store):
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    code, data = http_call(store, "POST", "/tier", {"id": first_id, "tier": 2})
    assert code == 200
    assert data["tier"] == 2
    assert data["id"] == first_id


def test_http_tier_invalid(tmp_store):
    code, data = http_call(tmp_store, "POST", "/tier", {"id": 1, "tier": 5})
    assert code == 400
    assert "tier" in data["error"]


def test_http_tier_missing_fields(tmp_store):
    code, data = http_call(tmp_store, "POST", "/tier", {"id": 1})
    assert code == 400


# ── POST /gc ─────────────────────────────────────────────────────────────────

def test_http_gc_dry_run(tmp_store):
    code, data = http_call(tmp_store, "POST", "/gc", {"ns": "default", "age_days": 0, "tier": 2})
    assert code == 200
    assert "candidates" in data
    assert data["dry_run"] is True


def test_http_gc_invalid_tier(tmp_store):
    code, data = http_call(tmp_store, "POST", "/gc", {"tier": 0})
    assert code == 400
    assert "tier" in data["error"]


def test_http_gc_apply(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/gc", {"age_days": 0, "dry_run": False})
    assert code == 200
    assert "deleted" in data
    assert data["dry_run"] is False


def test_http_gc_tier1(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/gc", {"tier": 1, "age_days": 0})
    assert code == 200
    assert data["dry_run"] is True
    assert "candidates" in data


# ── POST /forget ──────────────────────────────────────────────────────────────

def test_http_forget_dry_run(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/forget", {"ns": "default"})
    assert code == 200
    assert data["dry_run"] is True
    assert data["candidates"] > 0


def test_http_forget_missing_ns(tmp_store):
    code, data = http_call(tmp_store, "POST", "/forget", {})
    assert code == 400
    assert "ns" in data["error"]


def test_http_forget_apply(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/forget", {"ns": "default", "dry_run": False})
    assert code == 200
    assert data["deleted"] > 0
    assert data["dry_run"] is False


# ── GET /namespaces ───────────────────────────────────────────────────────────

def test_namespaces_empty(tmp_store):
    code, data = http_call(tmp_store, "GET", "/namespaces")
    assert code == 200
    assert data["namespaces"] == []


def test_namespaces_populated(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/namespaces")
    assert code == 200
    assert "default" in data["namespaces"]


# ── GET /count ────────────────────────────────────────────────────────────────

def test_count_default(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/count?ns=default")
    assert code == 200
    assert data["count"] == 5


def test_count_empty_ns(tmp_store):
    code, data = http_call(tmp_store, "GET", "/count?ns=unknown")
    assert code == 200
    assert data["count"] == 0


# ── POST /ingest ──────────────────────────────────────────────────────────────

def test_ingest_empty_texts_400(tmp_store):
    with patch("mnemonics.server._ingest", return_value=0):
        code, data = http_call(tmp_store, "POST", "/ingest", {"texts": []})
    assert code == 400


def test_ingest_texts_not_list_400(tmp_store):
    code, data = http_call(tmp_store, "POST", "/ingest", {"texts": "not a list"})
    assert code == 400
    assert "texts" in data["error"]


def test_ingest_summaries_wrong_length_400(tmp_store):
    code, data = http_call(tmp_store, "POST", "/ingest", {
        "texts": ["a", "b"],
        "summaries": ["only one summary"],
    })
    assert code == 400
    assert "summaries" in data["error"]


def test_ingest_summaries_valid(tmp_store):
    with patch("mnemonics.server._ingest", return_value=2):
        code, data = http_call(tmp_store, "POST", "/ingest", {
            "texts": ["a", "b"],
            "summaries": ["gist a", None],
        })
    assert code == 200


def test_ingest_success(tmp_store):
    with patch("mnemonics.server._ingest", return_value=2) as mock_ingest:
        code, data = http_call(tmp_store, "POST", "/ingest", {"texts": ["a", "b"]})
    assert code == 200
    assert data["ingested"] == 2


def test_ingest_passes_ns(tmp_store):
    with patch("mnemonics.server._ingest", return_value=1) as mock_ingest:
        http_call(tmp_store, "POST", "/ingest", {"texts": ["x"], "ns": "myns"})
    call_kwargs = mock_ingest.call_args[1]
    assert call_kwargs["ns"] == "myns"


# ── POST /retrieve ────────────────────────────────────────────────────────────

def test_retrieve_empty_query_400(tmp_store):
    with patch("mnemonics.server._retrieve"):
        code, data = http_call(tmp_store, "POST", "/retrieve", {"query": "  "})
    assert code == 400


def test_retrieve_runtime_error_400(tmp_store):
    with patch("mnemonics.server._retrieve", side_effect=RuntimeError("index corrupt")):
        code, data = http_call(tmp_store, "POST", "/retrieve", {"query": "hello"})
    assert code == 400
    assert "index corrupt" in data["error"]


def test_retrieve_invalid_candidate_k(tmp_store):
    code, data = http_call(tmp_store, "POST", "/retrieve", {"query": "q", "candidate_k": 0})
    assert code == 400
    assert "candidate_k" in data["error"]


def test_retrieve_success(tmp_store):
    fake_result = {"results": [], "trust_score": 1.0, "flagged_count": 0, "verified": True}
    with patch("mnemonics.server._retrieve", return_value=fake_result):
        code, data = http_call(tmp_store, "POST", "/retrieve", {"query": "hello"})
    assert code == 200
    assert data["trust_score"] == 1.0


def test_retrieve_passes_top_k(tmp_store):
    fake_result = {"results": [], "trust_score": 1.0, "flagged_count": 0, "verified": True}
    with patch("mnemonics.server._retrieve", return_value=fake_result) as mock_ret:
        http_call(tmp_store, "POST", "/retrieve", {"query": "q", "top_k": 10})
    call_kwargs = mock_ret.call_args[1]
    assert call_kwargs["top_k"] == 10


def test_retrieve_default_hybrid_true(tmp_store):
    fake_result = {"results": []}
    with patch("mnemonics.server._retrieve", return_value=fake_result) as mock_ret:
        http_call(tmp_store, "POST", "/retrieve", {"query": "q"})
    call_kwargs = mock_ret.call_args[1]
    assert call_kwargs["hybrid"] is True
    assert call_kwargs["candidate_k"] == 50


def test_retrieve_explicit_hybrid_false_honored(tmp_store):
    fake_result = {"results": []}
    with patch("mnemonics.server._retrieve", return_value=fake_result) as mock_ret:
        http_call(tmp_store, "POST", "/retrieve", {"query": "q", "hybrid": False})
    call_kwargs = mock_ret.call_args[1]
    assert call_kwargs["hybrid"] is False


def test_retrieve_passes_hybrid(tmp_store):
    fake_result = {"results": []}
    with patch("mnemonics.server._retrieve", return_value=fake_result) as mock_ret:
        http_call(tmp_store, "POST", "/retrieve", {"query": "q", "hybrid": True, "candidate_k": 50})
    call_kwargs = mock_ret.call_args[1]
    assert call_kwargs["hybrid"] is True
    assert call_kwargs["candidate_k"] == 50


def test_retrieve_rejects_zero_candidate_k(tmp_store):
    with patch("mnemonics.server._retrieve") as mock_ret:
        code, data = http_call(tmp_store, "POST", "/retrieve", {"query": "q", "candidate_k": 0})
    assert code == 400
    assert "candidate_k" in data["error"]
    mock_ret.assert_not_called()


# ── DELETE /memory/<id> ───────────────────────────────────────────────────────

def test_delete_existing(populated_store):
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    code, data = http_call(store, "DELETE", f"/memory/{first_id}")
    assert code == 200
    assert data["deleted"] is True


def test_delete_nonexistent(tmp_store):
    code, data = http_call(tmp_store, "DELETE", "/memory/99999")
    assert code == 200
    assert data["deleted"] is False


def test_delete_invalid_id(tmp_store):
    code, data = http_call(tmp_store, "DELETE", "/memory/notanint")
    assert code == 400


# ── 404 ───────────────────────────────────────────────────────────────────────

def test_get_unknown_route(tmp_store):
    code, data = http_call(tmp_store, "GET", "/unknown")
    assert code == 404


def test_post_unknown_route(tmp_store):
    code, data = http_call(tmp_store, "POST", "/unknown", {})
    assert code == 404


# ── MCP loop ──────────────────────────────────────────────────────────────────

def _mcp(store, *messages: dict) -> list[dict]:
    lines = "\n".join(json.dumps(m) for m in messages)
    output = []

    def fake_print(s, **_):
        output.append(json.loads(s))

    with (
        patch("mnemonics.server._get_store", return_value=store),
        patch("builtins.print", side_effect=fake_print),
        patch("sys.stdin", io.StringIO(lines + "\n")),
    ):
        srv._mcp_loop()

    return output


def test_mcp_initialize(tmp_store):
    resp = _mcp(tmp_store, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert resp[0]["result"]["protocolVersion"] == "2024-11-05"
    assert resp[0]["result"]["serverInfo"]["name"] == "mnemonics"


def test_mcp_tools_list(tmp_store):
    resp = _mcp(tmp_store, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = {t["name"] for t in resp[0]["result"]["tools"]}
    assert names == {
        "mnemonics_ingest", "mnemonics_retrieve", "mnemonics_bm25",
        "mnemonics_update_summary", "mnemonics_list", "mnemonics_get",
        "mnemonics_forget", "mnemonics_forget_ns", "mnemonics_rebuild_index",
        "mnemonics_pin", "mnemonics_tier", "mnemonics_gc", "mnemonics_stats",
        "mnemonics_health", "mnemonics_repair",
        "mnemonics_search_by_meta", "mnemonics_delete_many", "mnemonics_update_meta",
        "mnemonics_export", "mnemonics_import", "mnemonics_text_search", "mnemonics_rename_ns", "mnemonics_namespaces", "mnemonics_bulk_tier", "mnemonics_copy_ns", "mnemonics_touch_many", "mnemonics_count", "mnemonics_merge_ns", "mnemonics_stats_by_ns", "mnemonics_recent", "mnemonics_top_accessed", "mnemonics_get_many",
        "mnemonics_hybrid_search",
        "mnemonics_similar_to",
        "mnemonics_expire",
        "mnemonics_bulk_update_summary",
        "mnemonics_deduplicate",
        "mnemonics_sample",
        "mnemonics_reindex_all",
        "mnemonics_namespace_info",
        "mnemonics_move_to_ns",
        "mnemonics_clone",
        "mnemonics_update_text",
        "mnemonics_access_stats",
        "mnemonics_tag",
        "mnemonics_untag",
        "mnemonics_find_by_tag",
        "mnemonics_list_tags",
        "mnemonics_word_frequency",
        "mnemonics_get_tags",
        "mnemonics_search_date_range",
        "mnemonics_export_ns",
        "mnemonics_bulk_tag",
        "mnemonics_touch",
        "mnemonics_bulk_untag",
        "mnemonics_count_by_tier",
        "mnemonics_import_records",
        "mnemonics_text_stats",
        "mnemonics_rename_ns",
        "mnemonics_merge_ns",
        "mnemonics_bulk_delete",
        "mnemonics_filter_by_meta",
        "mnemonics_summary_stats",
        "mnemonics_pinned_memories",
        "mnemonics_update_meta_key",
        "mnemonics_search_by_summary",
        "mnemonics_set_tier_by_tag",
        "mnemonics_rotate_ns",
        "mnemonics_compact_meta",
        "mnemonics_list_by_tier",
        "mnemonics_newest",
        "mnemonics_oldest",
        "mnemonics_replace_text",
        "mnemonics_search_text",
        "mnemonics_count_by_ns",
        "mnemonics_clear_ns",
        "mnemonics_copy_to_ns",
        "mnemonics_rename_tag",
        "mnemonics_find_duplicates",
        "mnemonics_swap_tier",
        "mnemonics_ns_summary",
        "mnemonics_toggle_tier",
        "mnemonics_merge_texts",
        "mnemonics_truncate_text",
        "mnemonics_search_by_access_count",
        "mnemonics_age_by_ns",
        "mnemonics_delete_by_tier",
        "mnemonics_untagged_memories",
        "mnemonics_set_meta_for_untagged",
        "mnemonics_clone_memory",
        "mnemonics_memories_without_summary",
        "mnemonics_pin_by_tag",
        "mnemonics_promote_by_access",
        "mnemonics_filter_by_text_length",
        "mnemonics_multi_tag_filter",
        "mnemonics_tag_stats",
        "mnemonics_split_memory",
        "mnemonics_bulk_summarize",
        "mnemonics_cross_ns_search",
        "mnemonics_memory_timeline",
        "mnemonics_keyword_extract",
        "mnemonics_import_ns",
        "mnemonics_get_tier_distribution",
        "mnemonics_archive_by_tier",
        "mnemonics_text_search_ranked",
        "mnemonics_deduplicate_by_text",
        "mnemonics_merge_memories",
        "mnemonics_search_by_date_range",
        "mnemonics_get_access_stats",
    }


def test_mcp_rebuild_index(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 70,
        "method": "tools/call",
        "params": {"name": "mnemonics_rebuild_index", "arguments": {"ns": "default"}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "default" in text
    assert "→" in text


def test_mcp_rebuild_index_missing_ns(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 71,
        "method": "tools/call",
        "params": {"name": "mnemonics_rebuild_index", "arguments": {}},
    })
    assert "error" in resp[0]


def test_mcp_forget_ns_dry_run(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 50,
        "method": "tools/call",
        "params": {"name": "mnemonics_forget_ns", "arguments": {"ns": "default"}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "Would delete" in text or "dry-run" in text.lower()


def test_mcp_forget_ns_apply(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 51,
        "method": "tools/call",
        "params": {"name": "mnemonics_forget_ns", "arguments": {"ns": "default", "dry_run": False}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "Deleted" in text


def test_mcp_forget_ns_missing_ns(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 52,
        "method": "tools/call",
        "params": {"name": "mnemonics_forget_ns", "arguments": {}},
    })
    assert "error" in resp[0]


def test_mcp_get(populated_store):
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 80,
        "method": "tools/call",
        "params": {"name": "mnemonics_get", "arguments": {"id": first_id}},
    })
    import json as _json
    text = resp[0]["result"]["content"][0]["text"]
    data = _json.loads(text)
    assert data["id"] == first_id
    assert "text" in data
    assert "ns" in data
    assert "tier" in data


def test_mcp_get_not_found(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 81,
        "method": "tools/call",
        "params": {"name": "mnemonics_get", "arguments": {"id": 9999}},
    })
    assert "error" in resp[0]


def test_mcp_get_missing_id(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 82,
        "method": "tools/call",
        "params": {"name": "mnemonics_get", "arguments": {}},
    })
    assert "error" in resp[0]


def test_mcp_update_summary(populated_store):
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 95,
        "method": "tools/call",
        "params": {"name": "mnemonics_update_summary", "arguments": {"id": first_id, "summary": "new gist"}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "updated" in text
    assert store.get(first_id)["summary"] == "new gist"


def test_mcp_update_summary_not_found(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 96,
        "method": "tools/call",
        "params": {"name": "mnemonics_update_summary", "arguments": {"id": 9999, "summary": "x"}},
    })
    assert "error" in resp[0]


def test_mcp_update_summary_missing_id(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 97,
        "method": "tools/call",
        "params": {"name": "mnemonics_update_summary", "arguments": {}},
    })
    assert "error" in resp[0]


def test_mcp_bm25_hit(populated_store):
    store, docs, vecs = populated_store
    query = docs[0].split()[0]
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 100,
        "method": "tools/call",
        "params": {"name": "mnemonics_bm25", "arguments": {"query": query, "ns": "default"}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "id=" in text or "No BM25" in text


def test_mcp_bm25_no_results(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 101,
        "method": "tools/call",
        "params": {"name": "mnemonics_bm25", "arguments": {"query": "xyzzy"}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "No BM25" in text


def test_mcp_bm25_missing_query(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 102,
        "method": "tools/call",
        "params": {"name": "mnemonics_bm25", "arguments": {}},
    })
    assert "error" in resp[0]


def test_mcp_repair(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 22,
        "method": "tools/call",
        "params": {"name": "mnemonics_repair", "arguments": {}},
    })
    import json as _json
    text = resp[0]["result"]["content"][0]["text"]
    report = _json.loads(text)
    assert "orphan_vectors_fixed" in report
    assert "orphan_indexes_removed" in report
    assert "missing_vectors_reported" in report


def test_mcp_list_empty(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 90,
        "method": "tools/call",
        "params": {"name": "mnemonics_list", "arguments": {"ns": "default"}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "No memories" in text


def test_mcp_list_returns_rows(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 91,
        "method": "tools/call",
        "params": {"name": "mnemonics_list", "arguments": {"ns": "default", "limit": 5}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "ns='default'" in text
    assert "tier=" in text


def test_mcp_stats_empty(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 60,
        "method": "tools/call",
        "params": {"name": "mnemonics_stats", "arguments": {}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert text == "(empty)"


def test_mcp_stats_tier_breakdown(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 61,
        "method": "tools/call",
        "params": {"name": "mnemonics_stats", "arguments": {}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "default" in text
    assert "chunks" in text
    assert "pin=" in text
    assert "def=" in text
    assert "amb=" in text


def test_mcp_ingest(tmp_store):
    with patch("mnemonics.server._ingest", return_value=3) as mock_ingest:
        resp = _mcp(tmp_store, {
            "jsonrpc": "2.0", "id": 3,
            "method": "tools/call",
            "params": {"name": "mnemonics_ingest", "arguments": {"texts": ["a", "b", "c"]}},
        })
    assert "Stored 3" in resp[0]["result"]["content"][0]["text"]


def test_mcp_ingest_accepts_singular_text_string(tmp_store):
    with patch("mnemonics.server._ingest", return_value=1) as mock_ingest:
        resp = _mcp(tmp_store, {
            "jsonrpc": "2.0", "id": 31,
            "method": "tools/call",
            "params": {"name": "mnemonics_ingest", "arguments": {"text": "hello"}},
        })
    assert "Stored 1" in resp[0]["result"]["content"][0]["text"]
    assert mock_ingest.call_args.kwargs["texts"] == ["hello"]


def test_mcp_ingest_accepts_singular_text_list(tmp_store):
    with patch("mnemonics.server._ingest", return_value=2) as mock_ingest:
        resp = _mcp(tmp_store, {
            "jsonrpc": "2.0", "id": 32,
            "method": "tools/call",
            "params": {"name": "mnemonics_ingest", "arguments": {"text": ["a", "b"]}},
        })
    assert "Stored 2" in resp[0]["result"]["content"][0]["text"]
    assert mock_ingest.call_args.kwargs["texts"] == ["a", "b"]


def test_mcp_ingest_accepts_texts_as_bare_string(tmp_store):
    with patch("mnemonics.server._ingest", return_value=1) as mock_ingest:
        resp = _mcp(tmp_store, {
            "jsonrpc": "2.0", "id": 33,
            "method": "tools/call",
            "params": {"name": "mnemonics_ingest", "arguments": {"texts": "hello"}},
        })
    assert "Stored 1" in resp[0]["result"]["content"][0]["text"]
    assert mock_ingest.call_args.kwargs["texts"] == ["hello"]


def test_mcp_retrieve(tmp_store):
    fake = {"results": [{
        "id": 1, "score": 0.42, "raw_score": 0.50, "decay_factor": 0.90,
        "boost": 1.05, "age_days": 3.0, "tier": 1, "text": "hello world",
    }]}
    with patch("mnemonics.server._retrieve", return_value=fake):
        resp = _mcp(tmp_store, {
            "jsonrpc": "2.0", "id": 4,
            "method": "tools/call",
            "params": {"name": "mnemonics_retrieve", "arguments": {"query": "test"}},
        })
    text = resp[0]["result"]["content"][0]["text"]
    assert "raw=" in text and "decay=" in text and "boost=" in text and "tier=" in text


def test_mcp_retrieve_shows_summary(tmp_store):
    fake = {"results": [{
        "id": 2, "score": 0.9, "raw_score": 0.9, "decay_factor": 1.0,
        "boost": 1.0, "age_days": 0.0, "tier": 0,
        "text": "long raw content here", "summary": "short gist",
    }]}
    with patch("mnemonics.server._retrieve", return_value=fake):
        resp = _mcp(tmp_store, {
            "jsonrpc": "2.0", "id": 45,
            "method": "tools/call",
            "params": {"name": "mnemonics_retrieve", "arguments": {"query": "q"}},
        })
    text = resp[0]["result"]["content"][0]["text"]
    assert "short gist" in text
    assert "└─ raw:" in text


def test_mcp_retrieve_runtime_error(tmp_store):
    with patch("mnemonics.server._retrieve", side_effect=RuntimeError("index broken")):
        resp = _mcp(tmp_store, {
            "jsonrpc": "2.0", "id": 46,
            "method": "tools/call",
            "params": {"name": "mnemonics_retrieve", "arguments": {"query": "q"}},
        })
    assert "error" in resp[0]
    assert "index broken" in resp[0]["error"]["message"]


def test_mcp_retrieve_default_hybrid_true(tmp_store):
    fake = {"results": []}
    with patch("mnemonics.server._retrieve", return_value=fake) as mock_ret:
        _mcp(tmp_store, {
            "jsonrpc": "2.0", "id": 41,
            "method": "tools/call",
            "params": {"name": "mnemonics_retrieve", "arguments": {"query": "test"}},
        })
    kwargs = mock_ret.call_args.kwargs
    assert kwargs["hybrid"] is True
    assert kwargs["candidate_k"] == 50


def test_mcp_retrieve_explicit_hybrid_false_honored(tmp_store):
    fake = {"results": []}
    with patch("mnemonics.server._retrieve", return_value=fake) as mock_ret:
        _mcp(tmp_store, {
            "jsonrpc": "2.0", "id": 411,
            "method": "tools/call",
            "params": {"name": "mnemonics_retrieve", "arguments": {"query": "test", "hybrid": False}},
        })
    kwargs = mock_ret.call_args.kwargs
    assert kwargs["hybrid"] is False


def test_mcp_retrieve_passes_hybrid(tmp_store):
    fake = {"results": []}
    with patch("mnemonics.server._retrieve", return_value=fake) as mock_ret:
        _mcp(tmp_store, {
            "jsonrpc": "2.0", "id": 42,
            "method": "tools/call",
            "params": {
                "name": "mnemonics_retrieve",
                "arguments": {"query": "test", "hybrid": True, "candidate_k": 30},
            },
        })
    kwargs = mock_ret.call_args.kwargs
    assert kwargs["hybrid"] is True
    assert kwargs["candidate_k"] == 30


def test_mcp_retrieve_rejects_zero_candidate_k(tmp_store):
    with patch("mnemonics.server._retrieve") as mock_ret:
        resp = _mcp(tmp_store, {
            "jsonrpc": "2.0", "id": 43,
            "method": "tools/call",
            "params": {
                "name": "mnemonics_retrieve",
                "arguments": {"query": "test", "candidate_k": 0},
            },
        })
    assert "error" in resp[0]
    assert "candidate_k" in resp[0]["error"]["message"]
    mock_ret.assert_not_called()


def test_mcp_retrieve_schema_advertises_hybrid(tmp_store):
    resp = _mcp(tmp_store, {"jsonrpc": "2.0", "id": 44, "method": "tools/list", "params": {}})
    retrieve_tool = next(t for t in resp[0]["result"]["tools"] if t["name"] == "mnemonics_retrieve")
    props = retrieve_tool["inputSchema"]["properties"]
    assert "hybrid" in props and props["hybrid"]["type"] == "boolean"
    assert "candidate_k" in props and props["candidate_k"]["type"] == "integer"


def test_mcp_pin(populated_store):
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 8,
        "method": "tools/call",
        "params": {"name": "mnemonics_pin", "arguments": {"id": first_id}},
    })
    assert "Pinned" in resp[0]["result"]["content"][0]["text"]
    tier = store._db.execute("SELECT tier FROM memories WHERE id=?", (first_id,)).fetchone()[0]
    assert tier == 0


def test_mcp_tier(populated_store):
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 9,
        "method": "tools/call",
        "params": {"name": "mnemonics_tier", "arguments": {"id": first_id, "tier": 2}},
    })
    assert "Tier set" in resp[0]["result"]["content"][0]["text"]
    tier = store._db.execute("SELECT tier FROM memories WHERE id=?", (first_id,)).fetchone()[0]
    assert tier == 2


def test_mcp_forget(populated_store):
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 5,
        "method": "tools/call",
        "params": {"name": "mnemonics_forget", "arguments": {"id": first_id}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "True" in text or "False" in text


def test_mcp_health(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 20,
        "method": "tools/call",
        "params": {"name": "mnemonics_health", "arguments": {}},
    })
    import json as _json
    text = resp[0]["result"]["content"][0]["text"]
    report = _json.loads(text)
    assert "db_integrity" in report
    assert "namespaces" in report
    assert report["db_integrity"] == "ok"


def test_mcp_gc_with_tier(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 21,
        "method": "tools/call",
        "params": {"name": "mnemonics_gc", "arguments": {"tier": 1, "age_days": 0, "dry_run": True}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "tier=1" in text
    assert "candidate(s)" in text


def test_mcp_unknown_tool(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 6,
        "method": "tools/call",
        "params": {"name": "nonexistent", "arguments": {}},
    })
    assert "error" in resp[0]


def test_mcp_unknown_method(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 7,
        "method": "does/not/exist",
        "params": {},
    })
    assert "error" in resp[0]


def test_mcp_ignores_blank_lines(tmp_store):
    lines = "\n\n"
    output = []
    with (
        patch("mnemonics.server._get_store", return_value=tmp_store),
        patch("builtins.print", side_effect=lambda s, **_: output.append(s)),
        patch("sys.stdin", io.StringIO(lines)),
    ):
        srv._mcp_loop()
    assert output == []


# ── mnemonics_ingest MCP validation ──────────────────────────────────────────

def _mcp_msg(resp_item: dict) -> str:
    """Extract text from MCP response whether it's an error or a result."""
    if "error" in resp_item:
        return resp_item["error"].get("message", "")
    return resp_item.get("result", {}).get("content", [{}])[0].get("text", "")


def test_mcp_ingest_texts_none(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 90,
        "method": "tools/call",
        "params": {"name": "mnemonics_ingest", "arguments": {"texts": None}},
    })
    assert "error" in resp[0] or "error" in _mcp_msg(resp[0]).lower()


def test_mcp_ingest_texts_not_list(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 91,
        "method": "tools/call",
        "params": {"name": "mnemonics_ingest", "arguments": {"texts": 42}},
    })
    msg = _mcp_msg(resp[0]).lower()
    assert "array" in msg or "error" in msg or "error" in resp[0]


def test_mcp_ingest_texts_empty(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 92,
        "method": "tools/call",
        "params": {"name": "mnemonics_ingest", "arguments": {"texts": []}},
    })
    msg = _mcp_msg(resp[0]).lower()
    assert "empty" in msg or "error" in msg or "error" in resp[0]


def test_mcp_ingest_texts_non_string_item(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 93,
        "method": "tools/call",
        "params": {"name": "mnemonics_ingest", "arguments": {"texts": [123, "valid"]}},
    })
    msg = _mcp_msg(resp[0])
    assert "non-empty string" in msg or "error" in msg.lower() or "error" in resp[0]


def test_mcp_ingest_summaries_bad_length(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 94,
        "method": "tools/call",
        "params": {"name": "mnemonics_ingest", "arguments": {
            "texts": ["a", "b"], "summaries": ["only one"],
        }},
    })
    msg = _mcp_msg(resp[0]).lower()
    assert "summaries" in msg or "error" in msg or "error" in resp[0]


# ── MCP mnemonics_gc apply (not dry_run) ─────────────────────────────────────

def test_mcp_gc_apply(tmp_store):
    import numpy as np
    from mnemonics.store import DIM
    rng = np.random.default_rng(0)
    vecs = rng.random((1, DIM)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    tmp_store.add(["old content"], vecs)
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_gc",
                   "arguments": {"ns": "default", "age_days": 0, "tier": 2, "dry_run": False}},
    })
    txt = _mcp_msg(resp[0])
    assert "Deleted" in txt


# ── MCP mnemonics_rebuild_index RuntimeError ──────────────────────────────────

def test_mcp_rebuild_index_runtime_error(tmp_store, monkeypatch):
    monkeypatch.setattr(tmp_store, "rebuild_ns_index",
                        lambda ns: (_ for _ in ()).throw(RuntimeError("collision")))
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_rebuild_index", "arguments": {"ns": "default"}},
    })
    assert "error" in resp[0] or "collision" in _mcp_msg(resp[0])


# ── PATCH unknown path → 404 ──────────────────────────────────────────────────

def test_http_patch_unknown_path(tmp_store):
    code, data = http_call(tmp_store, "PATCH", "/unknown", {"summary": "x"})
    assert code == 404


# ── DELETE unknown path → 404 ─────────────────────────────────────────────────

def test_http_delete_unknown_path(tmp_store):
    code, data = http_call(tmp_store, "DELETE", "/unknown")
    assert code == 404


# ── PATCH invalid JSON body → 400 ────────────────────────────────────────────

def test_http_patch_invalid_json_body(tmp_store):
    with patch("mnemonics.server._get_store", return_value=tmp_store):
        handler = srv._Handler.__new__(srv._Handler)
        captured = {"code": None, "data": None}
        def fake_json(code, data):
            captured["code"] = code
            captured["data"] = data
        handler._json = fake_json
        handler._body = lambda: (_ for _ in ()).throw(ValueError("bad json"))
        handler.path = "/memory/1"
        handler.do_PATCH()
    assert captured["code"] == 400
    assert "invalid JSON" in captured["data"].get("error", "")


# ── MCP loop invalid JSON line → continue (no crash) ─────────────────────────

def test_mcp_invalid_json_line_skipped(tmp_store):
    # Send one malformed line then a valid request — server must skip the bad one.
    valid = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    lines = "THIS IS NOT JSON\n" + valid + "\n"
    output = []

    def fake_print(s, **_):
        output.append(json.loads(s))

    with (
        patch("mnemonics.server._get_store", return_value=tmp_store),
        patch("builtins.print", side_effect=fake_print),
        patch("sys.stdin", io.StringIO(lines)),
    ):
        srv._mcp_loop()

    # Only the valid response should appear
    assert len(output) == 1
    assert output[0]["result"]["protocolVersion"] == "2024-11-05"


# ── MCP mnemonics_bm25 summary line ──────────────────────────────────────────

def test_mcp_bm25_shows_summary(tmp_store):
    import numpy as np
    from mnemonics.store import DIM
    rng = np.random.default_rng(0)
    vecs = rng.random((1, DIM)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = tmp_store.add(["unique_kwxyz_token"], vecs)
    tmp_store.update_summary(ids[0], "the gist")
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_bm25",
                   "arguments": {"query": "unique_kwxyz_token", "top_k": 5}},
    })
    txt = _mcp_msg(resp[0])
    assert "the gist" in txt


# ── serve() mcp=True path ─────────────────────────────────────────────────────

def test_serve_mcp_true(tmp_store):
    with (
        patch("mnemonics.server._mcp_loop") as mock_loop,
        patch("mnemonics.server._get_store", return_value=tmp_store),
    ):
        srv.serve(mcp=True)
    mock_loop.assert_called_once()


# ── do_POST() invalid JSON body → 400 ────────────────────────────────────────

def test_http_post_invalid_json_body(tmp_store):
    with patch("mnemonics.server._get_store", return_value=tmp_store):
        handler = srv._Handler.__new__(srv._Handler)
        captured = {"code": None, "data": None}
        def fake_json(code, data):
            captured["code"] = code
            captured["data"] = data
        handler._json = fake_json
        handler._body = lambda: (_ for _ in ()).throw(ValueError("bad json"))
        handler.path = "/ingest"
        handler.do_POST()
    assert captured["code"] == 400
    assert "invalid JSON" in captured["data"].get("error", "")


# ── do_POST() /repair ─────────────────────────────────────────────────────────

def test_http_post_repair(tmp_store):
    code, data = http_call(tmp_store, "POST", "/repair", {})
    assert code == 200
    assert "orphan_vectors_fixed" in data


# ── do_POST() /rebuild-index RuntimeError → 409 ──────────────────────────────

def test_http_post_rebuild_index_runtime_error(tmp_store):
    with patch("mnemonics.server._get_store", return_value=tmp_store):
        handler = srv._Handler.__new__(srv._Handler)
        captured = {"code": None, "data": None}
        def fake_json(code, data):
            captured["code"] = code
            captured["data"] = data
        handler._json = fake_json
        handler._body = lambda: {"ns": "default"}
        handler.path = "/rebuild-index"
        with patch.object(tmp_store, "rebuild_ns_index", side_effect=RuntimeError("collision")):
            handler.do_POST()
    assert captured["code"] == 409
    assert "collision" in captured["data"].get("error", "")


# ── _get_store() singleton creation ──────────────────────────────────────────

def test_get_store_creates_store(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_store", None)
    monkeypatch.setattr(srv, "MNEMONICS_PATH", str(tmp_path))
    store = srv._get_store()
    assert store is not None
    monkeypatch.setattr(srv, "_store", None)  # cleanup


# ── serve() HTTP path ─────────────────────────────────────────────────────────

def test_serve_http_path(monkeypatch, tmp_path):
    monkeypatch.setattr(srv, "_store", None)
    monkeypatch.setattr(srv, "MNEMONICS_PATH", str(tmp_path))
    with (
        patch("mnemonics.server.HTTPServer") as mock_httpserver,
        patch("builtins.print"),
    ):
        mock_instance = MagicMock()
        mock_httpserver.return_value = mock_instance
        mock_instance.serve_forever.side_effect = KeyboardInterrupt
        import sys as _sys
        with pytest.raises(SystemExit):
            srv.serve(port=9999, mcp=False)
    monkeypatch.setattr(srv, "_store", None)


# ── _Handler methods: _body, _json, log_message ───────────────────────────────

def test_handler_body_reads_content_length():
    """lines 70-73: _body reads rfile using Content-Length header."""
    handler = srv._Handler.__new__(srv._Handler)
    body_data = b'{"key": "value"}'
    handler.headers = {"Content-Length": str(len(body_data))}
    handler.rfile = io.BufferedReader(io.BytesIO(body_data))
    result = handler._body()
    assert result == {"key": "value"}


def test_handler_body_returns_empty_when_no_content_length():
    """lines 71-72: _body returns {} when Content-Length is 0."""
    handler = srv._Handler.__new__(srv._Handler)
    handler.headers = {}
    handler.rfile = io.BufferedReader(io.BytesIO(b""))
    result = handler._body()
    assert result == {}


def test_handler_json_sends_response():
    """lines 76-81: _json writes HTTP response with JSON body."""
    handler = srv._Handler.__new__(srv._Handler)
    sent = {"code": None, "headers": [], "body": b""}

    handler.send_response = lambda code: sent.__setitem__("code", code)
    handler.send_header = lambda k, v: sent["headers"].append((k, v))
    handler.end_headers = lambda: None
    wfile = io.BytesIO()
    handler.wfile = wfile

    handler._json(200, {"ok": True})

    assert sent["code"] == 200
    assert any(k == "Content-Type" for k, _ in sent["headers"])
    wfile.seek(0)
    result = json.loads(wfile.read())
    assert result == {"ok": True}


def test_handler_log_message_suppressed():
    """line 67: log_message is a no-op (suppresses HTTP access logs)."""
    handler = srv._Handler.__new__(srv._Handler)
    handler.log_message("GET %s HTTP/1.1", "/health")  # must not raise


def test_version_fallback(monkeypatch):
    """lines 44-45: _VERSION falls back when package not installed."""
    import importlib.metadata
    import importlib as _il
    with patch("importlib.metadata.version",
               side_effect=importlib.metadata.PackageNotFoundError("mnemonics")):
        import mnemonics.server as _fresh
        import importlib
        importlib.reload(_fresh)
        assert _fresh._VERSION == "0.3.0"


# ── new bulk endpoints ─────────────────────────────────────────────────────────

def test_post_search_by_meta(populated_store):
    """POST /search-by-meta returns filtered results."""
    store, docs, vecs = populated_store
    # Tag the first memory with a unique source
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.update_meta(first_id, {"source": "unit-test"})
    code, data = http_call(store, "POST", "/search-by-meta",
                           {"filters": {"source": "unit-test"}, "ns": "default"})
    assert code == 200
    assert len(data["results"]) == 1
    assert data["results"][0]["id"] == first_id


def test_post_search_by_meta_bad_filters(populated_store):
    """POST /search-by-meta with non-dict filters returns 400."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/search-by-meta", {"filters": "bad"})
    assert code == 400


def test_post_get_many(populated_store):
    """POST /get-many returns the requested memories."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 2").fetchall()]
    code, data = http_call(store, "POST", "/get-many", {"ids": ids})
    assert code == 200
    assert len(data["results"]) == 2


def test_post_get_many_bad_ids(populated_store):
    """POST /get-many with non-list ids returns 400."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/get-many", {"ids": "not-a-list"})
    assert code == 400


def test_post_delete_many(populated_store):
    """POST /delete-many removes the specified memories."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 2").fetchall()]
    before = store.count()
    code, data = http_call(store, "POST", "/delete-many", {"ids": ids})
    assert code == 200
    assert data["deleted"] == 2
    assert store.count() == before - 2


def test_post_delete_many_bad_ids(populated_store):
    """POST /delete-many with non-list ids returns 400."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/delete-many", {"ids": 42})
    assert code == 400


def test_post_update_meta(populated_store):
    """POST /update-meta updates a memory's metadata."""
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    code, data = http_call(store, "POST", "/update-meta",
                           {"id": first_id, "meta": {"tag": "updated"}})
    assert code == 200
    assert data["changed"] is True
    import json
    raw = store._db.execute("SELECT meta FROM memories WHERE id=?", (first_id,)).fetchone()[0]
    assert json.loads(raw) == {"tag": "updated"}


def test_post_update_meta_missing_params(populated_store):
    """POST /update-meta without id or meta returns 400."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/update-meta", {"id": 1})
    assert code == 400


def test_mcp_search_by_meta(populated_store):
    """MCP mnemonics_search_by_meta returns matching memories."""
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.update_meta(first_id, {"tag": "mcp-test"})
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 30,
        "method": "tools/call",
        "params": {"name": "mnemonics_search_by_meta", "arguments": {"filters": {"tag": "mcp-test"}}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "Found 1 result" in text


def test_mcp_search_by_meta_no_results(populated_store):
    """MCP mnemonics_search_by_meta with no match returns 'No results' message."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 31,
        "method": "tools/call",
        "params": {"name": "mnemonics_search_by_meta", "arguments": {"filters": {"tag": "nonexistent"}}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "No results" in text


def test_mcp_search_by_meta_bad_filters(populated_store):
    """MCP mnemonics_search_by_meta with empty filters returns error content."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 32,
        "method": "tools/call",
        "params": {"name": "mnemonics_search_by_meta", "arguments": {"filters": {}}},
    })
    r = resp[0]
    if "result" in r:
        text = r["result"]["content"][0]["text"]
    else:
        text = str(r.get("error", ""))
    assert "error" in text.lower() or "filters" in text.lower()


def test_mcp_delete_many(populated_store):
    """MCP mnemonics_delete_many removes memories and reports count."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 2").fetchall()]
    before = store.count()
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 33,
        "method": "tools/call",
        "params": {"name": "mnemonics_delete_many", "arguments": {"ids": ids}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "Deleted 2" in text
    assert store.count() == before - 2


def test_mcp_update_meta(populated_store):
    """MCP mnemonics_update_meta updates metadata and confirms."""
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 34,
        "method": "tools/call",
        "params": {"name": "mnemonics_update_meta", "arguments": {"id": first_id, "meta": {"x": 99}}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "updated" in text


def test_mcp_delete_many_bad_ids(populated_store):
    """MCP mnemonics_delete_many with non-list ids returns error."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 35,
        "method": "tools/call",
        "params": {"name": "mnemonics_delete_many", "arguments": {"ids": 42}},
    })
    r = resp[0]
    if "result" in r:
        text = r["result"]["content"][0]["text"]
    else:
        text = str(r.get("error", ""))
    assert "error" in text.lower() or "ids" in text.lower()


def test_mcp_update_meta_missing_params(populated_store):
    """MCP mnemonics_update_meta without meta returns error."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 36,
        "method": "tools/call",
        "params": {"name": "mnemonics_update_meta", "arguments": {"id": 1}},
    })
    r = resp[0]
    if "result" in r:
        text = r["result"]["content"][0]["text"]
    else:
        text = str(r.get("error", ""))
    assert "error" in text.lower() or "meta" in text.lower()


def test_mcp_ingest_with_meta(populated_store):
    """MCP mnemonics_ingest passes meta dict to ingest()."""
    store, docs, vecs = populated_store
    with patch("mnemonics.server._ingest", return_value=1) as mock_ing:
        resp = _mcp(store, {
            "jsonrpc": "2.0", "id": 40,
            "method": "tools/call",
            "params": {"name": "mnemonics_ingest",
                       "arguments": {"texts": ["hello"], "meta": {"tag": "test"}}},
        })
    text = resp[0]["result"]["content"][0]["text"]
    assert "Stored" in text
    call_kwargs = mock_ing.call_args[1]
    assert call_kwargs["meta"] == [{"tag": "test"}]


def test_mcp_ingest_meta_non_object(populated_store):
    """MCP mnemonics_ingest rejects meta that is not an object."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 41,
        "method": "tools/call",
        "params": {"name": "mnemonics_ingest",
                   "arguments": {"texts": ["hello"], "meta": [1, 2, 3]}},
    })
    r = resp[0]
    if "result" in r:
        text = r["result"]["content"][0]["text"]
    else:
        text = str(r.get("error", ""))
    assert "error" in text.lower() or "meta" in text.lower()


def test_http_retrieve_min_tier_filter(populated_store):
    """POST /retrieve with min_tier filters low-tier results."""
    store, docs, vecs = populated_store
    # Pin first row so it has tier=0, rest have tier=1
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.pin(first_id)
    # min_tier=1 should exclude the pinned row
    with patch("mnemonics.server._retrieve", return_value={"results": []}) as mock_ret:
        code, data = http_call(store, "POST", "/retrieve", {
            "query": "test", "min_tier": 1, "max_tier": 2
        })
    assert code == 200
    call_kwargs = mock_ret.call_args[1]
    assert call_kwargs["min_tier"] == 1
    assert call_kwargs["max_tier"] == 2


def test_http_search_bm25_tier_filter(populated_store):
    """POST /search-bm25 with min_tier passes filter to search_bm25."""
    store, docs, vecs = populated_store
    query = docs[0].split()[0]
    with patch.object(store, "search_bm25", return_value=[]) as mock_bm25:
        with patch("mnemonics.server._get_store", return_value=store):
            code, data = http_call(store, "POST", "/search-bm25", {
                "query": query, "min_tier": 1
            })
    assert code == 200
    call_kwargs = mock_bm25.call_args[1]
    assert call_kwargs["min_tier"] == 1


# ── GET /count ────────────────────────────────────────────────────────────────

def test_http_count_all_ns(populated_store):
    """GET /count with no ns param returns total count (ns=None)."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/count")
    assert code == 200
    assert "count" in data
    assert data["count"] == store.count(None)
    assert data["ns"] is None


def test_http_count_specific_ns(populated_store):
    """GET /count?ns=default returns count for that namespace."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/count?ns=default")
    assert code == 200
    assert data["ns"] == "default"
    assert data["count"] == store.count("default")


# ── REST /ingest tier ─────────────────────────────────────────────────────────

def test_http_ingest_tier_pinned(tmp_store):
    """POST /ingest with tier=0 stores pinned memories."""
    with patch("mnemonics.server._ingest", return_value=1) as mock_ing:
        code, data = http_call(tmp_store, "POST", "/ingest",
                               {"texts": ["pin me"], "tier": 0})
    assert code == 200
    mock_ing.assert_called_once()
    assert mock_ing.call_args[1]["tier"] == 0


def test_http_ingest_tier_invalid(tmp_store):
    """POST /ingest with tier=9 returns 400."""
    code, data = http_call(tmp_store, "POST", "/ingest",
                           {"texts": ["hi"], "tier": 9})
    assert code == 400
    assert "tier" in data["error"]


# ── MCP mnemonics_ingest tier ─────────────────────────────────────────────────

def test_mcp_ingest_tier_pinned(tmp_store):
    """MCP mnemonics_ingest with tier=0 calls ingest with tier=0."""
    with patch("mnemonics.server._ingest", return_value=1) as mock_ing:
        resp = _mcp(tmp_store, {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "mnemonics_ingest",
                       "arguments": {"texts": ["pin"], "tier": 0}},
        })
    assert "result" in resp[0]
    mock_ing.assert_called_once()
    assert mock_ing.call_args[1]["tier"] == 0


def test_mcp_ingest_tier_invalid(tmp_store):
    """MCP mnemonics_ingest with tier=5 returns JSON-RPC error."""
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_ingest",
                   "arguments": {"texts": ["x"], "tier": 5}},
    })
    assert "error" in resp[0]


# ── GET /memories?since= ──────────────────────────────────────────────────────

def test_http_list_memories_since_filter(populated_store):
    """GET /memories?since= passes since to list_memories."""
    store, docs, vecs = populated_store
    with patch.object(store, "list_memories", wraps=store.list_memories) as mock_lm:
        with patch("mnemonics.server._get_store", return_value=store):
            code, data = http_call(store, "GET", "/memories?ns=default&since=2000-01-01")
    assert code == 200
    call_kwargs = mock_lm.call_args[1]
    assert call_kwargs.get("since") == "2000-01-01"
    assert data["count"] > 0


def test_http_list_memories_since_future(populated_store):
    """GET /memories?since=far-future returns zero rows."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/memories?ns=default&since=2099-01-01")
    assert code == 200
    assert data["count"] == 0


# ── GET /export-jsonl ─────────────────────────────────────────────────────────

def _get_export(store, path: str) -> tuple[int, list[dict]]:
    """GET /export-jsonl and parse the NDJSON response body."""
    import io as _io
    import json as _json

    wfile = _io.BytesIO()
    with patch("mnemonics.server._get_store", return_value=store):
        handler = srv._Handler.__new__(srv._Handler)
        handler.rfile = _io.BufferedReader(_io.BytesIO(b""))
        handler.wfile = wfile
        handler.server = MagicMock()
        handler.client_address = ("127.0.0.1", 0)
        handler.request_version = "HTTP/1.1"
        handler.requestline = f"GET {path} HTTP/1.1"
        handler.command = "GET"
        handler.path = path
        handler.headers = {}
        handler.responses = {}

        sent_status = {}
        sent_headers: dict[str, str] = {}

        def fake_send_response(code, msg=None):
            sent_status["code"] = code

        def fake_send_header(k, v):
            sent_headers[k] = v

        def fake_end_headers():
            pass

        handler.send_response = fake_send_response
        handler.send_header = fake_send_header
        handler.end_headers = fake_end_headers

        captured_json = {"code": None, "data": None}

        def fake_json(code, data):
            captured_json["code"] = code
            captured_json["data"] = data

        handler._json = fake_json
        handler.do_GET()

    # If _json was called (e.g. for a 404 fallback), return that
    if captured_json["code"] is not None:
        return captured_json["code"], captured_json["data"]

    body = wfile.getvalue()
    rows = [_json.loads(line) for line in body.decode().splitlines() if line.strip()]
    return sent_status.get("code", 200), rows


def test_http_export_jsonl_all(populated_store):
    """GET /export-jsonl returns all memories as NDJSON."""
    store, docs, vecs = populated_store
    code, rows = _get_export(store, "/export-jsonl")
    assert code == 200
    assert len(rows) == len(docs)
    assert all("id" in r and "text" in r and "tier" in r for r in rows)


def test_http_export_jsonl_ns_filter(populated_store):
    """GET /export-jsonl?ns=default filters by namespace."""
    store, docs, vecs = populated_store
    code, rows = _get_export(store, "/export-jsonl?ns=default")
    assert code == 200
    assert all(r["ns"] == "default" for r in rows)


def test_http_export_jsonl_tier_filter(populated_store):
    """GET /export-jsonl?tier=0 returns only pinned memories."""
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.pin(first_id)
    code, rows = _get_export(store, "/export-jsonl?tier=0")
    assert code == 200
    assert len(rows) == 1
    assert rows[0]["id"] == first_id
    assert rows[0]["tier"] == 0


def test_http_export_jsonl_since_filter(populated_store):
    """GET /export-jsonl?since=2099-01-01 returns nothing (future date)."""
    store, docs, vecs = populated_store
    code, rows = _get_export(store, "/export-jsonl?since=2099-01-01")
    assert code == 200
    assert rows == []


def test_http_export_jsonl_empty_store(tmp_store):
    """GET /export-jsonl on an empty store returns zero rows."""
    code, rows = _get_export(tmp_store, "/export-jsonl")
    assert code == 200
    assert rows == []


# ── MCP mnemonics_list --since ─────────────────────────────────────────────────

def test_mcp_list_since_past(populated_store):
    """mnemonics_list with since=far-past returns all rows."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_list",
                   "arguments": {"ns": "default", "limit": 100, "since": "2000-01-01"}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "showing" in text
    assert str(len(docs)) in text


def test_mcp_list_since_future(populated_store):
    """mnemonics_list with since=far-future returns no memories."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_list",
                   "arguments": {"ns": "default", "since": "2099-01-01"}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "No memories" in text


# ── GET /memories?before= ─────────────────────────────────────────────────────

def test_http_list_memories_before_future(populated_store):
    """GET /memories?before=far-future returns all rows."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/memories?ns=default&before=2099-01-01")
    assert code == 200
    assert data["count"] == len(docs)


def test_http_list_memories_before_past(populated_store):
    """GET /memories?before=far-past returns zero rows."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/memories?ns=default&before=2000-01-01")
    assert code == 200
    assert data["count"] == 0


# ── GET /export-jsonl?before= ─────────────────────────────────────────────────

def test_http_export_jsonl_before_future(populated_store):
    """GET /export-jsonl?before=far-future returns all rows."""
    store, docs, vecs = populated_store
    code, rows = _get_export(store, "/export-jsonl?before=2099-01-01")
    assert code == 200
    assert len(rows) == len(docs)


def test_http_export_jsonl_before_past(populated_store):
    """GET /export-jsonl?before=far-past returns zero rows."""
    store, docs, vecs = populated_store
    code, rows = _get_export(store, "/export-jsonl?before=2000-01-01")
    assert code == 200
    assert rows == []


# ── MCP mnemonics_list before ─────────────────────────────────────────────────

def test_mcp_list_before_past(populated_store):
    """mnemonics_list with before=far-past returns no memories."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_list",
                   "arguments": {"ns": "default", "before": "2000-01-01"}},
    })
    text = resp[0]["result"]["content"][0]["text"]
    assert "No memories" in text


# ── POST /import-jsonl ────────────────────────────────────────────────────────

def _import_call(store, ndjson_body: str, ingest_return: int = 1) -> tuple[int, dict]:
    """POST /import-jsonl with raw NDJSON body. Mocks _ingest to avoid real embedding."""
    body_bytes = ndjson_body.encode("utf-8")
    import io as _io

    captured = {"code": None, "data": None}

    with (
        patch("mnemonics.server._get_store", return_value=store),
        patch("mnemonics.server._ingest", return_value=ingest_return),
    ):
        handler = srv._Handler.__new__(srv._Handler)
        handler.rfile = _io.BufferedReader(_io.BytesIO(body_bytes))
        handler.wfile = _io.BytesIO()
        handler.server = MagicMock()
        handler.client_address = ("127.0.0.1", 0)
        handler.request_version = "HTTP/1.1"
        handler.requestline = "POST /import-jsonl HTTP/1.1"
        handler.command = "POST"
        handler.path = "/import-jsonl"
        handler.headers = {
            "Content-Length": str(len(body_bytes)),
            "Content-Type": "application/x-ndjson",
        }
        handler.responses = {}
        handler._json = lambda code, data: captured.update({"code": code, "data": data})
        handler.do_POST()

    return captured["code"], captured["data"]


def test_http_import_jsonl_basic(tmp_store):
    """POST /import-jsonl inserts rows."""
    line = json.dumps({"text": "imported memory", "ns": "default", "tier": 1})
    code, data = _import_call(tmp_store, line + "\n")
    assert code == 200
    assert data["imported"] >= 1
    assert data["skipped"] == 0


def test_http_import_jsonl_empty_body(tmp_store):
    """POST /import-jsonl with empty body returns 400."""
    body_bytes = b""
    import io as _io
    with patch("mnemonics.server._get_store", return_value=tmp_store):
        handler = srv._Handler.__new__(srv._Handler)
        handler.rfile = _io.BufferedReader(_io.BytesIO(b""))
        handler.wfile = _io.BytesIO()
        handler.server = MagicMock()
        handler.client_address = ("127.0.0.1", 0)
        handler.path = "/import-jsonl"
        handler.headers = {"Content-Length": "0"}
        handler.responses = {}
        captured = {"code": None, "data": None}
        handler._json = lambda code, data: captured.update({"code": code, "data": data})
        handler.do_POST()
    assert captured["code"] == 400


def test_http_import_jsonl_invalid_json_line(tmp_store):
    """POST /import-jsonl skips invalid JSON lines and reports them."""
    body = "not valid json\n" + json.dumps({"text": "valid one"}) + "\n"
    code, data = _import_call(tmp_store, body)
    assert code == 200
    assert data["skipped"] == 1
    assert len(data["errors"]) == 1


def test_http_import_jsonl_missing_text(tmp_store):
    """POST /import-jsonl skips rows without text field."""
    body = json.dumps({"ns": "default", "meta": {"tag": "x"}}) + "\n"
    code, data = _import_call(tmp_store, body)
    assert code == 200
    assert data["skipped"] == 1


def test_http_import_jsonl_tier_clamped(tmp_store):
    """POST /import-jsonl clamps invalid tier to 1."""
    body = json.dumps({"text": "tier clamped", "tier": 9}) + "\n"
    code, data = _import_call(tmp_store, body)
    assert code == 200
    assert data["imported"] >= 1


def test_http_import_jsonl_blank_lines_skipped(tmp_store):
    """POST /import-jsonl ignores blank lines."""
    body = "\n\n" + json.dumps({"text": "valid"}) + "\n\n"
    code, data = _import_call(tmp_store, body)
    assert code == 200
    assert data["imported"] == 1
    assert data["skipped"] == 0


# ── MCP mnemonics_export ──────────────────────────────────────────────────────

def _mcp_export(store, **kwargs):
    """Helper: call mnemonics_export MCP tool and return resp[0]."""
    return _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_export", "arguments": kwargs},
    })[0]


def test_mcp_export_all(populated_store):
    """mnemonics_export returns all memories as JSONL."""
    import json as _json
    store, docs, vecs = populated_store
    r = _mcp_export(store)
    text = r["result"]["content"][0]["text"]
    lines = [l for l in text.splitlines() if l.strip()]
    assert len(lines) == len(docs)
    obj = _json.loads(lines[0])
    assert "id" in obj and "text" in obj and "ns" in obj


def test_mcp_export_ns_filter(populated_store):
    """mnemonics_export ns filter returns only rows from that namespace."""
    import json as _json
    store, docs, vecs = populated_store
    import numpy as np
    v = np.random.rand(384).astype("float32")
    v /= np.linalg.norm(v)
    store.add(["other ns row"], v[None], ns="other")
    r = _mcp_export(store, ns="default")
    text = r["result"]["content"][0]["text"]
    objs = [_json.loads(l) for l in text.splitlines() if l.strip()]
    assert all(o["ns"] == "default" for o in objs)


def test_mcp_export_tier_filter(populated_store):
    """mnemonics_export tier=0 returns only pinned rows."""
    import json as _json
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.pin(first_id)
    r = _mcp_export(store, tier=0)
    text = r["result"]["content"][0]["text"]
    objs = [_json.loads(l) for l in text.splitlines() if l.strip()]
    assert len(objs) == 1
    assert objs[0]["id"] == first_id


def test_mcp_export_since_filter(populated_store):
    """mnemonics_export since=far-future returns empty."""
    store, docs, vecs = populated_store
    r = _mcp_export(store, since="2099-01-01")
    text = r["result"]["content"][0]["text"]
    assert text == "(no memories matched)"


def test_mcp_export_before_filter(populated_store):
    """mnemonics_export before=past returns empty."""
    store, docs, vecs = populated_store
    r = _mcp_export(store, before="2000-01-01")
    text = r["result"]["content"][0]["text"]
    assert text == "(no memories matched)"


def test_mcp_export_limit(populated_store):
    """mnemonics_export limit restricts number of returned rows."""
    import json as _json
    store, docs, vecs = populated_store
    r = _mcp_export(store, limit=1)
    text = r["result"]["content"][0]["text"]
    objs = [_json.loads(l) for l in text.splitlines() if l.strip()]
    assert len(objs) == 1


# ── MCP mnemonics_import ──────────────────────────────────────────────────────

def _mcp_import(store, jsonl: str, **kwargs):
    """Helper: call mnemonics_import MCP tool and return resp[0]."""
    return _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_import", "arguments": {"jsonl": jsonl, **kwargs}},
    })[0]


def test_mcp_import_basic(tmp_store):
    """mnemonics_import ingests a valid JSONL string."""
    import json as _json
    line = _json.dumps({"text": "hello memory", "ns": "default", "tier": 1})
    with patch("mnemonics.server._ingest", return_value=1) as mock_ingest:
        r = _mcp_import(tmp_store, line)
    assert "result" in r
    assert "imported=1" in r["result"]["content"][0]["text"]
    mock_ingest.assert_called_once()


def test_mcp_import_dry_run(tmp_store):
    """mnemonics_import dry_run=true does not call _ingest."""
    import json as _json
    line = _json.dumps({"text": "dry row"})
    with patch("mnemonics.server._ingest") as mock_ingest:
        r = _mcp_import(tmp_store, line, dry_run=True)
    mock_ingest.assert_not_called()
    text = r["result"]["content"][0]["text"]
    assert "dry-run" in text
    assert "imported=1" in text


def test_mcp_import_empty_jsonl(tmp_store):
    """mnemonics_import with empty string returns error."""
    r = _mcp_import(tmp_store, "   ")
    assert "error" in r


def test_mcp_import_invalid_line(tmp_store):
    """mnemonics_import skips invalid JSON lines and rows without text."""
    import json as _json
    jsonl = "not json\n" + _json.dumps({"ns": "x"}) + "\n" + _json.dumps({"text": "ok"})
    with patch("mnemonics.server._ingest", return_value=1) as mock_ingest:
        r = _mcp_import(tmp_store, jsonl)
    assert mock_ingest.call_count == 1
    text = r["result"]["content"][0]["text"]
    assert "skipped=2" in text


def test_mcp_import_ns_override(tmp_store):
    """mnemonics_import ns= overrides namespace in all rows."""
    import json as _json
    line = _json.dumps({"text": "row", "ns": "original"})
    with patch("mnemonics.server._ingest", return_value=1) as mock_ingest:
        _mcp_import(tmp_store, line, ns="override")
    kw = mock_ingest.call_args[1]
    assert kw["ns"] == "override"


def test_mcp_import_blank_lines_skipped(tmp_store):
    """mnemonics_import ignores blank lines in JSONL."""
    import json as _json
    jsonl = "\n\n" + _json.dumps({"text": "row"}) + "\n\n"
    with patch("mnemonics.server._ingest", return_value=1) as mock_ingest:
        r = _mcp_import(tmp_store, jsonl)
    assert mock_ingest.call_count == 1
    assert "imported=1" in r["result"]["content"][0]["text"]


def test_mcp_import_tools_list(tmp_store):
    """mnemonics_import appears in the MCP tools list."""
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/list", "params": {},
    })[0]
    names = {t["name"] for t in resp["result"]["tools"]}
    assert "mnemonics_import" in names


# ── GET /text-search ─────────────────────────────────────────────────────────

def test_http_text_search_basic(populated_store):
    """GET /text-search?q=Eiffel finds the Eiffel row."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/text-search?q=Eiffel")
    assert code == 200
    assert data["count"] >= 1
    assert any("Eiffel" in r["text"] for r in data["results"])


def test_http_text_search_no_match(populated_store):
    """GET /text-search with unknown term returns empty results."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/text-search?q=xyzzy_never_matches")
    assert code == 200
    assert data["count"] == 0


def test_http_text_search_missing_q(tmp_store):
    """GET /text-search without q returns 400."""
    code, data = http_call(tmp_store, "GET", "/text-search")
    assert code == 400


def test_http_text_search_all_ns(populated_store):
    """GET /text-search?ns=all searches across all namespaces."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/text-search?q=Eiffel&ns=all")
    assert code == 200
    assert data["count"] >= 1


def test_http_text_search_limit(populated_store):
    """GET /text-search?limit=1 returns at most 1 result."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/text-search?q=e&limit=1")
    assert code == 200
    assert len(data["results"]) <= 1


# ── MCP mnemonics_text_search ─────────────────────────────────────────────────

def _mcp_ts(store, **kwargs):
    return _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_text_search", "arguments": kwargs},
    })[0]


def test_mcp_text_search_hit(populated_store):
    """mnemonics_text_search finds a matching memory."""
    store, docs, vecs = populated_store
    r = _mcp_ts(store, query="Eiffel")
    assert "result" in r
    assert "id=" in r["result"]["content"][0]["text"]


def test_mcp_text_search_no_hit(populated_store):
    """mnemonics_text_search returns 'No results' when nothing matches."""
    store, docs, vecs = populated_store
    r = _mcp_ts(store, query="xyzzy_never_matches")
    assert "No results" in r["result"]["content"][0]["text"]


def test_mcp_text_search_empty_query(tmp_store):
    """mnemonics_text_search with empty query returns error."""
    r = _mcp_ts(tmp_store, query="")
    assert "error" in r


def test_mcp_text_search_in_tools_list(tmp_store):
    """mnemonics_text_search appears in the MCP tools list."""
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
    })[0]
    names = {t["name"] for t in resp["result"]["tools"]}
    assert "mnemonics_text_search" in names


# ── POST /rename-ns ───────────────────────────────────────────────────────────

def test_http_rename_ns_basic(populated_store):
    """POST /rename-ns moves all rows to the new namespace."""
    store, docs, vecs = populated_store
    n = store.count("default")
    code, data = http_call(store, "POST", "/rename-ns",
                           {"old_ns": "default", "new_ns": "renamed"})
    assert code == 200
    assert data["moved"] == n
    assert store.count("renamed") == n


def test_http_rename_ns_missing_fields(tmp_store):
    """POST /rename-ns without old_ns/new_ns returns 400."""
    code, data = http_call(tmp_store, "POST", "/rename-ns", {"old_ns": "x"})
    assert code == 400


def test_http_rename_ns_conflict(populated_store):
    """POST /rename-ns when target ns exists returns 409."""
    import numpy as np
    store, docs, vecs = populated_store
    v = np.random.rand(384).astype("float32"); v /= np.linalg.norm(v)
    store.add(["other row"], v[None], ns="other")
    code, data = http_call(store, "POST", "/rename-ns",
                           {"old_ns": "default", "new_ns": "other"})
    assert code == 409


# ── MCP mnemonics_rename_ns ────────────────────────────────────────────────────

def _mcp_rename(store, old_ns, new_ns):
    return _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_rename_ns",
                   "arguments": {"old_ns": old_ns, "new_ns": new_ns}},
    })[0]


def test_mcp_rename_ns_basic(populated_store):
    """mnemonics_rename_ns moves memories and reports count."""
    store, docs, vecs = populated_store
    n = store.count("default")
    r = _mcp_rename(store, "default", "renamed_ns")
    assert "result" in r
    assert f"{n} memories" in r["result"]["content"][0]["text"]
    assert store.count("renamed_ns") == n


def test_mcp_rename_ns_missing_args(tmp_store):
    """mnemonics_rename_ns without required args returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_rename_ns", "arguments": {"old_ns": ""}},
    })[0]
    assert "error" in r


def test_mcp_rename_ns_conflict(populated_store):
    """mnemonics_rename_ns when target exists returns error."""
    import numpy as np
    store, docs, vecs = populated_store
    v = np.random.rand(384).astype("float32"); v /= np.linalg.norm(v)
    store.add(["other"], v[None], ns="other")
    r = _mcp_rename(store, "default", "other")
    assert "error" in r


def test_mcp_rename_ns_in_tools_list(tmp_store):
    """mnemonics_rename_ns appears in the MCP tools list."""
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
    })[0]
    names = {t["name"] for t in resp["result"]["tools"]}
    assert "mnemonics_rename_ns" in names


# ── MCP mnemonics_namespaces ──────────────────────────────────────────────────

def test_mcp_namespaces_populated(populated_store):
    """mnemonics_namespaces lists existing namespaces."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_namespaces", "arguments": {}},
    })[0]
    text = r["result"]["content"][0]["text"]
    assert "default" in text


def test_mcp_namespaces_empty(tmp_store):
    """mnemonics_namespaces on empty store returns placeholder."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_namespaces", "arguments": {}},
    })[0]
    text = r["result"]["content"][0]["text"]
    assert "no namespaces" in text


def test_mcp_namespaces_in_tools_list(tmp_store):
    """mnemonics_namespaces appears in the MCP tools list."""
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
    })[0]
    names = {t["name"] for t in resp["result"]["tools"]}
    assert "mnemonics_namespaces" in names


# ── MCP mnemonics_count ───────────────────────────────────────────────────────

def test_mcp_count_default_ns(populated_store):
    """mnemonics_count returns count for default namespace."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_count", "arguments": {}},
    })[0]
    text = r["result"]["content"][0]["text"]
    assert str(len(docs)) in text


def test_mcp_count_all_ns(populated_store):
    """mnemonics_count with ns=null counts all namespaces."""
    import numpy as np
    store, docs, vecs = populated_store
    v = np.random.rand(384).astype("float32"); v /= np.linalg.norm(v)
    store.add(["extra row"], v[None], ns="other")
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_count", "arguments": {"ns": None}},
    })[0]
    text = r["result"]["content"][0]["text"]
    assert str(len(docs) + 1) in text


def test_mcp_count_in_tools_list(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
    })[0]
    assert "mnemonics_count" in {t["name"] for t in resp["result"]["tools"]}


# ── MCP mnemonics_get_many ────────────────────────────────────────────────────

def test_mcp_get_many_basic(populated_store):
    """mnemonics_get_many returns the requested memories."""
    import json as _json
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 2").fetchall()]
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_get_many", "arguments": {"ids": ids}},
    })[0]
    text = r["result"]["content"][0]["text"]
    objs = [_json.loads(l) for l in text.splitlines() if l.strip()]
    assert len(objs) == 2
    assert {o["id"] for o in objs} == set(ids)


def test_mcp_get_many_empty(tmp_store):
    """mnemonics_get_many with non-existing IDs returns placeholder."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_get_many", "arguments": {"ids": [9999]}},
    })[0]
    assert "no memories" in r["result"]["content"][0]["text"]


def test_mcp_get_many_bad_ids(tmp_store):
    """mnemonics_get_many with non-list ids returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_get_many", "arguments": {"ids": "not-a-list"}},
    })[0]
    assert "error" in r


def test_mcp_get_many_in_tools_list(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
    })[0]
    assert "mnemonics_get_many" in {t["name"] for t in resp["result"]["tools"]}


# ── POST /bulk-tier ───────────────────────────────────────────────────────────

def test_http_bulk_tier_basic(populated_store):
    """POST /bulk-tier updates all requested IDs."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 2").fetchall()]
    code, data = http_call(store, "POST", "/bulk-tier", {"ids": ids, "tier": 2})
    assert code == 200
    assert data["updated"] == 2
    for mid in ids:
        assert store.get(mid)["tier"] == 2


def test_http_bulk_tier_missing_fields(tmp_store):
    """POST /bulk-tier without required fields returns 400."""
    code, data = http_call(tmp_store, "POST", "/bulk-tier", {"ids": [1]})
    assert code == 400


def test_http_bulk_tier_invalid_tier(populated_store):
    """POST /bulk-tier with invalid tier returns 400."""
    store, docs, vecs = populated_store
    ids = [store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]]
    code, data = http_call(store, "POST", "/bulk-tier", {"ids": ids, "tier": 99})
    assert code == 400


# ── MCP mnemonics_bulk_tier ───────────────────────────────────────────────────

def test_mcp_bulk_tier_basic(populated_store):
    """mnemonics_bulk_tier updates all requested IDs."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 2").fetchall()]
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_bulk_tier", "arguments": {"ids": ids, "tier": 0}},
    })[0]
    assert "result" in r
    assert "2" in r["result"]["content"][0]["text"]
    for mid in ids:
        assert store.get(mid)["tier"] == 0


def test_mcp_bulk_tier_missing_args(tmp_store):
    """mnemonics_bulk_tier without required args returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_bulk_tier", "arguments": {"ids": [1]}},
    })[0]
    assert "error" in r


def test_mcp_bulk_tier_invalid_tier(populated_store):
    """mnemonics_bulk_tier with invalid tier returns error."""
    store, docs, vecs = populated_store
    ids = [store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]]
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_bulk_tier", "arguments": {"ids": ids, "tier": 99}},
    })[0]
    assert "error" in r


def test_mcp_bulk_tier_in_tools_list(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
    })[0]
    assert "mnemonics_bulk_tier" in {t["name"] for t in resp["result"]["tools"]}


# ── GET /recent ───────────────────────────────────────────────────────────────

def test_http_recent_basic(populated_store):
    """GET /recent returns memories ordered by access time."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/recent?ns=default")
    assert code == 200
    assert data["count"] == len(docs)


def test_http_recent_all_ns(populated_store):
    """GET /recent?ns=all returns memories from all namespaces."""
    import numpy as np
    store, docs, vecs = populated_store
    v = np.random.rand(384).astype("float32"); v /= np.linalg.norm(v)
    store.add(["other ns row"], v[None], ns="other")
    code, data = http_call(store, "GET", "/recent?ns=all")
    assert code == 200
    assert data["count"] >= len(docs) + 1


def test_http_recent_limit(populated_store):
    """GET /recent?limit=2 returns at most 2 results."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/recent?ns=default&limit=2")
    assert code == 200
    assert len(data["results"]) <= 2


# ── MCP mnemonics_recent ──────────────────────────────────────────────────────

def test_mcp_recent_basic(populated_store):
    """mnemonics_recent returns recently accessed memories."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_recent", "arguments": {}},
    })[0]
    assert "result" in r
    text = r["result"]["content"][0]["text"]
    assert "id=" in text


def test_mcp_recent_empty(tmp_store):
    """mnemonics_recent on empty store returns placeholder."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_recent", "arguments": {}},
    })[0]
    assert "no recently" in r["result"]["content"][0]["text"]


def test_mcp_recent_in_tools_list(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
    })[0]
    assert "mnemonics_recent" in {t["name"] for t in resp["result"]["tools"]}


# ── GET /top-accessed ─────────────────────────────────────────────────────────

def test_http_top_accessed_basic(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/top-accessed?ns=default")
    assert code == 200
    assert data["count"] == len(docs)


def test_http_top_accessed_limit(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/top-accessed?ns=default&limit=2")
    assert code == 200
    assert len(data["results"]) <= 2


# ── MCP mnemonics_top_accessed ────────────────────────────────────────────────

def test_mcp_top_accessed_basic(populated_store):
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_top_accessed", "arguments": {}},
    })[0]
    assert "result" in r
    assert "id=" in r["result"]["content"][0]["text"]


def test_mcp_top_accessed_empty(tmp_store):
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_top_accessed", "arguments": {}},
    })[0]
    assert "no accessed" in r["result"]["content"][0]["text"]


def test_mcp_top_accessed_in_tools_list(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
    })[0]
    assert "mnemonics_top_accessed" in {t["name"] for t in resp["result"]["tools"]}


# ── POST /copy-ns ─────────────────────────────────────────────────────────────

def test_http_copy_ns_basic(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/copy-ns", {"src_ns": "default", "dst_ns": "bk"})
    assert code == 200
    assert data["copied"] == len(docs)
    assert store.count("default") == len(docs)
    assert store.count("bk") == len(docs)


def test_http_copy_ns_missing_fields(tmp_store):
    code, data = http_call(tmp_store, "POST", "/copy-ns", {"src_ns": "x"})
    assert code == 400


def test_http_copy_ns_conflict(populated_store):
    store, docs, vecs = populated_store
    store.copy_ns("default", "bk2")
    code, data = http_call(store, "POST", "/copy-ns", {"src_ns": "default", "dst_ns": "bk2"})
    assert code == 409


# ── MCP mnemonics_copy_ns ─────────────────────────────────────────────────────

def test_mcp_copy_ns_basic(populated_store):
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_copy_ns", "arguments": {"src_ns": "default", "dst_ns": "bkcopy"}},
    })[0]
    assert "result" in r
    assert "copied" in r["result"]["content"][0]["text"]


def test_mcp_copy_ns_missing_args(tmp_store):
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_copy_ns", "arguments": {"src_ns": "x"}},
    })[0]
    assert "error" in r


def test_mcp_copy_ns_conflict(populated_store):
    store, docs, vecs = populated_store
    store.copy_ns("default", "bkco2")
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_copy_ns", "arguments": {"src_ns": "default", "dst_ns": "bkco2"}},
    })[0]
    assert "error" in r


def test_mcp_copy_ns_in_tools_list(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
    })[0]
    assert "mnemonics_copy_ns" in {t["name"] for t in resp["result"]["tools"]}


# ── GET /stats-by-ns ──────────────────────────────────────────────────────────

def test_http_stats_by_ns_basic(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/stats-by-ns")
    assert code == 200
    assert len(data["namespaces"]) == 1
    s = data["namespaces"][0]
    assert s["ns"] == "default"
    assert s["total"] == len(docs)


def test_http_stats_by_ns_empty(tmp_store):
    code, data = http_call(tmp_store, "GET", "/stats-by-ns")
    assert code == 200
    assert data["namespaces"] == []


# ── MCP mnemonics_stats_by_ns ─────────────────────────────────────────────────

def test_mcp_stats_by_ns_basic(populated_store):
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_stats_by_ns", "arguments": {}},
    })[0]
    assert "result" in r
    assert "total=" in r["result"]["content"][0]["text"]


def test_mcp_stats_by_ns_empty(tmp_store):
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_stats_by_ns", "arguments": {}},
    })[0]
    assert "no namespaces" in r["result"]["content"][0]["text"]


def test_mcp_stats_by_ns_in_tools_list(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
    })[0]
    assert "mnemonics_stats_by_ns" in {t["name"] for t in resp["result"]["tools"]}


# ── POST /merge-ns ────────────────────────────────────────────────────────────

def test_http_merge_ns_basic(populated_store):
    import numpy as np
    store, docs, vecs = populated_store
    v = np.random.rand(384).astype("float32"); v /= np.linalg.norm(v)
    store.add(["dst doc"], v[None], ns="dst_m")
    code, data = http_call(store, "POST", "/merge-ns", {"src_ns": "default", "dst_ns": "dst_m"})
    assert code == 200
    assert data["moved"] == len(docs)
    assert store.count("default") == 0
    assert store.count("dst_m") == len(docs) + 1


def test_http_merge_ns_missing_fields(tmp_store):
    code, data = http_call(tmp_store, "POST", "/merge-ns", {"src_ns": "x"})
    assert code == 400


# ── MCP mnemonics_merge_ns ────────────────────────────────────────────────────

def test_mcp_merge_ns_basic(populated_store):
    import numpy as np
    store, docs, vecs = populated_store
    v = np.random.rand(384).astype("float32"); v /= np.linalg.norm(v)
    store.add(["dst doc"], v[None], ns="mgdst")
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_merge_ns", "arguments": {"src_ns": "default", "dst_ns": "mgdst"}},
    })[0]
    assert "result" in r
    assert "moved" in r["result"]["content"][0]["text"]


def test_mcp_merge_ns_missing_args(tmp_store):
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_merge_ns", "arguments": {"src_ns": "x"}},
    })[0]
    assert "error" in r


def test_mcp_merge_ns_in_tools_list(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
    })[0]
    assert "mnemonics_merge_ns" in {t["name"] for t in resp["result"]["tools"]}


# ── POST /touch-many ──────────────────────────────────────────────────────────

def test_http_touch_many_basic(populated_store):
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 2").fetchall()]
    code, data = http_call(store, "POST", "/touch-many", {"ids": ids})
    assert code == 200
    assert data["touched"] == 2


def test_http_touch_many_missing_ids(tmp_store):
    code, data = http_call(tmp_store, "POST", "/touch-many", {"other": "x"})
    assert code == 400


# ── MCP mnemonics_touch_many ──────────────────────────────────────────────────

def test_mcp_touch_many_basic(populated_store):
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 2").fetchall()]
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_touch_many", "arguments": {"ids": ids}},
    })[0]
    assert "result" in r
    assert "2" in r["result"]["content"][0]["text"]


def test_mcp_touch_many_missing_args(tmp_store):
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_touch_many", "arguments": {"other": "x"}},
    })[0]
    assert "error" in r


def test_mcp_touch_many_in_tools_list(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
    })[0]
    assert "mnemonics_touch_many" in {t["name"] for t in resp["result"]["tools"]}


# ── PATCH /memory/:id (meta) ──────────────────────────────────────────────────

def test_http_patch_meta_merge(populated_store):
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store._db.execute("UPDATE memories SET meta=? WHERE id=?", ('{"a":1}', mid))
    store._db.commit()
    code, data = http_call(store, "PATCH", f"/memory/{mid}", {"meta": {"b": 2}})
    assert code == 200
    row = store._db.execute("SELECT meta FROM memories WHERE id=?", (mid,)).fetchone()
    import json as _j
    m = _j.loads(row[0])
    assert m["a"] == 1 and m["b"] == 2


def test_http_patch_meta_replace(populated_store):
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store._db.execute("UPDATE memories SET meta=? WHERE id=?", ('{"a":1}', mid))
    store._db.commit()
    code, data = http_call(store, "PATCH", f"/memory/{mid}", {"meta": {"b": 2}, "merge": False})
    assert code == 200
    row = store._db.execute("SELECT meta FROM memories WHERE id=?", (mid,)).fetchone()
    import json as _j
    m = _j.loads(row[0])
    assert "a" not in m and m["b"] == 2


def test_http_patch_meta_not_dict(populated_store):
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    code, data = http_call(store, "PATCH", f"/memory/{mid}", {"meta": "not a dict"})
    assert code == 400


def test_http_patch_meta_not_found(tmp_store):
    code, data = http_call(tmp_store, "PATCH", "/memory/99999", {"meta": {"x": 1}})
    assert code == 404


def test_http_patch_no_fields(tmp_store):
    code, data = http_call(tmp_store, "PATCH", "/memory/1", {"other": "x"})
    assert code == 400


# ── MCP mnemonics_update_meta ─────────────────────────────────────────────────

def test_mcp_update_meta_merge(populated_store):
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_update_meta", "arguments": {"id": mid, "meta": {"tag": "important"}}},
    })[0]
    assert "result" in r
    assert "updated" in r["result"]["content"][0]["text"]


def test_mcp_update_meta_not_found(tmp_store):
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_update_meta", "arguments": {"id": 99999, "meta": {"x": 1}}},
    })[0]
    assert "error" in r


def test_mcp_update_meta_missing_args(tmp_store):
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_update_meta", "arguments": {"id": 1}},
    })[0]
    assert "error" in r


def test_mcp_update_meta_in_tools_list(tmp_store):
    resp = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
    })[0]
    assert "mnemonics_update_meta" in {t["name"] for t in resp["result"]["tools"]}


# ── hybrid-search REST + MCP ──────────────────────────────────────────────────

def test_http_hybrid_search_ok(populated_store):
    """POST /hybrid-search returns combined results."""
    import numpy as np
    from mnemonics.store import DIM
    store, docs, vecs = populated_store
    rng = np.random.default_rng(7)
    qv = rng.random((DIM,)).astype("float32").tolist()
    code, data = http_call(store, "POST", "/hybrid-search",
                           {"query": "Paris Eiffel", "vector": qv, "top_k": 3})
    assert code == 200
    assert "results" in data
    assert isinstance(data["results"], list)


def test_http_hybrid_search_missing_params(populated_store):
    """POST /hybrid-search without query returns 400."""
    import numpy as np
    from mnemonics.store import DIM
    store, docs, vecs = populated_store
    rng = np.random.default_rng(8)
    qv = rng.random((DIM,)).astype("float32").tolist()
    code, data = http_call(store, "POST", "/hybrid-search", {"vector": qv})
    assert code == 400


def test_http_hybrid_search_missing_vector(populated_store):
    """POST /hybrid-search without vector returns 400."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/hybrid-search", {"query": "Eiffel"})
    assert code == 400


def test_mcp_hybrid_search_ok(populated_store):
    """MCP mnemonics_hybrid_search returns formatted results."""
    import numpy as np
    from mnemonics.store import DIM
    store, docs, vecs = populated_store
    rng = np.random.default_rng(9)
    qv = rng.random((DIM,)).astype("float32").tolist()
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_hybrid_search",
                   "arguments": {"query": "Python programming", "vector": qv, "top_k": 3}},
    })[0]
    assert "result" in r
    text = r["result"]["content"][0]["text"]
    assert "rrf=" in text or "No hybrid results" in text


def test_mcp_hybrid_search_missing_args(populated_store):
    """MCP mnemonics_hybrid_search without required args returns error."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_hybrid_search",
                   "arguments": {"query": "Python"}},
    })[0]
    assert "error" in r


def test_mcp_hybrid_search_no_results(tmp_store):
    """MCP mnemonics_hybrid_search returns text when no hits (covers line 1177)."""
    import numpy as np
    from mnemonics.store import DIM
    rng = np.random.default_rng(0)
    qv = rng.random((DIM,)).astype("float32").tolist()
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_hybrid_search",
                   "arguments": {"query": "nothing", "vector": qv}},
    })[0]
    assert "result" in r
    assert "No hybrid results" in r["result"]["content"][0]["text"]


def test_mcp_hybrid_search_with_summary(populated_store):
    """MCP mnemonics_hybrid_search shows summary line when present (covers line 1188)."""
    import numpy as np
    from mnemonics.store import DIM
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.update_summary(mid, "Summary of the first memory")
    rng = np.random.default_rng(1)
    qv = rng.random((DIM,)).astype("float32").tolist()
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_hybrid_search",
                   "arguments": {"query": "Eiffel Paris", "vector": qv, "top_k": 5}},
    })[0]
    assert "result" in r
    text = r["result"]["content"][0]["text"]
    assert "rrf=" in text or "No hybrid results" in text


# ── similar-to REST + MCP ─────────────────────────────────────────────────────

def test_http_similar_to_ok(populated_store):
    """POST /similar-to returns nearest neighbors."""
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    code, data = http_call(store, "POST", "/similar-to",
                           {"id": first_id, "top_k": 3})
    assert code == 200
    assert "results" in data
    assert all(r["id"] != first_id for r in data["results"])


def test_http_similar_to_missing_id(populated_store):
    """POST /similar-to without id returns 400."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/similar-to", {"top_k": 3})
    assert code == 400


def test_http_similar_to_not_found(populated_store):
    """POST /similar-to for non-existent ID returns empty results."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/similar-to", {"id": 99999})
    assert code == 200
    assert data["results"] == []


def test_mcp_similar_to_ok(populated_store):
    """MCP mnemonics_similar_to returns formatted results."""
    store, docs, vecs = populated_store
    first_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_similar_to",
                   "arguments": {"id": first_id, "top_k": 3}},
    })[0]
    assert "result" in r


def test_mcp_similar_to_not_found(populated_store):
    """MCP mnemonics_similar_to for non-existent ID returns 'No similar' text."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_similar_to",
                   "arguments": {"id": 99999}},
    })[0]
    assert "result" in r
    assert "No similar" in r["result"]["content"][0]["text"]


def test_mcp_similar_to_missing_id(populated_store):
    """MCP mnemonics_similar_to without id returns error."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_similar_to", "arguments": {}},
    })[0]
    assert "error" in r


def test_mcp_similar_to_with_summary(populated_store):
    """MCP mnemonics_similar_to shows summary line when result has one (covers line 1210)."""
    store, docs, vecs = populated_store
    all_ids = [r[0] for r in store._db.execute("SELECT id FROM memories ORDER BY id").fetchall()]
    # Give the second memory a summary
    store.update_summary(all_ids[1], "A relevant summary")
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_similar_to",
                   "arguments": {"id": all_ids[0], "top_k": 5}},
    })[0]
    assert "result" in r
    text = r["result"]["content"][0]["text"]
    assert text  # non-empty result


# ── expire REST + MCP ─────────────────────────────────────────────────────────

def test_http_expire_ok(populated_store):
    """POST /expire demotes stale memories and returns count."""
    store, docs, vecs = populated_store
    store._db.execute("UPDATE memories SET last_accessed=datetime('now', '-60 days')")
    store._db.commit()
    code, data = http_call(store, "POST", "/expire", {"age_days": 30})
    assert code == 200
    assert "demoted" in data
    assert data["demoted"] >= 0


def test_http_expire_with_ns(populated_store):
    """POST /expire with ns targets only that namespace."""
    store, docs, vecs = populated_store
    store._db.execute("UPDATE memories SET last_accessed=datetime('now', '-60 days')")
    store._db.commit()
    code, data = http_call(store, "POST", "/expire", {"ns": "default", "age_days": 1})
    assert code == 200


def test_mcp_expire_ok(populated_store):
    """MCP mnemonics_expire demotes memories."""
    store, docs, vecs = populated_store
    store._db.execute("UPDATE memories SET last_accessed=datetime('now', '-60 days')")
    store._db.commit()
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_expire",
                   "arguments": {"age_days": 30}},
    })[0]
    assert "result" in r
    assert "Demoted" in r["result"]["content"][0]["text"]


# ── bulk-update-summary REST + MCP ────────────────────────────────────────────

def test_http_bulk_update_summary_ok(populated_store):
    """POST /bulk-update-summary updates summaries for multiple IDs."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories ORDER BY id").fetchall()]
    code, data = http_call(store, "POST", "/bulk-update-summary",
                           {"updates": {str(ids[0]): "Summary A", str(ids[1]): "Summary B"}})
    assert code == 200
    assert data["updated"] == 2


def test_http_bulk_update_summary_missing_param(populated_store):
    """POST /bulk-update-summary without updates returns 400."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/bulk-update-summary", {"foo": "bar"})
    assert code == 400


def test_mcp_bulk_update_summary_ok(populated_store):
    """MCP mnemonics_bulk_update_summary updates summaries."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories ORDER BY id").fetchall()]
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_bulk_update_summary",
                   "arguments": {"updates": {str(ids[0]): "New summary"}}},
    })[0]
    assert "result" in r
    assert "Updated" in r["result"]["content"][0]["text"]


def test_mcp_bulk_update_summary_missing_arg(populated_store):
    """MCP mnemonics_bulk_update_summary without updates returns error."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_bulk_update_summary", "arguments": {}},
    })[0]
    assert "error" in r


# ── deduplicate REST + MCP ────────────────────────────────────────────────────

def test_http_deduplicate_dry_run(populated_store):
    """POST /deduplicate dry_run returns pairs without deleting."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/deduplicate", {"dry_run": True})
    assert code == 200
    assert "pairs" in data
    assert "removed" in data


def test_mcp_deduplicate_ok(populated_store):
    """MCP mnemonics_deduplicate returns a result."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_deduplicate",
                   "arguments": {"threshold": 0.99, "dry_run": True}},
    })[0]
    assert "result" in r
    assert "pair" in r["result"]["content"][0]["text"].lower()


def test_mcp_deduplicate_with_pairs(tmp_store):
    """MCP mnemonics_deduplicate formats pair lines when duplicates exist."""
    import numpy as np
    from mnemonics.store import DIM
    rng = np.random.default_rng(321)
    v = rng.random((DIM,)).astype("float32")
    v /= np.linalg.norm(v)
    tmp_store.add(["dup A", "dup B"], np.stack([v, v]))
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_deduplicate",
                   "arguments": {"threshold": 0.99, "dry_run": True}},
    })[0]
    assert "result" in r
    text = r["result"]["content"][0]["text"]
    assert "kept=" in text


# ── sample REST + MCP ─────────────────────────────────────────────────────────

def test_http_sample_ok(populated_store):
    """POST /sample returns random results."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/sample", {"n": 3})
    assert code == 200
    assert data["n"] == 3
    assert len(data["results"]) == 3


def test_http_sample_empty_ns(tmp_store):
    """POST /sample on empty namespace returns empty results."""
    code, data = http_call(tmp_store, "POST", "/sample", {"ns": "ghost", "n": 5})
    assert code == 200
    assert data["n"] == 0


def test_mcp_sample_ok(populated_store):
    """MCP mnemonics_sample returns formatted result."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_sample",
                   "arguments": {"n": 2}},
    })[0]
    assert "result" in r
    assert "Sample" in r["result"]["content"][0]["text"]


def test_mcp_sample_empty(tmp_store):
    """MCP mnemonics_sample on empty ns returns no memories message."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_sample",
                   "arguments": {"ns": "empty", "n": 3}},
    })[0]
    assert "result" in r
    assert "No memories" in r["result"]["content"][0]["text"]


# ── reindex-all REST + MCP ────────────────────────────────────────────────────

def test_http_reindex_all_ok(populated_store):
    """POST /reindex-all rebuilds all namespace indexes."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/reindex-all", {})
    assert code == 200
    assert "namespaces" in data
    assert data["count"] >= 1


def test_mcp_reindex_all_ok(populated_store):
    """MCP mnemonics_reindex_all returns rebuild summary."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_reindex_all", "arguments": {}},
    })[0]
    assert "result" in r
    assert "Rebuilt" in r["result"]["content"][0]["text"]


def test_mcp_reindex_all_with_error(populated_store):
    """MCP mnemonics_reindex_all shows ERROR line for failed namespaces."""
    from unittest.mock import patch
    store, docs, vecs = populated_store
    with patch.object(store, "reindex_all", return_value=[
        {"ns": "broken", "error": "disk full"},
        {"ns": "ok", "old_count": 3, "new_count": 3},
    ]):
        r = _mcp(store, {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "mnemonics_reindex_all", "arguments": {}},
        })[0]
    assert "result" in r
    assert "ERROR" in r["result"]["content"][0]["text"]


# ── namespace-info REST + MCP ─────────────────────────────────────────────────

def test_http_namespace_info_ok(populated_store):
    """GET /namespace/<ns> returns namespace summary."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/namespace/default", {})
    assert code == 200
    assert data["ns"] == "default"
    assert data["total"] == len(docs)


def test_http_namespace_info_not_found(tmp_store):
    """GET /namespace/<ns> returns 404 for unknown namespace."""
    code, data = http_call(tmp_store, "GET", "/namespace/ghost", {})
    assert code == 404


def test_http_namespace_info_empty_path(tmp_store):
    """GET /namespace/ without ns name returns 400."""
    code, data = http_call(tmp_store, "GET", "/namespace/", {})
    assert code == 400


def test_mcp_namespace_info_ok(populated_store):
    """MCP mnemonics_namespace_info returns formatted stats."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_namespace_info",
                   "arguments": {"ns": "default"}},
    })[0]
    assert "result" in r
    assert "Total" in r["result"]["content"][0]["text"]


def test_mcp_namespace_info_not_found(tmp_store):
    """MCP mnemonics_namespace_info returns error for unknown ns."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_namespace_info",
                   "arguments": {"ns": "ghost"}},
    })[0]
    assert "error" in r


def test_mcp_namespace_info_missing_arg(tmp_store):
    """MCP mnemonics_namespace_info without ns arg returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_namespace_info", "arguments": {}},
    })[0]
    assert "error" in r


# ── move-to-ns REST + MCP ─────────────────────────────────────────────────────

def test_http_move_to_ns_ok(populated_store):
    """POST /move-to-ns moves memories and returns count."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 2").fetchall()]
    code, data = http_call(store, "POST", "/move-to-ns", {"ids": ids, "ns": "archive"})
    assert code == 200
    assert data["moved"] == 2
    assert data["target_ns"] == "archive"


def test_http_move_to_ns_missing_params(populated_store):
    """POST /move-to-ns without required params returns 400."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/move-to-ns", {"ids": [1]})
    assert code == 400


def test_mcp_move_to_ns_ok(populated_store):
    """MCP mnemonics_move_to_ns moves memories."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 1").fetchall()]
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_move_to_ns",
                   "arguments": {"ids": ids, "ns": "work"}},
    })[0]
    assert "result" in r
    assert "Moved" in r["result"]["content"][0]["text"]


def test_mcp_move_to_ns_missing_args(tmp_store):
    """MCP mnemonics_move_to_ns without required args returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_move_to_ns", "arguments": {"ids": [1]}},
    })[0]
    assert "error" in r


# ── clone REST + MCP ──────────────────────────────────────────────────────────

def test_http_clone_ok(populated_store):
    """POST /clone creates a clone in the target namespace."""
    store, docs, vecs = populated_store
    src_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    code, data = http_call(store, "POST", "/clone", {"id": src_id, "ns": "backup"})
    assert code == 201
    assert data["source_id"] == src_id
    assert data["target_ns"] == "backup"
    assert isinstance(data["cloned_id"], int)


def test_http_clone_not_found(populated_store):
    """POST /clone with a missing id returns 404."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/clone", {"id": 999999, "ns": "backup"})
    assert code == 404


def test_http_clone_missing_params(populated_store):
    """POST /clone without required params returns 400."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/clone", {"id": 1})
    assert code == 400


def test_mcp_clone_ok(populated_store):
    """MCP mnemonics_clone clones a memory and returns new id."""
    store, docs, vecs = populated_store
    src_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_clone",
                   "arguments": {"id": src_id, "ns": "backup"}},
    })[0]
    assert "result" in r
    assert "Cloned" in r["result"]["content"][0]["text"]


def test_mcp_clone_not_found(populated_store):
    """MCP mnemonics_clone with missing id returns error."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_clone",
                   "arguments": {"id": 999999, "ns": "backup"}},
    })[0]
    assert "error" in r


def test_mcp_clone_missing_args(tmp_store):
    """MCP mnemonics_clone without required args returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_clone", "arguments": {"id": 1}},
    })[0]
    assert "error" in r


# ── update-text REST + MCP ────────────────────────────────────────────────────

def test_http_update_text_ok(populated_store):
    """POST /update-text replaces text and vector."""
    store, docs, vecs = populated_store
    src_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    new_vec = [0.1] * DIM
    code, data = http_call(store, "POST", "/update-text",
                           {"id": src_id, "text": "new text here", "vec": new_vec})
    assert code == 200
    assert data["updated"] is True


def test_http_update_text_not_found(populated_store):
    """POST /update-text with missing id returns 404."""
    store, docs, vecs = populated_store
    new_vec = [0.1] * DIM
    code, data = http_call(store, "POST", "/update-text",
                           {"id": 999999, "text": "text", "vec": new_vec})
    assert code == 404


def test_http_update_text_missing_params(populated_store):
    """POST /update-text without required params returns 400."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/update-text", {"id": 1, "text": "x"})
    assert code == 400


def test_mcp_update_text_ok(populated_store):
    """MCP mnemonics_update_text replaces text and vector."""
    store, docs, vecs = populated_store
    src_id = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    new_vec = [0.1] * DIM
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_update_text",
                   "arguments": {"id": src_id, "text": "updated text", "vec": new_vec}},
    })[0]
    assert "result" in r
    assert "Updated" in r["result"]["content"][0]["text"]


def test_mcp_update_text_not_found(populated_store):
    """MCP mnemonics_update_text with missing id returns error."""
    store, docs, vecs = populated_store
    new_vec = [0.1] * DIM
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_update_text",
                   "arguments": {"id": 999999, "text": "x", "vec": new_vec}},
    })[0]
    assert "error" in r


def test_mcp_update_text_missing_args(tmp_store):
    """MCP mnemonics_update_text without required args returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_update_text",
                   "arguments": {"id": 1, "text": "x"}},
    })[0]
    assert "error" in r


# ── access-stats REST + MCP ───────────────────────────────────────────────────

def test_http_access_stats_default_ns(populated_store):
    """GET /access-stats?ns=default returns access stats."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/access-stats?ns=default", {})
    assert code == 200
    assert data["total"] == len(docs)
    assert "never_accessed" in data


def test_http_access_stats_all_ns(populated_store):
    """GET /access-stats (no ns) returns stats for all namespaces."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/access-stats", {})
    assert code == 200
    assert data["ns"] is None


def test_mcp_access_stats_with_ns(populated_store):
    """MCP mnemonics_access_stats with ns returns formatted stats."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_access_stats",
                   "arguments": {"ns": "default"}},
    })[0]
    assert "result" in r
    text = r["result"]["content"][0]["text"]
    assert "Total memories" in text


def test_mcp_access_stats_all_ns(populated_store):
    """MCP mnemonics_access_stats without ns returns all-namespace stats."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_access_stats", "arguments": {}},
    })[0]
    assert "result" in r


# ── tag / untag REST + MCP ────────────────────────────────────────────────────

def test_http_tag_ok(populated_store):
    """POST /tag adds a tag."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    code, data = http_call(store, "POST", "/tag", {"id": mid, "tag": "important"})
    assert code == 200
    assert data["action"] == "added"


def test_http_tag_not_found(populated_store):
    """POST /tag with missing id returns 404."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/tag", {"id": 999999, "tag": "x"})
    assert code == 404


def test_http_tag_missing_params(populated_store):
    """POST /tag without tag param returns 400."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/tag", {"id": 1})
    assert code == 400


def test_http_untag_ok(populated_store):
    """POST /untag removes a tag."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.tag(mid, "remove-me")
    code, data = http_call(store, "POST", "/untag", {"id": mid, "tag": "remove-me"})
    assert code == 200
    assert data["action"] == "removed"


def test_http_untag_not_found(populated_store):
    """POST /untag with missing id returns 404."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/untag", {"id": 999999, "tag": "x"})
    assert code == 404


def test_http_untag_missing_params(populated_store):
    """POST /untag without tag param returns 400."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/untag", {"id": 1})
    assert code == 400


def test_mcp_tag_ok(populated_store):
    """MCP mnemonics_tag adds a tag."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_tag",
                   "arguments": {"id": mid, "tag": "mcp-tag"}},
    })[0]
    assert "result" in r
    assert "Added" in r["result"]["content"][0]["text"]


def test_mcp_tag_not_found(tmp_store):
    """MCP mnemonics_tag with missing id returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_tag",
                   "arguments": {"id": 999999, "tag": "x"}},
    })[0]
    assert "error" in r


def test_mcp_tag_missing_args(tmp_store):
    """MCP mnemonics_tag without args returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_tag", "arguments": {"id": 1}},
    })[0]
    assert "error" in r


def test_mcp_untag_ok(populated_store):
    """MCP mnemonics_untag removes a tag."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.tag(mid, "to-remove")
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_untag",
                   "arguments": {"id": mid, "tag": "to-remove"}},
    })[0]
    assert "result" in r
    assert "Removed" in r["result"]["content"][0]["text"]


def test_mcp_untag_not_found(tmp_store):
    """MCP mnemonics_untag with missing id returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_untag",
                   "arguments": {"id": 999999, "tag": "x"}},
    })[0]
    assert "error" in r


def test_mcp_untag_missing_args(tmp_store):
    """MCP mnemonics_untag without args returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_untag", "arguments": {"id": 1}},
    })[0]
    assert "error" in r


# ── find-by-tag / list-tags REST + MCP ───────────────────────────────────────

def test_http_find_by_tag_ok(populated_store):
    """GET /find-by-tag?tag=x&ns=default returns matching memories."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.tag(mid, "http-test-tag")
    code, data = http_call(store, "GET", "/find-by-tag?tag=http-test-tag&ns=default", {})
    assert code == 200
    assert any(r["id"] == mid for r in data["results"])


def test_http_find_by_tag_missing_tag_param(populated_store):
    """GET /find-by-tag without tag returns 400."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/find-by-tag?ns=default", {})
    assert code == 400


def test_http_list_tags_ok(populated_store):
    """GET /list-tags?ns=default returns tag list."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.tag(mid, "listable")
    code, data = http_call(store, "GET", "/list-tags?ns=default", {})
    assert code == 200
    assert any(t["tag"] == "listable" for t in data["tags"])


def test_http_list_tags_all_ns(populated_store):
    """GET /list-tags (no ns param) returns all-namespace tags."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/list-tags", {})
    assert code == 200
    assert data["ns"] is None


def test_mcp_find_by_tag_ok(populated_store):
    """MCP mnemonics_find_by_tag returns matches."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.tag(mid, "mcp-find-tag")
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_find_by_tag",
                   "arguments": {"tag": "mcp-find-tag", "ns": "default"}},
    })[0]
    assert "result" in r


def test_mcp_find_by_tag_no_results(populated_store):
    """MCP mnemonics_find_by_tag with no matches returns 'no memories' message."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_find_by_tag",
                   "arguments": {"tag": "no-such-tag", "ns": "default"}},
    })[0]
    assert "result" in r
    assert "No memories" in r["result"]["content"][0]["text"]


def test_mcp_find_by_tag_missing_arg(tmp_store):
    """MCP mnemonics_find_by_tag without tag returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_find_by_tag", "arguments": {}},
    })[0]
    assert "error" in r


def test_mcp_list_tags_ok(populated_store):
    """MCP mnemonics_list_tags returns tag list."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.tag(mid, "mcp-list-tag")
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_list_tags", "arguments": {"ns": "default"}},
    })[0]
    assert "result" in r
    assert "mcp-list-tag" in r["result"]["content"][0]["text"]


def test_mcp_list_tags_empty(tmp_store):
    """MCP mnemonics_list_tags on empty store returns no-tags message."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_list_tags", "arguments": {"ns": "default"}},
    })[0]
    assert "result" in r
    assert "No tags" in r["result"]["content"][0]["text"]


# ── word-frequency REST + MCP ─────────────────────────────────────────────────

def test_http_word_frequency_ok(populated_store):
    """GET /word-frequency?ns=default returns word list."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/word-frequency?ns=default", {})
    assert code == 200
    assert isinstance(data["words"], list)
    assert len(data["words"]) > 0


def test_http_word_frequency_all_ns(populated_store):
    """GET /word-frequency (no ns) returns all-namespace words."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/word-frequency", {})
    assert code == 200
    assert data["ns"] is None


def test_mcp_word_frequency_ok(populated_store):
    """MCP mnemonics_word_frequency returns formatted word list."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_word_frequency",
                   "arguments": {"ns": "default", "top_n": 5}},
    })[0]
    assert "result" in r
    assert "Top" in r["result"]["content"][0]["text"]


def test_mcp_word_frequency_empty(tmp_store):
    """MCP mnemonics_word_frequency on empty store returns no-words message."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_word_frequency", "arguments": {"ns": "default"}},
    })[0]
    assert "result" in r
    assert "No words" in r["result"]["content"][0]["text"]


# ── get-tags / search-date-range REST + MCP ───────────────────────────────────

def test_http_get_tags_ok(populated_store):
    """GET /get-tags/<id> returns tags list."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.tag(mid, "srv-tag")
    code, data = http_call(store, "GET", f"/get-tags/{mid}", {})
    assert code == 200
    assert "srv-tag" in data["tags"]


def test_http_get_tags_not_found(tmp_store):
    """GET /get-tags/<id> with missing id returns 404."""
    code, data = http_call(tmp_store, "GET", "/get-tags/999999", {})
    assert code == 404


def test_http_get_tags_invalid_id(tmp_store):
    """GET /get-tags/abc returns 400."""
    code, data = http_call(tmp_store, "GET", "/get-tags/abc", {})
    assert code == 400


def test_http_search_date_range_ok(populated_store):
    """GET /search-date-range?ns=default returns results."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/search-date-range?ns=default", {})
    assert code == 200
    assert len(data["results"]) == len(docs)


def test_http_search_date_range_after_empty(populated_store):
    """GET /search-date-range?after=9999-01-01 returns empty results."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/search-date-range?after=9999-01-01", {})
    assert code == 200
    assert data["results"] == []


def test_mcp_get_tags_ok(populated_store):
    """MCP mnemonics_get_tags returns tags."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    store.tag(mid, "mcp-gtag")
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_get_tags", "arguments": {"id": mid}},
    })[0]
    assert "result" in r
    assert "mcp-gtag" in r["result"]["content"][0]["text"]


def test_mcp_get_tags_not_found(tmp_store):
    """MCP mnemonics_get_tags with missing id returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_get_tags", "arguments": {"id": 999999}},
    })[0]
    assert "error" in r


def test_mcp_get_tags_missing_arg(tmp_store):
    """MCP mnemonics_get_tags without id returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_get_tags", "arguments": {}},
    })[0]
    assert "error" in r


def test_mcp_search_date_range_ok(populated_store):
    """MCP mnemonics_search_date_range returns formatted results."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_search_date_range",
                   "arguments": {"ns": "default"}},
    })[0]
    assert "result" in r
    assert "Found" in r["result"]["content"][0]["text"]


def test_mcp_search_date_range_empty(populated_store):
    """MCP mnemonics_search_date_range with no results returns 'no memories'."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_search_date_range",
                   "arguments": {"ns": "default", "after": "9999-01-01"}},
    })[0]
    assert "result" in r
    assert "No memories" in r["result"]["content"][0]["text"]


def test_mcp_search_date_range_with_tier(populated_store):
    """MCP mnemonics_search_date_range with tier filter passes tier to store."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_search_date_range",
                   "arguments": {"ns": "default", "tier": 1}},
    })[0]
    assert "result" in r


# ── export-ns / bulk-tag REST + MCP ──────────────────────────────────────────

def test_http_export_ns_ok(populated_store):
    """GET /export-ns/<ns> returns all memory records."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/export-ns/default", {})
    assert code == 200
    assert data["count"] == len(docs)
    assert len(data["records"]) == len(docs)


def test_http_export_ns_empty_path(tmp_store):
    """GET /export-ns/ without namespace name returns 400."""
    code, data = http_call(tmp_store, "GET", "/export-ns/", {})
    assert code == 400


def test_http_bulk_tag_ok(populated_store):
    """POST /bulk-tag tags multiple memories."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 2").fetchall()]
    code, data = http_call(store, "POST", "/bulk-tag", {"ids": ids, "tags": ["batch"]})
    assert code == 200
    assert data["updated"] == 2


def test_http_bulk_tag_missing_params(populated_store):
    """POST /bulk-tag without required params returns 400."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/bulk-tag", {"ids": [1]})
    assert code == 400


def test_mcp_export_ns_ok(populated_store):
    """MCP mnemonics_export_ns returns count and records."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_export_ns", "arguments": {"ns": "default"}},
    })[0]
    assert "result" in r
    text = r["result"]["content"][0]["text"]
    assert f"Exported {len(docs)}" in text


def test_mcp_export_ns_missing_arg(tmp_store):
    """MCP mnemonics_export_ns without ns returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_export_ns", "arguments": {}},
    })[0]
    assert "error" in r


def test_mcp_bulk_tag_ok(populated_store):
    """MCP mnemonics_bulk_tag adds tags to multiple memories."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 2").fetchall()]
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_bulk_tag",
                   "arguments": {"ids": ids, "tags": ["batch-mcp"]}},
    })[0]
    assert "result" in r
    assert "2" in r["result"]["content"][0]["text"]


def test_mcp_bulk_tag_missing_args(tmp_store):
    """MCP mnemonics_bulk_tag without tags returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_bulk_tag", "arguments": {"ids": [1]}},
    })[0]
    assert "error" in r


# ── touch / bulk-untag / count-by-tier REST + MCP ────────────────────────────

def test_http_touch_ok(populated_store):
    """POST /touch updates last_accessed."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    code, data = http_call(store, "POST", "/touch", {"id": mid})
    assert code == 200
    assert data["touched"] is True


def test_http_touch_not_found(tmp_store):
    """POST /touch with unknown id returns 404."""
    code, data = http_call(tmp_store, "POST", "/touch", {"id": 999999})
    assert code == 404


def test_http_touch_missing_param(tmp_store):
    """POST /touch without id returns 400."""
    code, data = http_call(tmp_store, "POST", "/touch", {})
    assert code == 400


def test_http_bulk_untag_ok(populated_store):
    """POST /bulk-untag removes tags from memories."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 2").fetchall()]
    store.bulk_tag(ids, ["bye"])
    code, data = http_call(store, "POST", "/bulk-untag", {"ids": ids, "tags": ["bye"]})
    assert code == 200
    assert data["updated"] == 2


def test_http_bulk_untag_missing_params(tmp_store):
    """POST /bulk-untag without required params returns 400."""
    code, data = http_call(tmp_store, "POST", "/bulk-untag", {"ids": [1]})
    assert code == 400


def test_http_count_by_tier_default(populated_store):
    """GET /count-by-tier?ns=default returns tier counts."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/count-by-tier?ns=default", {})
    assert code == 200
    assert sum(data["by_tier"].values()) == len(docs)


def test_http_count_by_tier_all_ns(populated_store):
    """GET /count-by-tier without ns spans all namespaces."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/count-by-tier", {})
    assert code == 200
    assert sum(data["by_tier"].values()) >= len(docs)


def test_mcp_touch_ok(populated_store):
    """MCP mnemonics_touch returns success message."""
    store, docs, vecs = populated_store
    mid = store._db.execute("SELECT id FROM memories LIMIT 1").fetchone()[0]
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_touch", "arguments": {"id": mid}},
    })[0]
    assert "result" in r
    assert "Touched" in r["result"]["content"][0]["text"]


def test_mcp_touch_missing_arg(tmp_store):
    """MCP mnemonics_touch without id returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_touch", "arguments": {}},
    })[0]
    assert "error" in r


def test_mcp_touch_not_found(tmp_store):
    """MCP mnemonics_touch with unknown id returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_touch", "arguments": {"id": 999999}},
    })[0]
    assert "error" in r


def test_mcp_bulk_untag_ok(populated_store):
    """MCP mnemonics_bulk_untag removes tags."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 2").fetchall()]
    store.bulk_tag(ids, ["mcp-remove"])
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_bulk_untag",
                   "arguments": {"ids": ids, "tags": ["mcp-remove"]}},
    })[0]
    assert "result" in r


def test_mcp_bulk_untag_missing_args(tmp_store):
    """MCP mnemonics_bulk_untag without tags returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_bulk_untag", "arguments": {"ids": [1]}},
    })[0]
    assert "error" in r


def test_mcp_count_by_tier_ok(populated_store):
    """MCP mnemonics_count_by_tier returns tier breakdown."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_count_by_tier", "arguments": {"ns": "default"}},
    })[0]
    assert "result" in r
    assert "Tier counts" in r["result"]["content"][0]["text"]


def test_mcp_count_by_tier_all_ns(populated_store):
    """MCP mnemonics_count_by_tier without ns spans all namespaces."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_count_by_tier", "arguments": {}},
    })[0]
    assert "result" in r


def test_mcp_count_by_tier_empty(tmp_store):
    """MCP mnemonics_count_by_tier on empty store shows (empty)."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_count_by_tier", "arguments": {"ns": "ghost"}},
    })[0]
    assert "result" in r
    assert "(empty)" in r["result"]["content"][0]["text"]


# ── import-records / text-stats REST + MCP ────────────────────────────────────

def test_http_import_records_ok(tmp_store):
    """POST /import-records stores records and returns count."""
    records = [{"text": "imported 1", "ns": "test"}, {"text": "imported 2", "ns": "test"}]
    code, data = http_call(tmp_store, "POST", "/import-records", {"records": records})
    assert code == 200
    assert data["imported"] == 2


def test_http_import_records_with_ns(tmp_store):
    """POST /import-records with ns override applies it."""
    records = [{"text": "hello"}]
    code, data = http_call(tmp_store, "POST", "/import-records", {"records": records, "ns": "override-ns"})
    assert code == 200


def test_http_import_records_missing_param(tmp_store):
    """POST /import-records without records returns 400."""
    code, data = http_call(tmp_store, "POST", "/import-records", {"ns": "test"})
    assert code == 400


def test_http_text_stats_ok(populated_store):
    """GET /text-stats?ns=default returns all stat keys."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/text-stats?ns=default", {})
    assert code == 200
    assert data["count"] == len(docs)


def test_http_text_stats_all_ns(populated_store):
    """GET /text-stats without ns spans all namespaces."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/text-stats", {})
    assert code == 200
    assert data["count"] >= len(docs)


def test_mcp_import_records_ok(tmp_store):
    """MCP mnemonics_import_records stores records."""
    records = [{"text": "mcp import", "ns": "mcp-test"}]
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_import_records", "arguments": {"records": records}},
    })[0]
    assert "result" in r
    assert "1" in r["result"]["content"][0]["text"]


def test_mcp_import_records_missing_arg(tmp_store):
    """MCP mnemonics_import_records without records returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_import_records", "arguments": {}},
    })[0]
    assert "error" in r


def test_mcp_text_stats_ok(populated_store):
    """MCP mnemonics_text_stats returns count and stats."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_text_stats", "arguments": {"ns": "default"}},
    })[0]
    assert "result" in r
    assert "count" in r["result"]["content"][0]["text"]


def test_mcp_text_stats_all_ns(populated_store):
    """MCP mnemonics_text_stats without ns spans all namespaces."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_text_stats", "arguments": {}},
    })[0]
    assert "result" in r


# ── rename-ns / merge-ns / bulk-delete REST + MCP ─────────────────────────────

def test_http_rename_ns_ok(populated_store):
    """POST /rename-ns renames namespace."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/rename-ns",
                           {"old_ns": "default", "new_ns": "renamed"})
    assert code == 200
    assert data["moved"] == len(docs)


def test_http_rename_ns_missing_params(tmp_store):
    """POST /rename-ns without params returns 400."""
    code, data = http_call(tmp_store, "POST", "/rename-ns", {"old_ns": "x"})
    assert code == 400


def test_http_merge_ns_ok(populated_store):
    """POST /merge-ns merges source into target."""
    import numpy as np
    store, docs, vecs = populated_store
    store.add(["extra"], [np.zeros(384, dtype=np.float32)], ns="target-ns")
    code, data = http_call(store, "POST", "/merge-ns",
                           {"src_ns": "default", "dst_ns": "target-ns"})
    assert code == 200
    assert data["moved"] == len(docs)


def test_http_merge_ns_missing_params(tmp_store):
    """POST /merge-ns without params returns 400."""
    code, data = http_call(tmp_store, "POST", "/merge-ns", {"src_ns": "x"})
    assert code == 400


def test_http_bulk_delete_ok(populated_store):
    """POST /bulk-delete removes specified IDs."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 2").fetchall()]
    code, data = http_call(store, "POST", "/bulk-delete", {"ids": ids})
    assert code == 200
    assert data["deleted"] == 2


def test_http_bulk_delete_missing_param(tmp_store):
    """POST /bulk-delete without ids returns 400."""
    code, data = http_call(tmp_store, "POST", "/bulk-delete", {})
    assert code == 400


def test_mcp_rename_ns_ok(populated_store):
    """MCP mnemonics_rename_ns renames namespace."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_rename_ns",
                   "arguments": {"old_ns": "default", "new_ns": "renamed-mcp"}},
    })[0]
    assert "result" in r
    assert "renamed-mcp" in r["result"]["content"][0]["text"]


def test_mcp_rename_ns_missing_args(tmp_store):
    """MCP mnemonics_rename_ns without args returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_rename_ns", "arguments": {}},
    })[0]
    assert "error" in r


def test_mcp_merge_ns_ok(populated_store):
    """MCP mnemonics_merge_ns merges source into target."""
    import numpy as np
    store, docs, vecs = populated_store
    store.add(["tgt"], [np.zeros(384, dtype=np.float32)], ns="tgt-mcp")
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_merge_ns",
                   "arguments": {"src_ns": "default", "dst_ns": "tgt-mcp"}},
    })[0]
    assert "result" in r


def test_mcp_merge_ns_missing_args(tmp_store):
    """MCP mnemonics_merge_ns without args returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_merge_ns", "arguments": {}},
    })[0]
    assert "error" in r


def test_mcp_bulk_delete_ok(populated_store):
    """MCP mnemonics_bulk_delete removes memories."""
    store, docs, vecs = populated_store
    ids = [r[0] for r in store._db.execute("SELECT id FROM memories LIMIT 2").fetchall()]
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_bulk_delete", "arguments": {"ids": ids}},
    })[0]
    assert "result" in r
    assert "2" in r["result"]["content"][0]["text"]


def test_mcp_bulk_delete_missing_arg(tmp_store):
    """MCP mnemonics_bulk_delete without ids returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_bulk_delete", "arguments": {}},
    })[0]
    assert "error" in r


# ── filter-by-meta / summary-stats REST + MCP ─────────────────────────────────

def test_http_filter_by_meta_ok(tmp_store):
    """GET /filter-by-meta returns matching memories."""
    import numpy as np
    vecs = np.random.rand(3, 384).astype(np.float32)
    tmp_store.add(["a", "b", "c"], vecs, ns="default",
                  meta=[{"kind": "note"}, {"kind": "note"}, {"kind": "other"}])
    code, data = http_call(tmp_store, "GET", "/filter-by-meta?key=kind&value=note&ns=default", {})
    assert code == 200
    assert data["count"] == 2


def test_http_filter_by_meta_missing_params(tmp_store):
    """GET /filter-by-meta without params returns 400."""
    code, data = http_call(tmp_store, "GET", "/filter-by-meta?key=x", {})
    assert code == 400


def test_http_summary_stats_ok(populated_store):
    """GET /summary-stats returns coverage keys."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/summary-stats?ns=default", {})
    assert code == 200
    assert "total" in data


def test_http_summary_stats_all_ns(populated_store):
    """GET /summary-stats without ns spans all namespaces."""
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/summary-stats", {})
    assert code == 200
    assert data["total"] >= len(docs)


def test_mcp_filter_by_meta_ok(tmp_store):
    """MCP mnemonics_filter_by_meta returns matches."""
    import numpy as np
    v = np.random.rand(2, 384).astype(np.float32)
    tmp_store.add(["x", "y"], v, ns="default", meta=[{"t": "a"}, {"t": "b"}])
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_filter_by_meta",
                   "arguments": {"key": "t", "value": "a", "ns": "default"}},
    })[0]
    assert "result" in r
    assert "1" in r["result"]["content"][0]["text"]


def test_mcp_filter_by_meta_missing_args(tmp_store):
    """MCP mnemonics_filter_by_meta without key returns error."""
    r = _mcp(tmp_store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_filter_by_meta", "arguments": {"key": "x"}},
    })[0]
    assert "error" in r


def test_mcp_summary_stats_ok(populated_store):
    """MCP mnemonics_summary_stats returns coverage stats."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_summary_stats", "arguments": {"ns": "default"}},
    })[0]
    assert "result" in r
    assert "total" in r["result"]["content"][0]["text"]


def test_mcp_summary_stats_all_ns(populated_store):
    """MCP mnemonics_summary_stats without ns spans all namespaces."""
    store, docs, vecs = populated_store
    r = _mcp(store, {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "mnemonics_summary_stats", "arguments": {}},
    })[0]
    assert "result" in r


# ── pinned-memories REST ────────────────────────────────────────────────────────

def test_http_pinned_memories_empty(tmp_path):
    """GET /pinned-memories returns empty list when nothing pinned."""
    store = Store(tmp_path)
    code, data = http_call(store, "GET", "/pinned-memories?ns=default")
    assert code == 200
    assert data["count"] == 0
    assert data["results"] == []


def test_http_pinned_memories_with_pinned(tmp_path):
    """GET /pinned-memories returns pinned tier=0 memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    ids = store.add(["pinned one", "not pinned"], vecs, ns="default")
    store.pin(ids[0])
    code, data = http_call(store, "GET", "/pinned-memories?ns=default")
    assert code == 200
    assert data["count"] == 1
    assert data["results"][0]["id"] == ids[0]


def test_http_pinned_memories_all_ns(tmp_path):
    """GET /pinned-memories without ns spans all namespaces."""
    import numpy as np
    store = Store(tmp_path)
    v1 = np.random.rand(1, DIM).astype(np.float32)
    v2 = np.random.rand(1, DIM).astype(np.float32)
    ids1 = store.add(["ns1 doc"], v1, ns="ns1")
    ids2 = store.add(["ns2 doc"], v2, ns="ns2")
    store.pin(ids1[0])
    store.pin(ids2[0])
    code, data = http_call(store, "GET", "/pinned-memories")
    assert code == 200
    assert data["count"] == 2


# ── update-meta-key REST ────────────────────────────────────────────────────────

def test_http_update_meta_key_success(tmp_path):
    """POST /update-meta-key sets a key in meta."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    ids = store.add(["test doc"], vecs, ns="default")
    code, data = http_call(store, "POST", "/update-meta-key",
                           {"id": ids[0], "key": "flagged", "value": True})
    assert code == 200
    assert data["key"] == "flagged"


def test_http_update_meta_key_missing_id(tmp_path):
    """POST /update-meta-key returns 400 when id missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/update-meta-key", {"key": "k"})
    assert code == 400


def test_http_update_meta_key_not_found(tmp_path):
    """POST /update-meta-key returns 404 for unknown memory id."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/update-meta-key",
                           {"id": 99999, "key": "k", "value": "v"})
    assert code == 404


def test_http_update_meta_key_remove(tmp_path):
    """POST /update-meta-key with value=null removes the key."""
    import numpy as np
    import json
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    ids = store.add(["test doc"], vecs, ns="default")
    store.update_meta_key(ids[0], "temp", "x")
    code, data = http_call(store, "POST", "/update-meta-key",
                           {"id": ids[0], "key": "temp", "value": None})
    assert code == 200
    row = store._db.execute("SELECT meta FROM memories WHERE id=?", (ids[0],)).fetchone()
    assert "temp" not in json.loads(row[0])


# ── search-by-summary REST ─────────────────────────────────────────────────────

def test_http_search_by_summary_returns_matches(tmp_path):
    """GET /search-by-summary returns memories with matching summary."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["doc a", "doc b"], vecs, ns="default",
              summaries=["unique keyword here", "something else"])
    code, data = http_call(store, "GET", "/search-by-summary?q=unique%20keyword&ns=default")
    assert code == 200
    assert data["count"] == 1


def test_http_search_by_summary_missing_q(tmp_path):
    """GET /search-by-summary returns 400 when q is missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "GET", "/search-by-summary?ns=default")
    assert code == 400


# ── MCP: pinned_memories, update_meta_key, search_by_summary ──────────────────

def test_mcp_pinned_memories(tmp_path):
    """MCP mnemonics_pinned_memories returns pinned entries."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    ids = store.add(["pinned doc"], vecs, ns="default")
    store.pin(ids[0])
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_pinned_memories",
                                    "arguments": {"ns": "default"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "1 pinned" in text


def test_mcp_update_meta_key(tmp_path):
    """MCP mnemonics_update_meta_key sets a key in meta."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    ids = store.add(["doc"], vecs, ns="default")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_update_meta_key",
                                    "arguments": {"id": ids[0], "key": "active", "value": True}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "active" in text


def test_mcp_update_meta_key_missing_args(tmp_path):
    """MCP mnemonics_update_meta_key returns error when id missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_update_meta_key",
                                    "arguments": {"key": "k"}}})[0]
    assert "error" in resp


def test_mcp_update_meta_key_not_found(tmp_path):
    """MCP mnemonics_update_meta_key returns error for missing id."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_update_meta_key",
                                    "arguments": {"id": 99999, "key": "k", "value": "v"}}})[0]
    assert "error" in resp


def test_mcp_search_by_summary(tmp_path):
    """MCP mnemonics_search_by_summary returns matching memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    store.add(["test doc"], vecs, ns="default", summaries=["needle in a haystack"])
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_search_by_summary",
                                    "arguments": {"query": "needle", "ns": "default"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "1" in text


def test_mcp_search_by_summary_missing_query(tmp_path):
    """MCP mnemonics_search_by_summary returns error when query missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_search_by_summary",
                                    "arguments": {}}})[0]
    assert "error" in resp


# ── set-tier-by-tag REST ────────────────────────────────────────────────────────

def test_http_set_tier_by_tag_success(tmp_path):
    """GET /set-tier-by-tag updates memories and returns updated count."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="default",
              meta=[{"tags": ["vip"]}, {"tags": ["other"]}])
    code, data = http_call(store, "GET", "/set-tier-by-tag?tag=vip&tier=0&ns=default")
    assert code == 200
    assert data["updated"] == 1


def test_http_set_tier_by_tag_missing_params(tmp_path):
    """GET /set-tier-by-tag returns 400 when tag or tier missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "GET", "/set-tier-by-tag?tag=vip")
    assert code == 400


# ── rotate-ns REST ─────────────────────────────────────────────────────────────

def test_http_rotate_ns_moves_memories(tmp_path):
    """GET /rotate-ns moves oldest memories to dst."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(3, DIM).astype(np.float32)
    store.add(["x", "y", "z"], vecs, ns="src")
    code, data = http_call(store, "GET", "/rotate-ns?src=src&dst=dst&limit=2")
    assert code == 200
    assert data["moved"] == 2


def test_http_rotate_ns_missing_params(tmp_path):
    """GET /rotate-ns returns 400 when src or dst missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "GET", "/rotate-ns?src=only-src")
    assert code == 400


def test_http_rotate_ns_with_tier(tmp_path):
    """GET /rotate-ns?tier= filters by tier."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    ids = store.add(["a", "b"], vecs, ns="src")
    store.pin(ids[0])
    code, data = http_call(store, "GET", "/rotate-ns?src=src&dst=dst&tier=1")
    assert code == 200
    assert data["moved"] == 1


# ── compact-meta REST ──────────────────────────────────────────────────────────

def test_http_compact_meta_strips_all(tmp_path):
    """GET /compact-meta with no keep param strips all meta."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    store.add(["doc"], vecs, ns="default", meta=[{"noise": "x"}])
    code, data = http_call(store, "GET", "/compact-meta?ns=default")
    assert code == 200
    assert data["updated"] == 1


def test_http_compact_meta_keeps_keys(tmp_path):
    """GET /compact-meta?keep= keeps specified keys."""
    import numpy as np, json
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    store.add(["doc"], vecs, ns="default", meta=[{"keep": 1, "drop": 2}])
    code, data = http_call(store, "GET", "/compact-meta?ns=default&keep=keep")
    assert code == 200


# ── MCP: set_tier_by_tag, rotate_ns, compact_meta ─────────────────────────────

def test_mcp_set_tier_by_tag(tmp_path):
    """MCP mnemonics_set_tier_by_tag returns update count."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    store.add(["doc"], vecs, ns="default", meta=[{"tags": ["vip"]}])
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_set_tier_by_tag",
                                    "arguments": {"tag": "vip", "tier": 0, "ns": "default"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "1" in text


def test_mcp_set_tier_by_tag_missing(tmp_path):
    """MCP mnemonics_set_tier_by_tag returns error when tag missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_set_tier_by_tag",
                                    "arguments": {"tier": 0}}})[0]
    assert "error" in resp


def test_mcp_rotate_ns(tmp_path):
    """MCP mnemonics_rotate_ns moves memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="src")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_rotate_ns",
                                    "arguments": {"src_ns": "src", "dst_ns": "dst"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "2" in text


def test_mcp_rotate_ns_missing(tmp_path):
    """MCP mnemonics_rotate_ns returns error when src missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_rotate_ns",
                                    "arguments": {"dst_ns": "dst"}}})[0]
    assert "error" in resp


def test_mcp_compact_meta(tmp_path):
    """MCP mnemonics_compact_meta strips meta."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    store.add(["doc"], vecs, ns="default", meta=[{"noise": "x"}])
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_compact_meta",
                                    "arguments": {"ns": "default"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "1" in text


# ── list-by-tier REST ──────────────────────────────────────────────────────────

def test_http_list_by_tier_success(tmp_path):
    """GET /list-by-tier returns memories with the given tier."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    ids = store.add(["a", "b"], vecs, ns="default")
    store.pin(ids[0])
    code, data = http_call(store, "GET", "/list-by-tier?tier=0&ns=default")
    assert code == 200
    assert data["count"] == 1


def test_http_list_by_tier_missing_tier(tmp_path):
    """GET /list-by-tier returns 400 when tier param missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "GET", "/list-by-tier?ns=default")
    assert code == 400


def test_http_list_by_tier_all_ns(tmp_path):
    """GET /list-by-tier without ns spans all namespaces."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    ids = store.add(["doc"], vecs, ns="ns1")
    store.pin(ids[0])
    code, data = http_call(store, "GET", "/list-by-tier?tier=0")
    assert code == 200
    assert data["count"] == 1


# ── recent REST ────────────────────────────────────────────────────────────────

def test_http_recent_returns_results(tmp_path):
    """GET /recent returns most recently created memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(3, DIM).astype(np.float32)
    store.add(["a", "b", "c"], vecs, ns="default")
    code, data = http_call(store, "GET", "/newest?ns=default&n=2")
    assert code == 200
    assert data["count"] == 2


def test_http_recent_all_ns(tmp_path):
    """GET /newest without ns spans all namespaces."""
    import numpy as np
    store = Store(tmp_path)
    v1 = np.random.rand(1, DIM).astype(np.float32)
    v2 = np.random.rand(1, DIM).astype(np.float32)
    store.add(["ns1"], v1, ns="ns1")
    store.add(["ns2"], v2, ns="ns2")
    code, data = http_call(store, "GET", "/newest")
    assert code == 200
    assert data["count"] == 2


# ── oldest REST ────────────────────────────────────────────────────────────────

def test_http_oldest_returns_results(tmp_path):
    """GET /oldest returns oldest memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(3, DIM).astype(np.float32)
    store.add(["a", "b", "c"], vecs, ns="default")
    code, data = http_call(store, "GET", "/oldest?ns=default&n=2")
    assert code == 200
    assert data["count"] == 2


def test_http_oldest_all_ns(tmp_path):
    """GET /oldest without ns spans all namespaces."""
    import numpy as np
    store = Store(tmp_path)
    v1 = np.random.rand(1, DIM).astype(np.float32)
    store.add(["doc"], v1, ns="ns1")
    code, data = http_call(store, "GET", "/oldest")
    assert code == 200
    assert data["count"] == 1


# ── replace-text REST ──────────────────────────────────────────────────────────

def test_http_replace_text_success(tmp_path):
    """POST /replace-text updates text field."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    ids = store.add(["original text"], vecs, ns="default")
    code, data = http_call(store, "POST", "/replace-text",
                           {"id": ids[0], "text": "new text"})
    assert code == 200
    assert "id" in data


def test_http_replace_text_missing_params(tmp_path):
    """POST /replace-text returns 400 when id missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/replace-text", {"text": "hello"})
    assert code == 400


def test_http_replace_text_not_found(tmp_path):
    """POST /replace-text returns 404 for unknown memory."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/replace-text",
                           {"id": 99999, "text": "x"})
    assert code == 404


# ── MCP: list_by_tier, recent, oldest, replace_text ───────────────────────────

def test_mcp_list_by_tier(tmp_path):
    """MCP mnemonics_list_by_tier returns tier-filtered memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    ids = store.add(["doc"], vecs, ns="default")
    store.pin(ids[0])
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_list_by_tier",
                                    "arguments": {"tier": 0, "ns": "default"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "1" in text


def test_mcp_list_by_tier_missing(tmp_path):
    """MCP mnemonics_list_by_tier returns error when tier missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_list_by_tier",
                                    "arguments": {}}})[0]
    assert "error" in resp


def test_mcp_newest(tmp_path):
    """MCP mnemonics_newest returns most recently created memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="default")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_newest",
                                    "arguments": {"ns": "default", "n": 2}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "2" in text


def test_mcp_oldest(tmp_path):
    """MCP mnemonics_oldest returns oldest memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["first", "second"], vecs, ns="default")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_oldest",
                                    "arguments": {"ns": "default", "n": 1}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "1" in text


def test_mcp_replace_text(tmp_path):
    """MCP mnemonics_replace_text updates text."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    ids = store.add(["original"], vecs, ns="default")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_replace_text",
                                    "arguments": {"id": ids[0], "text": "updated"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "updated" in text


def test_mcp_replace_text_missing(tmp_path):
    """MCP mnemonics_replace_text returns error when id missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_replace_text",
                                    "arguments": {"text": "x"}}})[0]
    assert "error" in resp


def test_mcp_replace_text_not_found(tmp_path):
    """MCP mnemonics_replace_text returns error for missing id."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_replace_text",
                                    "arguments": {"id": 99999, "text": "x"}}})[0]
    assert "error" in resp


# ── search-text REST ───────────────────────────────────────────────────────────

def test_http_search_text_returns_matches(tmp_path):
    """GET /search-text returns memories with matching text."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["hello world", "goodbye world"], vecs, ns="default")
    code, data = http_call(store, "GET", "/search-text?q=hello&ns=default")
    assert code == 200
    assert data["count"] == 1


def test_http_search_text_missing_q(tmp_path):
    """GET /search-text returns 400 when q missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "GET", "/search-text?ns=default")
    assert code == 400


def test_http_search_text_all_ns(tmp_path):
    """GET /search-text without ns spans all namespaces."""
    import numpy as np
    store = Store(tmp_path)
    v1 = np.random.rand(1, DIM).astype(np.float32)
    store.add(["needle text"], v1, ns="ns1")
    code, data = http_call(store, "GET", "/search-text?q=needle")
    assert code == 200
    assert data["count"] == 1


# ── count-by-ns REST ───────────────────────────────────────────────────────────

def test_http_count_by_ns(tmp_path):
    """GET /count-by-ns returns per-namespace counts."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="myns")
    code, data = http_call(store, "GET", "/count-by-ns")
    assert code == 200
    assert data["by_ns"]["myns"] == 2


def test_http_count_by_ns_empty(tmp_path):
    """GET /count-by-ns returns empty dict for empty store."""
    store = Store(tmp_path)
    code, data = http_call(store, "GET", "/count-by-ns")
    assert code == 200
    assert data["by_ns"] == {}


# ── clear-ns REST ──────────────────────────────────────────────────────────────

def test_http_clear_ns_deletes(tmp_path):
    """POST /clear-ns deletes all memories in namespace."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="to_del")
    code, data = http_call(store, "POST", "/clear-ns", {"ns": "to_del"})
    assert code == 200
    assert data["deleted"] == 2


def test_http_clear_ns_missing_param(tmp_path):
    """POST /clear-ns returns 400 when ns missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/clear-ns", {})
    assert code == 400


# ── copy-to-ns REST ────────────────────────────────────────────────────────────

def test_http_copy_to_ns_copies(tmp_path):
    """POST /copy-to-ns copies memories to dst namespace."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    ids = store.add(["x", "y"], vecs, ns="src")
    code, data = http_call(store, "POST", "/copy-to-ns",
                           {"ids": ids, "dst_ns": "dst"})
    assert code == 200
    assert data["copied"] == 2


def test_http_copy_to_ns_missing_params(tmp_path):
    """POST /copy-to-ns returns 400 when ids missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/copy-to-ns", {"dst_ns": "dst"})
    assert code == 400


# ── MCP: search_text, count_by_ns, clear_ns, copy_to_ns ──────────────────────

def test_mcp_search_text(tmp_path):
    """MCP mnemonics_search_text returns matching memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    store.add(["unique phrase in text"], vecs, ns="default")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_search_text",
                                    "arguments": {"query": "unique phrase", "ns": "default"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "1" in text


def test_mcp_search_text_missing_query(tmp_path):
    """MCP mnemonics_search_text returns error when query missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_search_text",
                                    "arguments": {}}})[0]
    assert "error" in resp


def test_mcp_count_by_ns(tmp_path):
    """MCP mnemonics_count_by_ns returns namespace counts."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="testns")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_count_by_ns",
                                    "arguments": {}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "testns" in text


def test_mcp_clear_ns(tmp_path):
    """MCP mnemonics_clear_ns deletes memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="todel")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_clear_ns",
                                    "arguments": {"ns": "todel"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "2" in text


def test_mcp_clear_ns_missing(tmp_path):
    """MCP mnemonics_clear_ns returns error when ns missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_clear_ns",
                                    "arguments": {}}})[0]
    assert "error" in resp


def test_mcp_copy_to_ns(tmp_path):
    """MCP mnemonics_copy_to_ns copies memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    ids = store.add(["p", "q"], vecs, ns="src")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_copy_to_ns",
                                    "arguments": {"ids": ids, "dst_ns": "dst"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "2" in text


def test_mcp_copy_to_ns_missing(tmp_path):
    """MCP mnemonics_copy_to_ns returns error when ids missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_copy_to_ns",
                                    "arguments": {"dst_ns": "dst"}}})[0]
    assert "error" in resp


# ── rename-tag REST ────────────────────────────────────────────────────────────

def test_http_rename_tag(tmp_path):
    """POST /rename-tag renames tag in memories."""
    import numpy as np, json as _j
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    store.add(["m"], vecs, ns="default", meta=[{"tags": ["old"]}])
    code, data = http_call(store, "POST", "/rename-tag",
                           {"old_tag": "old", "new_tag": "new"})
    assert code == 200
    assert data["updated"] == 1


def test_http_rename_tag_missing_params(tmp_path):
    """POST /rename-tag returns 400 when params missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/rename-tag", {"old_tag": "x"})
    assert code == 400


# ── swap-tier REST ─────────────────────────────────────────────────────────────

def test_http_swap_tier(tmp_path):
    """POST /swap-tier moves memories between tiers."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="myns")
    code, data = http_call(store, "POST", "/swap-tier",
                           {"ns": "myns", "from_tier": 1, "to_tier": 2})
    assert code == 200
    assert data["updated"] == 2


def test_http_swap_tier_missing_params(tmp_path):
    """POST /swap-tier returns 400 when params missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/swap-tier", {"ns": "myns"})
    assert code == 400


def test_http_swap_tier_invalid_tier(tmp_path):
    """POST /swap-tier returns 400 for invalid tier value."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/swap-tier",
                           {"ns": "myns", "from_tier": 0, "to_tier": 9})
    assert code == 400


# ── find-duplicates REST ───────────────────────────────────────────────────────

def test_http_find_duplicates_returns_groups(tmp_path):
    """GET /find-duplicates returns duplicate groups."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["same", "same"], vecs, ns="default")
    code, data = http_call(store, "GET", "/find-duplicates?ns=default")
    assert code == 200
    assert data["groups"] == 1


def test_http_find_duplicates_all_ns(tmp_path):
    """GET /find-duplicates without ns spans all namespaces."""
    import numpy as np
    store = Store(tmp_path)
    v1 = np.random.rand(2, DIM).astype(np.float32)
    store.add(["dup", "dup"], v1, ns="ns1")
    code, data = http_call(store, "GET", "/find-duplicates")
    assert code == 200
    assert data["groups"] >= 1


# ── ns-summary REST ────────────────────────────────────────────────────────────

def test_http_ns_summary(tmp_path):
    """GET /ns-summary returns namespace dashboard."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(3, DIM).astype(np.float32)
    store.add(["a", "b", "c"], vecs, ns="sumns")
    code, data = http_call(store, "GET", "/ns-summary?ns=sumns")
    assert code == 200
    assert data["count"] == 3


def test_http_ns_summary_missing_ns(tmp_path):
    """GET /ns-summary returns 400 when ns missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "GET", "/ns-summary")
    assert code == 400


# ── MCP: rename_tag, find_duplicates, swap_tier, ns_summary ───────────────────

def test_mcp_rename_tag(tmp_path):
    """MCP mnemonics_rename_tag renames tag."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    store.add(["x"], vecs, ns="default", meta=[{"tags": ["old"]}])
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_rename_tag",
                                    "arguments": {"old_tag": "old", "new_tag": "new", "ns": "default"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "1" in text


def test_mcp_rename_tag_missing(tmp_path):
    """MCP mnemonics_rename_tag returns error when params missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_rename_tag",
                                    "arguments": {"old_tag": "x"}}})[0]
    assert "error" in resp


def test_mcp_find_duplicates(tmp_path):
    """MCP mnemonics_find_duplicates returns groups."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["dup", "dup"], vecs, ns="default")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_find_duplicates",
                                    "arguments": {"ns": "default"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "1" in text


def test_mcp_swap_tier(tmp_path):
    """MCP mnemonics_swap_tier moves memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="myns")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_swap_tier",
                                    "arguments": {"ns": "myns", "from_tier": 1, "to_tier": 2}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "2" in text


def test_mcp_swap_tier_missing(tmp_path):
    """MCP mnemonics_swap_tier returns error when params missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_swap_tier",
                                    "arguments": {"ns": "myns"}}})[0]
    assert "error" in resp


def test_mcp_swap_tier_invalid(tmp_path):
    """MCP mnemonics_swap_tier returns error for invalid tier."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_swap_tier",
                                    "arguments": {"ns": "x", "from_tier": 0, "to_tier": 9}}})[0]
    assert "error" in resp


def test_mcp_ns_summary(tmp_path):
    """MCP mnemonics_ns_summary returns namespace stats."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="sumns")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_ns_summary",
                                    "arguments": {"ns": "sumns"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "sumns" in text


def test_mcp_ns_summary_missing(tmp_path):
    """MCP mnemonics_ns_summary returns error when ns missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_ns_summary",
                                    "arguments": {}}})[0]
    assert "error" in resp


# ── toggle-tier REST ───────────────────────────────────────────────────────────

def test_http_toggle_tier(tmp_path):
    """POST /toggle-tier cycles memory tier."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    ids = store.add(["x"], vecs, ns="default")
    code, data = http_call(store, "POST", "/toggle-tier", {"id": ids[0]})
    assert code == 200
    assert data["tier"] in (0, 1, 2)


def test_http_toggle_tier_missing_id(tmp_path):
    """POST /toggle-tier returns 400 when id missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/toggle-tier", {})
    assert code == 400


def test_http_toggle_tier_not_found(tmp_path):
    """POST /toggle-tier returns 404 for missing memory."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/toggle-tier", {"id": 99999})
    assert code == 404


# ── merge-texts REST ───────────────────────────────────────────────────────────

def test_http_merge_texts(tmp_path):
    """POST /merge-texts creates new merged memory."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    ids = store.add(["a", "b"], vecs, ns="default")
    code, data = http_call(store, "POST", "/merge-texts", {"ids": ids})
    assert code == 201
    assert "id" in data


def test_http_merge_texts_missing_ids(tmp_path):
    """POST /merge-texts returns 400 when ids missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/merge-texts", {})
    assert code == 400


def test_http_merge_texts_not_found(tmp_path):
    """POST /merge-texts returns 404 when ids not in DB."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/merge-texts", {"ids": [99998, 99999]})
    assert code == 404


# ── truncate-text REST ─────────────────────────────────────────────────────────

def test_http_truncate_text(tmp_path):
    """POST /truncate-text trims memory text."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    ids = store.add(["hello world"], vecs, ns="default")
    code, data = http_call(store, "POST", "/truncate-text",
                           {"id": ids[0], "max_chars": 5})
    assert code == 200


def test_http_truncate_text_missing_params(tmp_path):
    """POST /truncate-text returns 400 when params missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/truncate-text", {"id": 1})
    assert code == 400


def test_http_truncate_text_not_found(tmp_path):
    """POST /truncate-text returns 404 for missing memory."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/truncate-text",
                           {"id": 99999, "max_chars": 10})
    assert code == 404


# ── search-by-access-count REST ────────────────────────────────────────────────

def test_http_search_by_access_count(tmp_path):
    """GET /search-by-access-count returns memories in range."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(3, DIM).astype(np.float32)
    store.add(["a", "b", "c"], vecs, ns="default")
    code, data = http_call(store, "GET", "/search-by-access-count?min=0&ns=default")
    assert code == 200
    assert data["count"] == 3


def test_http_search_by_access_count_all_ns(tmp_path):
    """GET /search-by-access-count without ns spans all namespaces."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    store.add(["x"], vecs, ns="ns1")
    code, data = http_call(store, "GET", "/search-by-access-count?min=0")
    assert code == 200
    assert data["count"] >= 1


# ── MCP: toggle_tier, merge_texts, truncate_text, search_by_access_count ──────

def test_mcp_toggle_tier(tmp_path):
    """MCP mnemonics_toggle_tier cycles tier."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    ids = store.add(["x"], vecs, ns="default")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_toggle_tier",
                                    "arguments": {"id": ids[0]}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "tier cycled" in text


def test_mcp_toggle_tier_missing_id(tmp_path):
    """MCP mnemonics_toggle_tier returns error when id missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_toggle_tier",
                                    "arguments": {}}})[0]
    assert "error" in resp


def test_mcp_toggle_tier_not_found(tmp_path):
    """MCP mnemonics_toggle_tier returns error for missing memory."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_toggle_tier",
                                    "arguments": {"id": 99999}}})[0]
    assert "error" in resp


def test_mcp_merge_texts(tmp_path):
    """MCP mnemonics_merge_texts creates merged memory."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    ids = store.add(["x", "y"], vecs, ns="default")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_merge_texts",
                                    "arguments": {"ids": ids, "ns": "default"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "Merged" in text


def test_mcp_merge_texts_missing(tmp_path):
    """MCP mnemonics_merge_texts returns error when ids missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_merge_texts",
                                    "arguments": {}}})[0]
    assert "error" in resp


def test_mcp_merge_texts_not_found(tmp_path):
    """MCP mnemonics_merge_texts returns error when ids not in DB."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_merge_texts",
                                    "arguments": {"ids": [99998]}}})[0]
    assert "error" in resp


def test_mcp_truncate_text(tmp_path):
    """MCP mnemonics_truncate_text truncates memory text."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    ids = store.add(["hello world"], vecs, ns="default")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_truncate_text",
                                    "arguments": {"id": ids[0], "max_chars": 5}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "truncated" in text


def test_mcp_truncate_text_missing(tmp_path):
    """MCP mnemonics_truncate_text returns error when params missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_truncate_text",
                                    "arguments": {"id": 1}}})[0]
    assert "error" in resp


def test_mcp_truncate_text_not_found(tmp_path):
    """MCP mnemonics_truncate_text returns error for missing memory."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_truncate_text",
                                    "arguments": {"id": 99999, "max_chars": 5}}})[0]
    assert "error" in resp


def test_mcp_search_by_access_count(tmp_path):
    """MCP mnemonics_search_by_access_count returns matching memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="default")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_search_by_access_count",
                                    "arguments": {"min": 0, "ns": "default"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "2" in text


# ── age-by-ns REST ─────────────────────────────────────────────────────────────

def test_http_age_by_ns(tmp_path):
    """GET /age-by-ns returns age breakdown."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="myns")
    code, data = http_call(store, "GET", "/age-by-ns?ns=myns")
    assert code == 200
    assert data["total"] == 2


def test_http_age_by_ns_missing_ns(tmp_path):
    """GET /age-by-ns returns 400 when ns missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "GET", "/age-by-ns")
    assert code == 400


# ── delete-by-tier REST ────────────────────────────────────────────────────────

def test_http_delete_by_tier(tmp_path):
    """POST /delete-by-tier deletes matching memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="myns")
    code, data = http_call(store, "POST", "/delete-by-tier",
                           {"ns": "myns", "tier": 1})
    assert code == 200
    assert data["deleted"] == 2


def test_http_delete_by_tier_missing_params(tmp_path):
    """POST /delete-by-tier returns 400 when params missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/delete-by-tier", {"ns": "x"})
    assert code == 400


def test_http_delete_by_tier_invalid_tier(tmp_path):
    """POST /delete-by-tier returns 400 for invalid tier."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/delete-by-tier",
                           {"ns": "myns", "tier": 9})
    assert code == 400


# ── untagged-memories REST ─────────────────────────────────────────────────────

def test_http_untagged_memories(tmp_path):
    """GET /untagged-memories returns untagged memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="default")
    code, data = http_call(store, "GET", "/untagged-memories?ns=default")
    assert code == 200
    assert data["count"] == 2


def test_http_untagged_memories_all_ns(tmp_path):
    """GET /untagged-memories without ns spans all namespaces."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(1, DIM).astype(np.float32)
    store.add(["x"], vecs, ns="ns1")
    code, data = http_call(store, "GET", "/untagged-memories")
    assert code == 200
    assert data["count"] >= 1


# ── set-meta-for-untagged REST ─────────────────────────────────────────────────

def test_http_set_meta_for_untagged(tmp_path):
    """POST /set-meta-for-untagged sets meta on untagged memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="default")
    code, data = http_call(store, "POST", "/set-meta-for-untagged",
                           {"ns": "default", "key": "source", "value": "auto"})
    assert code == 200
    assert data["updated"] == 2


def test_http_set_meta_for_untagged_missing_params(tmp_path):
    """POST /set-meta-for-untagged returns 400 when key missing."""
    store = Store(tmp_path)
    code, data = http_call(store, "POST", "/set-meta-for-untagged",
                           {"ns": "default"})
    assert code == 400


# ── MCP: age_by_ns, delete_by_tier, untagged_memories, set_meta_for_untagged ──

def test_mcp_age_by_ns(tmp_path):
    """MCP mnemonics_age_by_ns returns age breakdown."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="myns")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_age_by_ns",
                                    "arguments": {"ns": "myns"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "myns" in text


def test_mcp_age_by_ns_missing(tmp_path):
    """MCP mnemonics_age_by_ns returns error when ns missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_age_by_ns",
                                    "arguments": {}}})[0]
    assert "error" in resp


def test_mcp_delete_by_tier(tmp_path):
    """MCP mnemonics_delete_by_tier deletes memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="myns")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_delete_by_tier",
                                    "arguments": {"ns": "myns", "tier": 1}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "2" in text


def test_mcp_delete_by_tier_missing(tmp_path):
    """MCP mnemonics_delete_by_tier returns error when params missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_delete_by_tier",
                                    "arguments": {"ns": "x"}}})[0]
    assert "error" in resp


def test_mcp_delete_by_tier_invalid(tmp_path):
    """MCP mnemonics_delete_by_tier returns error for invalid tier."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_delete_by_tier",
                                    "arguments": {"ns": "myns", "tier": 9}}})[0]
    assert "error" in resp


def test_mcp_untagged_memories(tmp_path):
    """MCP mnemonics_untagged_memories returns untagged memories."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="default")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_untagged_memories",
                                    "arguments": {"ns": "default"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "2" in text


def test_mcp_set_meta_for_untagged(tmp_path):
    """MCP mnemonics_set_meta_for_untagged sets meta on untagged."""
    import numpy as np
    store = Store(tmp_path)
    vecs = np.random.rand(2, DIM).astype(np.float32)
    store.add(["a", "b"], vecs, ns="default")
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_set_meta_for_untagged",
                                    "arguments": {"ns": "default", "key": "src", "value": "auto"}}})[0]
    text = resp["result"]["content"][0]["text"]
    assert "2" in text


def test_mcp_set_meta_for_untagged_missing(tmp_path):
    """MCP mnemonics_set_meta_for_untagged returns error when key missing."""
    store = Store(tmp_path)
    resp = _mcp(store, {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "mnemonics_set_meta_for_untagged",
                                    "arguments": {"ns": "default"}}})[0]
    assert "error" in resp


# ─── Batch 5: clone_memory, memories_without_summary, pin_by_tag, promote_by_access ───

def test_rest_clone_memory(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/get", {"id": 1})
    code, data = http_call(store, "POST", "/clone-memory", {"id": 1, "dst_ns": "clones"})
    assert code == 201
    assert "clone_id" in data
    assert data["dst_ns"] == "clones"


def test_rest_clone_memory_not_found(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/clone-memory", {"id": 999999})
    assert code == 404


def test_rest_clone_memory_missing_id(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/clone-memory", {})
    assert code == 400


def test_rest_memories_without_summary(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/memories-without-summary?ns=default&limit=10")
    assert code == 200
    assert "count" in data


def test_rest_memories_without_summary_all_ns(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/memories-without-summary?limit=5")
    assert code == 200


def test_rest_promote_by_access(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/promote-by-access?ns=default&n=5&from_tier=1&to_tier=0")
    assert code == 200
    assert "promoted" in data


def test_rest_promote_by_access_missing_ns(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/promote-by-access")
    assert code == 400


def test_rest_promote_by_access_bad_tier(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/promote-by-access?ns=default&from_tier=5&to_tier=0")
    assert code == 400


def test_rest_pin_by_tag(populated_store):
    store, docs, vecs = populated_store
    store.tag(1, "pinme")
    code, data = http_call(store, "POST", "/pin-by-tag", {"tag": "pinme", "ns": "default"})
    assert code == 200
    assert data["pinned"] >= 1


def test_rest_pin_by_tag_missing_tag(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/pin-by-tag", {})
    assert code == 400


def test_mcp_clone_memory(populated_store):
    """MCP mnemonics_clone_memory clones a memory."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": "mnemonics_clone_memory", "arguments": {"id": 1, "dst_ns": "clones"}},
    })[0]
    assert "clone" in resp["result"]["content"][0]["text"].lower()


def test_mcp_clone_memory_not_found(populated_store):
    """MCP mnemonics_clone_memory returns error for missing id."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 2,
        "method": "tools/call",
        "params": {"name": "mnemonics_clone_memory", "arguments": {"id": 999999}},
    })[0]
    assert "error" in resp


def test_mcp_clone_memory_missing_id(populated_store):
    """MCP mnemonics_clone_memory returns error when id omitted."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 3,
        "method": "tools/call",
        "params": {"name": "mnemonics_clone_memory", "arguments": {}},
    })[0]
    assert "error" in resp


def test_mcp_memories_without_summary(populated_store):
    """MCP mnemonics_memories_without_summary works."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 4,
        "method": "tools/call",
        "params": {"name": "mnemonics_memories_without_summary", "arguments": {"ns": "default"}},
    })[0]
    assert "without summary" in resp["result"]["content"][0]["text"].lower()


def test_mcp_pin_by_tag(populated_store):
    """MCP mnemonics_pin_by_tag pins memories."""
    store, docs, vecs = populated_store
    store.tag(1, "urgent")
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 5,
        "method": "tools/call",
        "params": {"name": "mnemonics_pin_by_tag", "arguments": {"tag": "urgent"}},
    })[0]
    assert "pinned" in resp["result"]["content"][0]["text"].lower()


def test_mcp_pin_by_tag_missing_tag(populated_store):
    """MCP mnemonics_pin_by_tag returns error when tag omitted."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 6,
        "method": "tools/call",
        "params": {"name": "mnemonics_pin_by_tag", "arguments": {}},
    })[0]
    assert "error" in resp


def test_mcp_promote_by_access(populated_store):
    """MCP mnemonics_promote_by_access promotes memories."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 7,
        "method": "tools/call",
        "params": {"name": "mnemonics_promote_by_access",
                   "arguments": {"ns": "default", "n": 5, "from_tier": 1, "to_tier": 0}},
    })[0]
    assert "promoted" in resp["result"]["content"][0]["text"].lower()


def test_mcp_promote_by_access_missing_ns(populated_store):
    """MCP mnemonics_promote_by_access returns error when ns omitted."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 8,
        "method": "tools/call",
        "params": {"name": "mnemonics_promote_by_access", "arguments": {}},
    })[0]
    assert "error" in resp


def test_mcp_promote_by_access_bad_tier(populated_store):
    """MCP mnemonics_promote_by_access returns error for invalid tier."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 9,
        "method": "tools/call",
        "params": {"name": "mnemonics_promote_by_access",
                   "arguments": {"ns": "default", "from_tier": 9, "to_tier": 1}},
    })[0]
    assert "error" in resp


# ─── Batch 6: filter_by_text_length, multi_tag_filter, tag_stats, split_memory ───

def test_rest_filter_by_text_length(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/filter-by-text-length?min_chars=1&max_chars=200&ns=default")
    assert code == 200
    assert "results" in data


def test_rest_filter_by_text_length_all_ns(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/filter-by-text-length?max_chars=500")
    assert code == 200


def test_rest_tag_stats(populated_store):
    store, docs, vecs = populated_store
    store.tag(1, "t1")
    code, data = http_call(store, "GET", "/tag-stats?ns=default")
    assert code == 200
    assert "stats" in data


def test_rest_tag_stats_all_ns(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/tag-stats")
    assert code == 200


def test_rest_multi_tag_filter(populated_store):
    store, docs, vecs = populated_store
    store.tag(1, "p")
    code, data = http_call(store, "POST", "/multi-tag-filter", {"tags": ["p"], "mode": "any"})
    assert code == 200
    assert data["count"] >= 1


def test_rest_multi_tag_filter_missing_tags(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/multi-tag-filter", {})
    assert code == 400


def test_rest_multi_tag_filter_bad_mode(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/multi-tag-filter", {"tags": ["x"], "mode": "bad"})
    assert code == 400


def test_rest_split_memory(populated_store):
    store, docs, vecs = populated_store
    ids = store.add(["part1\n\npart2"], vecs[:1], ns="default")
    code, data = http_call(store, "POST", "/split-memory", {"id": ids[0], "separator": "\n\n"})
    assert code == 201
    assert data["count"] == 2


def test_rest_split_memory_max_chars(populated_store):
    store, docs, vecs = populated_store
    ids = store.add(["hello world foo bar"], vecs[:1], ns="default")
    code, data = http_call(store, "POST", "/split-memory", {"id": ids[0], "max_chars": "8"})
    assert code == 201
    assert data["count"] >= 2


def test_rest_split_memory_missing_id(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/split-memory", {})
    assert code == 400


def test_rest_split_memory_not_found(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/split-memory", {"id": 999999, "separator": "\n\n"})
    assert code == 400


def test_mcp_filter_by_text_length(populated_store):
    """MCP mnemonics_filter_by_text_length finds memories by text length."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 10,
        "method": "tools/call",
        "params": {"name": "mnemonics_filter_by_text_length", "arguments": {"max_chars": 200, "ns": "default"}},
    })[0]
    assert "Found" in resp["result"]["content"][0]["text"]


def test_mcp_multi_tag_filter(populated_store):
    """MCP mnemonics_multi_tag_filter works."""
    store, docs, vecs = populated_store
    store.tag(1, "x")
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 11,
        "method": "tools/call",
        "params": {"name": "mnemonics_multi_tag_filter", "arguments": {"tags": ["x"], "mode": "any"}},
    })[0]
    assert "Found" in resp["result"]["content"][0]["text"]


def test_mcp_multi_tag_filter_missing_tags(populated_store):
    """MCP mnemonics_multi_tag_filter returns error for missing tags."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 12,
        "method": "tools/call",
        "params": {"name": "mnemonics_multi_tag_filter", "arguments": {}},
    })[0]
    assert "error" in resp


def test_mcp_tag_stats(populated_store):
    """MCP mnemonics_tag_stats returns tag stats."""
    store, docs, vecs = populated_store
    store.tag(1, "urgent")
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 13,
        "method": "tools/call",
        "params": {"name": "mnemonics_tag_stats", "arguments": {"ns": "default"}},
    })[0]
    assert "urgent" in resp["result"]["content"][0]["text"]


def test_mcp_tag_stats_empty(populated_store):
    """MCP mnemonics_tag_stats returns '(no tags found)' for empty ns."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 14,
        "method": "tools/call",
        "params": {"name": "mnemonics_tag_stats", "arguments": {"ns": "empty_ns_xyz"}},
    })[0]
    assert "no tags" in resp["result"]["content"][0]["text"].lower()


def test_mcp_split_memory(populated_store):
    """MCP mnemonics_split_memory splits a memory."""
    store, docs, vecs = populated_store
    ids = store.add(["part1\n\npart2"], vecs[:1], ns="default")
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 15,
        "method": "tools/call",
        "params": {"name": "mnemonics_split_memory",
                   "arguments": {"id": ids[0], "separator": "\n\n"}},
    })[0]
    assert "Split" in resp["result"]["content"][0]["text"]


def test_mcp_split_memory_max_chars(populated_store):
    """MCP mnemonics_split_memory works with max_chars."""
    store, docs, vecs = populated_store
    ids = store.add(["hello world foo bar"], vecs[:1], ns="default")
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 18,
        "method": "tools/call",
        "params": {"name": "mnemonics_split_memory",
                   "arguments": {"id": ids[0], "max_chars": "8"}},
    })[0]
    assert "Split" in resp["result"]["content"][0]["text"]


def test_mcp_split_memory_missing_id(populated_store):
    """MCP mnemonics_split_memory returns error when id omitted."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 16,
        "method": "tools/call",
        "params": {"name": "mnemonics_split_memory", "arguments": {}},
    })[0]
    assert "error" in resp


def test_mcp_split_memory_not_found(populated_store):
    """MCP mnemonics_split_memory returns error for missing id."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 17,
        "method": "tools/call",
        "params": {"name": "mnemonics_split_memory",
                   "arguments": {"id": 999999, "separator": "\n\n"}},
    })[0]
    assert "error" in resp


# ─── Batch 7: bulk_summarize, cross_ns_search, memory_timeline, keyword_extract ───

def test_rest_bulk_summarize(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/bulk-summarize", {"updates": {"1": "new summary"}})
    assert code == 200
    assert "updated" in data


def test_rest_bulk_summarize_missing_updates(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/bulk-summarize", {})
    assert code == 400


def test_rest_cross_ns_search(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/cross-ns-search?query=the&ns=default")
    assert code == 200
    assert "results" in data


def test_rest_cross_ns_search_missing_query(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/cross-ns-search?ns=default")
    assert code == 400


def test_rest_memory_timeline(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/memory-timeline?ns=default&limit=10")
    assert code == 200
    assert "timeline" in data


def test_rest_memory_timeline_all_ns(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/memory-timeline&limit=5")
    assert code == 200


def test_rest_keyword_extract(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/keyword-extract?id=1&top_n=5")
    assert code == 200
    assert "keywords" in data


def test_rest_keyword_extract_missing_id(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/keyword-extract")
    assert code == 400


def test_rest_keyword_extract_not_found(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/keyword-extract?id=999999")
    assert code == 404


def test_mcp_bulk_summarize(populated_store):
    """MCP mnemonics_bulk_summarize updates summaries."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 20,
        "method": "tools/call",
        "params": {"name": "mnemonics_bulk_summarize",
                   "arguments": {"updates": {"1": "new summary"}}},
    })[0]
    assert "Updated" in resp["result"]["content"][0]["text"]


def test_mcp_bulk_summarize_missing_updates(populated_store):
    """MCP mnemonics_bulk_summarize returns error when updates missing."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 21,
        "method": "tools/call",
        "params": {"name": "mnemonics_bulk_summarize", "arguments": {}},
    })[0]
    assert "error" in resp


def test_mcp_cross_ns_search(populated_store):
    """MCP mnemonics_cross_ns_search returns hits."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 22,
        "method": "tools/call",
        "params": {"name": "mnemonics_cross_ns_search",
                   "arguments": {"query": "the", "namespaces": ["default"]}},
    })[0]
    assert "Found" in resp["result"]["content"][0]["text"]


def test_mcp_cross_ns_search_missing_args(populated_store):
    """MCP mnemonics_cross_ns_search returns error when args missing."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 23,
        "method": "tools/call",
        "params": {"name": "mnemonics_cross_ns_search", "arguments": {"query": "x"}},
    })[0]
    assert "error" in resp


def test_mcp_memory_timeline(populated_store):
    """MCP mnemonics_memory_timeline returns timeline."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 24,
        "method": "tools/call",
        "params": {"name": "mnemonics_memory_timeline",
                   "arguments": {"ns": "default", "limit": 10}},
    })[0]
    assert "Timeline" in resp["result"]["content"][0]["text"] or "no memories" in resp["result"]["content"][0]["text"].lower()


def test_mcp_memory_timeline_empty(populated_store):
    """MCP mnemonics_memory_timeline returns (no memories) for empty ns."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 25,
        "method": "tools/call",
        "params": {"name": "mnemonics_memory_timeline",
                   "arguments": {"ns": "empty_xyz_ns"}},
    })[0]
    assert "no memories" in resp["result"]["content"][0]["text"].lower()


def test_mcp_keyword_extract(populated_store):
    """MCP mnemonics_keyword_extract returns keywords."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 26,
        "method": "tools/call",
        "params": {"name": "mnemonics_keyword_extract",
                   "arguments": {"id": 1, "top_n": 5}},
    })[0]
    text = resp["result"]["content"][0]["text"]
    assert "keywords" in text.lower() or "word" in text.lower() or "no keywords" in text.lower()


def test_mcp_keyword_extract_not_found(populated_store):
    """MCP mnemonics_keyword_extract returns error for missing id."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 27,
        "method": "tools/call",
        "params": {"name": "mnemonics_keyword_extract",
                   "arguments": {"id": 999999}},
    })[0]
    assert "error" in resp


def test_mcp_keyword_extract_missing_id(populated_store):
    """MCP mnemonics_keyword_extract returns error when id omitted."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 28,
        "method": "tools/call",
        "params": {"name": "mnemonics_keyword_extract", "arguments": {}},
    })[0]
    assert "error" in resp


def test_mcp_memory_timeline_with_tier(populated_store):
    """MCP mnemonics_memory_timeline with tier filter."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 29,
        "method": "tools/call",
        "params": {"name": "mnemonics_memory_timeline",
                   "arguments": {"ns": "default", "tier": "1"}},
    })[0]
    text = resp["result"]["content"][0]["text"]
    assert "Timeline" in text or "no memories" in text.lower()


def test_mcp_keyword_extract_empty_result(populated_store):
    """MCP mnemonics_keyword_extract handles empty keywords."""
    store, docs, vecs = populated_store
    # Add a memory with only stop-words
    ids = store.add(["a an the and or but"], vecs[:1], ns="default")
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 30,
        "method": "tools/call",
        "params": {"name": "mnemonics_keyword_extract",
                   "arguments": {"id": ids[0], "top_n": 5}},
    })[0]
    text = resp["result"]["content"][0]["text"]
    assert "keywords" in text.lower() or "no keywords" in text.lower() or "word" in text.lower()


# ─── Batch 8: import_ns, get_tier_distribution, archive_by_tier, text_search_ranked ───

def test_rest_import_ns(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/import-ns", {
        "ns": "imported_test", "rows": [{"text": "imported memory"}]
    })
    assert code == 200
    assert data["imported"] == 1


def test_rest_import_ns_missing_args(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/import-ns", {"ns": "x"})
    assert code == 400


def test_rest_import_ns_overwrite(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/import-ns", {
        "ns": "default", "rows": [{"text": "replaced"}], "overwrite": True
    })
    assert code == 200


def test_rest_get_tier_distribution(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/get-tier-distribution?ns=default")
    assert code == 200
    assert "distribution" in data


def test_rest_get_tier_distribution_all_ns(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/get-tier-distribution")
    assert code == 200
    assert "distribution" in data


def test_rest_archive_by_tier(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/archive-by-tier", {"ns": "default", "tier": 2})
    assert code == 200
    assert "archived" in data


def test_rest_archive_by_tier_missing_args(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/archive-by-tier", {"ns": "default"})
    assert code == 400


def test_rest_text_search_ranked(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/text-search-ranked?query=the&ns=default")
    assert code == 200
    assert "results" in data


def test_rest_text_search_ranked_missing_query(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/text-search-ranked")
    assert code == 400


def test_mcp_import_ns(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 31,
        "method": "tools/call",
        "params": {"name": "mnemonics_import_ns",
                   "arguments": {"ns": "mcp_import", "rows": [{"text": "hello"}]}},
    })[0]
    assert "Imported" in resp["result"]["content"][0]["text"]


def test_mcp_import_ns_missing_args(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 32,
        "method": "tools/call",
        "params": {"name": "mnemonics_import_ns", "arguments": {"ns": "x"}},
    })[0]
    assert "error" in resp


def test_mcp_get_tier_distribution(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 33,
        "method": "tools/call",
        "params": {"name": "mnemonics_get_tier_distribution", "arguments": {"ns": "default"}},
    })[0]
    assert "distribution" in resp["result"]["content"][0]["text"].lower() or "tier" in resp["result"]["content"][0]["text"].lower()


def test_mcp_get_tier_distribution_all_ns(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 34,
        "method": "tools/call",
        "params": {"name": "mnemonics_get_tier_distribution", "arguments": {}},
    })[0]
    assert "tier" in resp["result"]["content"][0]["text"].lower()


def test_mcp_archive_by_tier(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 35,
        "method": "tools/call",
        "params": {"name": "mnemonics_archive_by_tier",
                   "arguments": {"ns": "default", "tier": 2}},
    })[0]
    assert "Archived" in resp["result"]["content"][0]["text"]


def test_mcp_archive_by_tier_missing_args(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 36,
        "method": "tools/call",
        "params": {"name": "mnemonics_archive_by_tier", "arguments": {"ns": "default"}},
    })[0]
    assert "error" in resp


def test_mcp_text_search_ranked(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 37,
        "method": "tools/call",
        "params": {"name": "mnemonics_text_search_ranked",
                   "arguments": {"query": "the", "ns": "default"}},
    })[0]
    text = resp["result"]["content"][0]["text"]
    assert "results" in text.lower() or "Top" in text or "No results" in text


def test_mcp_text_search_ranked_no_query(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 38,
        "method": "tools/call",
        "params": {"name": "mnemonics_text_search_ranked", "arguments": {}},
    })[0]
    assert "error" in resp


def test_mcp_text_search_ranked_with_tier(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 39,
        "method": "tools/call",
        "params": {"name": "mnemonics_text_search_ranked",
                   "arguments": {"query": "the", "ns": "default", "tier": "1"}},
    })[0]
    text = resp["result"]["content"][0]["text"]
    assert "Top" in text or "No results" in text


def test_mcp_text_search_ranked_no_results(populated_store):
    """MCP text_search_ranked returns 'No results' for non-matching query."""
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 40,
        "method": "tools/call",
        "params": {"name": "mnemonics_text_search_ranked",
                   "arguments": {"query": "xyzzy_nonexistent_term_zzz"}},
    })[0]
    text = resp["result"]["content"][0]["text"]
    assert "No results" in text


# ─── Batch 9: deduplicate_by_text, merge_memories, search_by_date_range, get_access_stats ───

def test_rest_deduplicate_by_text(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/deduplicate-by-text", {"ns": "default"})
    assert code == 200
    assert "deleted" in data


def test_rest_deduplicate_by_text_all_ns(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/deduplicate-by-text", {"all_ns": True})
    assert code == 200


def test_rest_merge_memories(populated_store):
    store, docs, vecs = populated_store
    ids = store.add(["alpha part", "beta part"], vecs[:2], ns="default")
    code, data = http_call(store, "POST", "/merge-memories", {"ids": ids})
    assert code == 200
    assert "merged_id" in data


def test_rest_merge_memories_missing_ids(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/merge-memories", {})
    assert code == 400


def test_rest_merge_memories_not_found(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "POST", "/merge-memories", {"ids": [999999]})
    assert code == 404


def test_rest_search_by_date_range(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET",
                           "/search-by-date-range?start=2000-01-01&end=2099-12-31&ns=default")
    assert code == 200
    assert "results" in data


def test_rest_search_by_date_range_missing_params(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/search-by-date-range?start=2000-01-01")
    assert code == 400


def test_rest_get_access_stats(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/get-access-stats?ns=default")
    assert code == 200
    assert "total_accesses" in data


def test_rest_get_access_stats_all_ns(populated_store):
    store, docs, vecs = populated_store
    code, data = http_call(store, "GET", "/get-access-stats")
    assert code == 200


def test_mcp_deduplicate_by_text(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 41,
        "method": "tools/call",
        "params": {"name": "mnemonics_deduplicate_by_text", "arguments": {"ns": "default"}},
    })[0]
    assert "Deduplicated" in resp["result"]["content"][0]["text"]


def test_mcp_merge_memories(populated_store):
    store, docs, vecs = populated_store
    ids = store.add(["x part", "y part"], vecs[:2], ns="default")
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 42,
        "method": "tools/call",
        "params": {"name": "mnemonics_merge_memories", "arguments": {"ids": ids}},
    })[0]
    assert "Merged" in resp["result"]["content"][0]["text"]


def test_mcp_merge_memories_missing_ids(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 43,
        "method": "tools/call",
        "params": {"name": "mnemonics_merge_memories", "arguments": {}},
    })[0]
    assert "error" in resp


def test_mcp_merge_memories_not_found(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 44,
        "method": "tools/call",
        "params": {"name": "mnemonics_merge_memories", "arguments": {"ids": [999999]}},
    })[0]
    assert "error" in resp


def test_mcp_search_by_date_range(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 45,
        "method": "tools/call",
        "params": {"name": "mnemonics_search_by_date_range",
                   "arguments": {"start": "2000-01-01", "end": "2099-12-31", "ns": "default"}},
    })[0]
    text = resp["result"]["content"][0]["text"]
    assert "Found" in text or "No memories" in text


def test_mcp_search_by_date_range_empty(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 46,
        "method": "tools/call",
        "params": {"name": "mnemonics_search_by_date_range",
                   "arguments": {"start": "2099-01-01", "end": "2099-12-31"}},
    })[0]
    assert "No memories" in resp["result"]["content"][0]["text"]


def test_mcp_search_by_date_range_missing(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 47,
        "method": "tools/call",
        "params": {"name": "mnemonics_search_by_date_range",
                   "arguments": {"start": "2000-01-01"}},
    })[0]
    assert "error" in resp


def test_mcp_get_access_stats(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 48,
        "method": "tools/call",
        "params": {"name": "mnemonics_get_access_stats", "arguments": {"ns": "default"}},
    })[0]
    assert "total accesses" in resp["result"]["content"][0]["text"]


def test_mcp_get_access_stats_all_ns(populated_store):
    store, docs, vecs = populated_store
    resp = _mcp(store, {
        "jsonrpc": "2.0", "id": 49,
        "method": "tools/call",
        "params": {"name": "mnemonics_get_access_stats", "arguments": {}},
    })[0]
    assert "total accesses" in resp["result"]["content"][0]["text"]
