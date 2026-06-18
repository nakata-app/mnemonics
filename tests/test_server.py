"""Tests for REST and MCP server logic."""
import io
import json
import threading
from http.client import HTTPConnection
from unittest.mock import MagicMock, patch

import pytest

from mnemonics import server as srv


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
