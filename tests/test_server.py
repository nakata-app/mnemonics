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
