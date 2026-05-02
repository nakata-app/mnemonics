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
        elif method == "DELETE":
            handler.do_DELETE()

    return captured["code"], captured["data"]


# ── GET /health ───────────────────────────────────────────────────────────────

def test_health(tmp_store):
    code, data = http_call(tmp_store, "GET", "/health")
    assert code == 200
    assert data["status"] == "ok"


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
    assert names == {"mnemonics_ingest", "mnemonics_retrieve", "mnemonics_forget"}


def test_mcp_ingest(tmp_store):
    with patch("mnemonics.server._ingest", return_value=3) as mock_ingest:
        resp = _mcp(tmp_store, {
            "jsonrpc": "2.0", "id": 3,
            "method": "tools/call",
            "params": {"name": "mnemonics_ingest", "arguments": {"texts": ["a", "b", "c"]}},
        })
    assert "Stored 3" in resp[0]["result"]["content"][0]["text"]


def test_mcp_retrieve(tmp_store):
    fake = {"results": [], "trust_score": 1.0, "flagged_count": 0, "verified": True}
    with patch("mnemonics.server._retrieve", return_value=fake):
        resp = _mcp(tmp_store, {
            "jsonrpc": "2.0", "id": 4,
            "method": "tools/call",
            "params": {"name": "mnemonics_retrieve", "arguments": {"query": "test"}},
        })
    assert "trust_score" in resp[0]["result"]["content"][0]["text"]


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
