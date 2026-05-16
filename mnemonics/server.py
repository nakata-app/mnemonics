"""MCP + REST server for mnemonics.

Endpoints:
  GET  /health
  POST /ingest   {"texts": [...], "ns": "default", "meta": [...]}
  POST /retrieve {"query": "...", "ns": "default", "top_k": 5}
  GET  /namespaces
  GET  /count?ns=default
  DELETE /memory/<id>

MCP tools (JSON-RPC over stdio):
  mnemonics_ingest   — store memories
  mnemonics_retrieve — semantic search
  mnemonics_forget   — delete a memory by id
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from pathlib import Path

from mnemonics.store import Store
from mnemonics.ingest import ingest as _ingest
from mnemonics.retrieve import retrieve as _retrieve

MNEMONICS_PORT = int(os.environ.get("MNEMONICS_PORT", "7810"))
MNEMONICS_PATH = os.environ.get("MNEMONICS_PATH", "~/.mnemonics")

_store: Store | None = None


def _get_store() -> Store:
    global _store
    if _store is None:
        _store = Store(MNEMONICS_PATH)
    return _store


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _json(self, code: int, data: Any) -> None:
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/health":
            self._json(200, {"status": "ok", "version": "0.2.1"})
        elif path == "/namespaces":
            self._json(200, {"namespaces": _get_store().list_namespaces()})
        elif path == "/count":
            ns = (self.path.split("ns=")[-1] if "ns=" in self.path else "default")
            self._json(200, {"ns": ns, "count": _get_store().count(ns)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            body = self._body()
        except Exception:
            self._json(400, {"error": "invalid JSON"})
            return

        if self.path == "/ingest":
            texts = body.get("texts", [])
            if not isinstance(texts, list):
                self._json(400, {"error": "texts must be an array of strings"})
                return
            if not texts:
                self._json(400, {"error": "texts must not be empty"})
                return
            n = _ingest(
                texts=texts,
                store=_get_store(),
                ns=body.get("ns", "default"),
                meta=body.get("meta"),
            )
            self._json(200, {"ingested": n})

        elif self.path == "/retrieve":
            query = body.get("query", "").strip()
            if not query:
                self._json(400, {"error": "query must not be empty"})
                return
            result = _retrieve(
                query=query,
                store=_get_store(),
                ns=body.get("ns", "default"),
                top_k=int(body.get("top_k", 5)),
                decay=bool(body.get("decay", True)),
            )
            self._json(200, result)

        else:
            self._json(404, {"error": "not found"})

    def do_DELETE(self) -> None:
        if self.path.startswith("/memory/"):
            try:
                mid = int(self.path.split("/memory/")[1])
                ok = _get_store().delete(mid)
                self._json(200, {"deleted": ok})
            except ValueError:
                self._json(400, {"error": "invalid id"})
        else:
            self._json(404, {"error": "not found"})


# ── MCP stdio mode ───────────────────────────────────────────────────────────

def _mcp_loop() -> None:
    """JSON-RPC over stdin/stdout for MCP clients (Claude Code, Cursor, Metis)."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue

        rid = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {})

        def ok(result: Any) -> None:
            print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}), flush=True)

        def err(msg: str, code: int = -32600) -> None:
            print(json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}), flush=True)

        if method == "initialize":
            ok({
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "mnemonics", "version": "0.2.1"},
                "capabilities": {"tools": {}},
            })

        elif method == "tools/list":
            ok({"tools": [
                {
                    "name": "mnemonics_ingest",
                    "description": "Store text memories into mnemonics. Chunks, embeds and persists.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "texts": {"type": "array", "items": {"type": "string"}},
                            "ns": {"type": "string", "description": "Namespace (default: 'default')"},
                        },
                        "required": ["texts"],
                    },
                },
                {
                    "name": "mnemonics_retrieve",
                    "description": "Semantic search with tier-aware decay. Pinned (tier 0) memories never decay; tier 1 has 90-day half-life, tier 2 has 14-day. Set decay=false to see raw cosine scores.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "ns": {"type": "string"},
                            "top_k": {"type": "integer"},
                            "decay": {"type": "boolean", "description": "Apply decay scoring (default true)"},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "mnemonics_forget",
                    "description": "Delete a memory by id.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                        "required": ["id"],
                    },
                },
                {
                    "name": "mnemonics_pin",
                    "description": "Pin a memory (tier=0, never decays). Use for decisions, key facts.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                        "required": ["id"],
                    },
                },
                {
                    "name": "mnemonics_tier",
                    "description": "Set memory tier: 0=pinned, 1=default, 2=ambient (fast decay).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "tier": {"type": "integer", "enum": [0, 1, 2]},
                        },
                        "required": ["id", "tier"],
                    },
                },
                {
                    "name": "mnemonics_gc",
                    "description": "Garbage-collect ambient (tier 2) memories never accessed and older than age_days. Default dry_run=true returns candidates only.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Limit to one namespace (default: all)"},
                            "age_days": {"type": "integer", "description": "Minimum age in days (default: 30)"},
                            "dry_run": {"type": "boolean", "description": "If true, only list candidates (default: true)"},
                        },
                    },
                },
                {
                    "name": "mnemonics_stats",
                    "description": "List all namespaces with their chunk counts.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ]})

        elif method == "tools/call":
            name = params.get("name")
            args = params.get("arguments", {})

            if name == "mnemonics_ingest":
                texts = args.get("texts", [])
                if not isinstance(texts, list):
                    err("texts must be an array of strings, not a single string")
                    continue
                if not texts:
                    err("texts must not be empty (did you pass 'text' instead of 'texts'?)")
                    continue
                if not all(isinstance(t, str) and t.strip() for t in texts):
                    err("each item in texts must be a non-empty string")
                    continue
                n = _ingest(texts=texts, store=_get_store(), ns=args.get("ns", "default"))
                ok({"content": [{"type": "text", "text": f"Stored {n} chunks."}]})

            elif name == "mnemonics_retrieve":
                result = _retrieve(
                    query=args["query"],
                    store=_get_store(),
                    ns=args.get("ns", "default"),
                    top_k=int(args.get("top_k", 5)),
                    decay=bool(args.get("decay", True)),
                )
                tier_label = {0: "pin", 1: "def", 2: "amb"}
                lines = [
                    f"[{r['score']:.3f}] [raw={r['raw_score']:.3f} decay={r['decay_factor']:.2f} "
                    f"boost={r['boost']:.2f} age={r['age_days']:.0f}d "
                    f"tier={tier_label.get(r['tier'], '?')}] {r['text'][:200]}"
                    for r in result["results"]
                ]
                ok({"content": [{"type": "text", "text": "\n".join(lines)}]})

            elif name == "mnemonics_forget":
                deleted = _get_store().delete(int(args["id"]))
                ok({"content": [{"type": "text", "text": f"Deleted: {deleted}"}]})

            elif name == "mnemonics_pin":
                pinned = _get_store().pin(int(args["id"]))
                ok({"content": [{"type": "text", "text": f"Pinned: {pinned}"}]})

            elif name == "mnemonics_tier":
                changed = _get_store().set_tier(int(args["id"]), int(args["tier"]))
                ok({"content": [{"type": "text", "text": f"Tier set: {changed}"}]})

            elif name == "mnemonics_gc":
                store = _get_store()
                ns = args.get("ns")
                age_days = int(args.get("age_days", 30))
                dry_run = bool(args.get("dry_run", True))
                cands = store.gc_candidates(ns=ns, age_days=age_days)
                if dry_run:
                    text = (
                        f"{len(cands)} candidate(s) (dry-run):\n"
                        + "\n".join(f"  id={c['id']} ns={c['ns']} age={c['age_days']}d" for c in cands[:30])
                    )
                else:
                    n = store.gc(ns=ns, age_days=age_days)
                    text = f"Deleted {n} row(s)."
                ok({"content": [{"type": "text", "text": text}]})

            elif name == "mnemonics_stats":
                store = _get_store()
                lines = [f"  {ns}: {store.count(ns)} chunks" for ns in store.list_namespaces()]
                ok({"content": [{"type": "text", "text": "\n".join(lines) or "(empty)"}]})

            else:
                err(f"unknown tool: {name}")
        else:
            err(f"unknown method: {method}")


def serve(port: int = MNEMONICS_PORT, mcp: bool = False) -> None:
    if mcp:
        _mcp_loop()
        return
    print(f"[mnemonics] listening on 127.0.0.1:{port}", flush=True)
    # Bind to localhost only. Do NOT change to "0.0.0.0" — that would expose
    # the entire memory store to anyone on the local network.
    server = HTTPServer(("127.0.0.1", port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
