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
  mnemonics_pin      — pin a memory (tier=0, never decays)
  mnemonics_tier     — set memory tier (0/1/2)
  mnemonics_gc       — garbage-collect old ambient/default memories
  mnemonics_stats    — list namespaces with chunk counts
  mnemonics_health   — store health check (DB integrity, index vs SQL counts)
  mnemonics_repair   — auto-repair orphan vectors and orphan index files
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
            self._json(200, {"status": "ok", "version": "0.3.0"})
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
            summaries = body.get("summaries")
            if summaries is not None and (
                not isinstance(summaries, list)
                or len(summaries) != len(texts)
                or any(s is not None and not isinstance(s, str) for s in summaries)
            ):
                self._json(400, {"error": "summaries must be an array of (string|null), same length as texts"})
                return
            n = _ingest(
                texts=texts,
                store=_get_store(),
                ns=body.get("ns", "default"),
                meta=body.get("meta"),
                summaries=summaries,
            )
            self._json(200, {"ingested": n})

        elif self.path == "/retrieve":
            query = body.get("query", "").strip()
            if not query:
                self._json(400, {"error": "query must not be empty"})
                return
            hybrid = bool(body.get("hybrid", True))
            candidate_k = int(body.get("candidate_k", 50))
            if candidate_k < 1:
                self._json(400, {"error": "candidate_k must be >= 1"})
                return
            try:
                result = _retrieve(
                    query=query,
                    store=_get_store(),
                    ns=body.get("ns", "default"),
                    top_k=int(body.get("top_k", 5)),
                    decay=bool(body.get("decay", True)),
                    hybrid=hybrid,
                    candidate_k=candidate_k,
                    rerank=bool(body.get("rerank", False)),
                )
            except RuntimeError as e:
                self._json(400, {"error": str(e)})
                return
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
                "serverInfo": {"name": "mnemonics", "version": "0.3.0"},
                "capabilities": {"tools": {}},
            })

        elif method == "tools/list":
            ok({"tools": [
                {
                    "name": "mnemonics_ingest",
                    "description": "Store text memories into mnemonics. Chunks, embeds and persists. Pass `texts` as an array of strings, or `text` as a single string (alias). Optional `summaries` parallel to `texts` adds a second keyword surface for BM25 retrieval (e.g. raw transcript + GLM gist); embeddings still come from the raw text.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "texts": {"type": "array", "items": {"type": "string"}, "description": "Array of memory strings to store."},
                            "text": {"type": "string", "description": "Convenience alias for `texts: [text]`. Either `text` or `texts` is required."},
                            "ns": {"type": "string", "description": "Namespace (default: 'default')"},
                            "summaries": {
                                "type": "array",
                                "items": {"type": ["string", "null"]},
                                "description": "Optional, one entry per text. Null entries are stored as no-summary.",
                            },
                        },
                    },
                },
                {
                    "name": "mnemonics_retrieve",
                    "description": "Hybrid semantic + keyword search (vector cosine fused with BM25 via Reciprocal Rank Fusion) with tier-aware decay. Pinned (tier 0) memories never decay; tier 1 has 90-day half-life, tier 2 has 14-day. Set decay=false to see raw scores. Set hybrid=false to fall back to vector-only retrieval (rarely needed; hybrid wins or ties in every measured query class). Set rerank=true to add an AdaptMem cross-encoder rerank stage over the widened candidate band (requires adaptmem installed).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "ns": {"type": "string"},
                            "top_k": {"type": "integer"},
                            "decay": {"type": "boolean", "description": "Apply decay scoring (default true)"},
                            "hybrid": {"type": "boolean", "description": "Fuse vector + BM25 via RRF (default true)"},
                            "candidate_k": {"type": "integer", "description": "Per-channel pool size when hybrid=true (default 50)"},
                            "rerank": {"type": "boolean", "description": "Cross-encoder rerank via AdaptMem over the candidate band (default false)"},
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
                    "description": "Garbage-collect memories older than age_days. Default tier=2 targets ambient (never-accessed) rows; tier=1 also sweeps default-tier rows. dry_run=true (default) returns candidates only.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Limit to one namespace (default: all)"},
                            "age_days": {"type": "integer", "description": "Minimum age in days (default: 30)"},
                            "dry_run": {"type": "boolean", "description": "If true, only list candidates (default: true)"},
                            "tier": {"type": "integer", "enum": [1, 2], "description": "Target tier (1=default, 2=ambient; default: 2)"},
                        },
                    },
                },
                {
                    "name": "mnemonics_health",
                    "description": "Store health check: DB integrity, WAL size, per-namespace SQL vs index count (orphan/missing vectors), and orphan index files. Returns a JSON report.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "mnemonics_repair",
                    "description": "Auto-repair store issues: rebuild indexes with orphan vectors, delete orphan .bin files. Missing vectors (sql > idx) are reported but not fixed. Returns a JSON summary.",
                    "inputSchema": {"type": "object", "properties": {}},
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
                texts = args.get("texts")
                # Defensive: accept singular `text` (string or list) as alias for `texts`.
                # Empirically a common caller mistake; silent-drop here cost real memories.
                if texts is None and "text" in args:
                    alias = args["text"]
                    texts = [alias] if isinstance(alias, str) else alias
                if isinstance(texts, str):
                    texts = [texts]
                if texts is None:
                    texts = []
                if not isinstance(texts, list):
                    err("texts must be an array of strings (or a single string)")
                    continue
                if not texts:
                    err("texts must not be empty (did you pass 'text' instead of 'texts'?)")
                    continue
                if not all(isinstance(t, str) and t.strip() for t in texts):
                    err("each item in texts must be a non-empty string")
                    continue
                summaries = args.get("summaries")
                if summaries is not None and (
                    not isinstance(summaries, list)
                    or len(summaries) != len(texts)
                    or any(s is not None and not isinstance(s, str) for s in summaries)
                ):
                    err("summaries must be an array of (string|null) the same length as texts")
                    continue
                n = _ingest(
                    texts=texts,
                    store=_get_store(),
                    ns=args.get("ns", "default"),
                    summaries=summaries,
                )
                ok({"content": [{"type": "text", "text": f"Stored {n} chunks."}]})

            elif name == "mnemonics_retrieve":
                candidate_k = int(args.get("candidate_k", 50))
                if candidate_k < 1:
                    err("candidate_k must be >= 1")
                    continue
                try:
                    result = _retrieve(
                        query=args["query"],
                        store=_get_store(),
                        ns=args.get("ns", "default"),
                        top_k=int(args.get("top_k", 5)),
                        decay=bool(args.get("decay", True)),
                        hybrid=bool(args.get("hybrid", True)),
                        candidate_k=candidate_k,
                        rerank=bool(args.get("rerank", False)),
                    )
                except RuntimeError as e:
                    err(str(e))
                    continue
                tier_label = {0: "pin", 1: "def", 2: "amb"}
                lines = []
                for r in result["results"]:
                    header = (
                        f"[{r['score']:.3f}] [raw={r['raw_score']:.3f} decay={r['decay_factor']:.2f} "
                        f"boost={r['boost']:.2f} age={r['age_days']:.0f}d "
                        f"tier={tier_label.get(r['tier'], '?')}]"
                    )
                    # If a summary is stored alongside the raw chunk, surface
                    # it on its own line — gist on top, raw evidence below.
                    summary = r.get("summary")
                    if summary:
                        lines.append(f"{header} {summary[:200]}")
                        lines.append(f"    └─ raw: {r['text'][:200]}")
                    else:
                        lines.append(f"{header} {r['text'][:200]}")
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
                tier = int(args.get("tier", 2))
                cands = store.gc_candidates(ns=ns, age_days=age_days, tier=tier)
                if dry_run:
                    text = (
                        f"{len(cands)} candidate(s) (dry-run, tier={tier}):\n"
                        + "\n".join(f"  id={c['id']} ns={c['ns']} age={c['age_days']}d" for c in cands[:30])
                    )
                else:
                    n = store.gc(ns=ns, age_days=age_days, tier=tier)
                    text = f"Deleted {n} row(s) (tier={tier})."
                ok({"content": [{"type": "text", "text": text}]})

            elif name == "mnemonics_health":
                report = _get_store().health_check()
                ok({"content": [{"type": "text", "text": json.dumps(report, indent=2)}]})

            elif name == "mnemonics_repair":
                fix = _get_store().repair()
                ok({"content": [{"type": "text", "text": json.dumps(fix, indent=2)}]})

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
