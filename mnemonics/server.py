"""MCP + REST server for mnemonics.

Endpoints:
  GET  /health
  GET  /doctor                 — store health report (DB integrity, index vs SQL, capacity)
  POST /ingest   {"texts": [...], "ns": "default", "meta": [...]}
  POST /retrieve {"query": "...", "ns": "default", "top_k": 5}
  GET  /stats                  — per-namespace tier breakdown (pin/def/amb counts)
  POST /pin           {"id": N}       — pin a memory (tier=0, never decays)
  POST /tier          {"id": N, "tier": 0|1|2}
  POST /forget-ns     {"ns": "...", "before": "...", "tier": N, "dry_run": true}
  POST /rebuild-index {"ns": "..."}  — rebuild index from SQL source of truth
  POST /gc       {"ns": "...", "age_days": 30, "tier": 2, "dry_run": true}
  POST /forget   {"ns": "...", "before": "2026-01-01", "tier": 1, "dry_run": true}
  POST /repair                 — auto-fix orphan vectors and orphan index files
  GET  /namespaces
  GET  /count?ns=default
  DELETE /memory/<id>

MCP tools (JSON-RPC over stdio):
  mnemonics_ingest   — store memories
  mnemonics_retrieve — semantic search
  mnemonics_forget       — delete a memory by id
  mnemonics_forget_ns    — bulk delete all memories in a namespace (dry-run by default)
  mnemonics_rebuild_index — rebuild hnswlib index for a namespace from SQL source of truth
  mnemonics_pin      — pin a memory (tier=0, never decays)
  mnemonics_tier     — set memory tier (0/1/2)
  mnemonics_gc       — garbage-collect old ambient/default memories
  mnemonics_stats    — list namespaces with chunk counts
  mnemonics_health   — store health check (DB integrity, index vs SQL counts)
  mnemonics_repair   — auto-repair orphan vectors and orphan index files
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

try:
    _VERSION = importlib.metadata.version("mnemonics")
except importlib.metadata.PackageNotFoundError:
    _VERSION = "0.3.0"
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

    def do_POST(self) -> None:
        try:
            body = self._body()
        except Exception:
            self._json(400, {"error": "invalid JSON"})
            return

        if self.path == "/repair":
            self._json(200, _get_store().repair())

        elif self.path == "/rebuild-index":
            ns = body.get("ns", "").strip()
            if not ns:
                self._json(400, {"error": "ns is required"})
                return
            try:
                old_n, new_n = _get_store().rebuild_ns_index(ns)
                self._json(200, {"ns": ns, "old_count": old_n, "new_count": new_n, "removed": old_n - new_n})
            except RuntimeError as e:
                self._json(409, {"error": str(e)})

        elif self.path == "/gc":
            store = _get_store()
            ns = body.get("ns") or None
            age_days = int(body.get("age_days", 30))
            tier = int(body.get("tier", 2))
            dry_run = bool(body.get("dry_run", True))
            if tier not in (1, 2):
                self._json(400, {"error": "tier must be 1 or 2"})
                return
            candidates = store.gc_candidates(ns=ns, age_days=age_days, tier=tier)
            if dry_run:
                self._json(200, {"candidates": len(candidates), "dry_run": True})
            else:
                n = store.gc(ns=ns, age_days=age_days, tier=tier)
                self._json(200, {"deleted": n, "dry_run": False})

        elif self.path == "/forget":
            ns = body.get("ns", "").strip()
            if not ns:
                self._json(400, {"error": "ns is required"})
                return
            store = _get_store()
            before = body.get("before") or None
            tier_val = body.get("tier")
            tier_filter = int(tier_val) if tier_val is not None else None  # type: ignore[assignment]
            dry_run = bool(body.get("dry_run", True))
            candidates = store.forget_candidates(ns=ns, before=before, tier=tier_filter)
            if dry_run:
                self._json(200, {"candidates": len(candidates), "dry_run": True})
            else:
                n = store.forget(ns=ns, before=before, tier=tier_filter)
                self._json(200, {"deleted": n, "dry_run": False})

        elif self.path == "/search-bm25":
            query = body.get("query", "").strip()
            if not query:
                self._json(400, {"error": "query is required"})
                return
            ns_val = body.get("ns", "default")
            top_k = int(body.get("top_k", 5))
            hits = _get_store().search_bm25(query, ns=ns_val, top_k=top_k)
            self._json(200, {"query": query, "ns": ns_val, "results": hits})

        elif self.path == "/forget-ns":
            ns = body.get("ns", "").strip()
            if not ns:
                self._json(400, {"error": "ns is required"})
                return
            store = _get_store()
            before = body.get("before") or None
            tier_val = body.get("tier")
            tier_filter = int(tier_val) if tier_val is not None else None  # type: ignore[assignment]
            dry_run = bool(body.get("dry_run", True))
            candidates = store.forget_candidates(ns=ns, before=before, tier=tier_filter)
            if dry_run:
                self._json(200, {"ns": ns, "candidates": len(candidates), "dry_run": True})
            else:
                n = store.forget(ns=ns, before=before, tier=tier_filter)
                self._json(200, {"ns": ns, "deleted": n, "dry_run": False})

        elif self.path == "/ingest":
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

        elif self.path == "/pin":
            mid = body.get("id")
            if mid is None:
                self._json(400, {"error": "id is required"})
                return
            pinned = _get_store().pin(int(mid))
            self._json(200, {"id": int(mid), "pinned": pinned})

        elif self.path == "/tier":
            mid = body.get("id")
            tier_val = body.get("tier")
            if mid is None or tier_val is None:
                self._json(400, {"error": "id and tier are required"})
                return
            if int(tier_val) not in (0, 1, 2):
                self._json(400, {"error": "tier must be 0, 1, or 2"})
                return
            changed = _get_store().set_tier(int(mid), int(tier_val))
            self._json(200, {"id": int(mid), "tier": int(tier_val), "changed": changed})

        elif self.path == "/search-by-meta":
            filters = body.get("filters")
            if not isinstance(filters, dict):
                self._json(400, {"error": "filters must be a JSON object"})
                return
            ns_val = body.get("ns", "default")
            limit = int(body.get("limit", 100))
            results = _get_store().search_by_meta(filters, ns=ns_val, limit=limit)
            self._json(200, {"ns": ns_val, "filters": filters, "results": results})

        elif self.path == "/get-many":
            ids = body.get("ids")
            if not isinstance(ids, list):
                self._json(400, {"error": "ids must be an array of integers"})
                return
            results = _get_store().get_many([int(i) for i in ids])
            self._json(200, {"results": results})

        elif self.path == "/delete-many":
            ids = body.get("ids")
            if not isinstance(ids, list):
                self._json(400, {"error": "ids must be an array of integers"})
                return
            deleted = _get_store().delete_many([int(i) for i in ids])
            self._json(200, {"deleted": deleted})

        elif self.path == "/update-meta":
            mid = body.get("id")
            meta = body.get("meta")
            if mid is None or not isinstance(meta, dict):
                self._json(400, {"error": "id and meta (object) are required"})
                return
            changed = _get_store().update_meta(int(mid), meta)
            self._json(200, {"id": int(mid), "changed": changed})

        else:
            self._json(404, {"error": "not found"})

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path == "/health":
            self._json(200, {"status": "ok", "version": _VERSION})
        elif path == "/doctor":
            self._json(200, _get_store().health_check())
        elif path == "/namespaces":
            self._json(200, {"namespaces": _get_store().list_namespaces()})
        elif path == "/count":
            ns = (self.path.split("ns=")[-1] if "ns=" in self.path else "default")
            self._json(200, {"ns": ns, "count": _get_store().count(ns)})
        elif path == "/stats":
            store = _get_store()
            rows = store._db.execute(
                "SELECT ns, tier, COUNT(*) FROM memories GROUP BY ns, tier ORDER BY ns, tier"
            ).fetchall()
            ns_data: dict = {}
            for ns_name, tier, cnt in rows:
                ns_data.setdefault(ns_name, {})[tier] = cnt
            result = []
            for ns_name, tiers in sorted(ns_data.items()):
                total = sum(tiers.values())
                result.append({
                    "ns": ns_name,
                    "total": total,
                    "pin": tiers.get(0, 0),
                    "def": tiers.get(1, 0),
                    "amb": tiers.get(2, 0),
                })
            self._json(200, {"namespaces": result})
        elif path == "/memories":
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            params: dict[str, str] = {}
            for part in qs.split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = v
            ns_val = params.get("ns", "default")
            limit = min(int(params.get("limit", "20")), 100)
            offset = int(params.get("offset", "0"))
            tier_param = params.get("tier")
            tier_val = int(tier_param) if tier_param is not None else None
            rows = _get_store().list_memories(ns=ns_val, limit=limit, offset=offset, tier=tier_val)
            self._json(200, {"ns": ns_val, "offset": offset, "count": len(rows), "rows": rows})
        elif path.startswith("/memory/"):
            try:
                mid = int(path.split("/memory/")[1])
            except ValueError:
                self._json(400, {"error": "invalid id"})
                return
            row = _get_store().get(mid)
            if row is None:
                self._json(404, {"error": f"memory {mid} not found"})
            else:
                self._json(200, row)
        else:
            self._json(404, {"error": "not found"})

    def do_PATCH(self) -> None:
        try:
            body = self._body()
        except Exception:
            self._json(400, {"error": "invalid JSON"})
            return
        if self.path.startswith("/memory/"):
            try:
                mid = int(self.path.split("/memory/")[1])
            except ValueError:
                self._json(400, {"error": "invalid id"})
                return
            if "summary" not in body:
                self._json(400, {"error": "only 'summary' field is patchable"})
                return
            changed = _get_store().update_summary(mid, body.get("summary"))
            if changed:
                self._json(200, {"id": mid, "updated": True})
            else:
                self._json(404, {"error": f"memory {mid} not found"})
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
                "serverInfo": {"name": "mnemonics", "version": _VERSION},
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
                            "meta": {
                                "type": "object",
                                "description": "Optional metadata dict attached to every chunk (e.g. {\"tag\": \"work\", \"source\": \"slack\"}). Same dict applied to all texts in this call.",
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
                            "min_tier": {"type": "integer", "description": "Only return memories with tier >= this value (0=pinned, 1=default, 2=ambient)"},
                            "max_tier": {"type": "integer", "description": "Only return memories with tier <= this value"},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "mnemonics_bm25",
                    "description": "Pure BM25 keyword search — no vector encoding. Instant, exact-token matching. Best for date strings, IDs, names, or any query where you know the exact words to look for. Returns id, text, tier, score.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Keyword query"},
                            "ns": {"type": "string", "description": "Namespace to search (default: 'default')"},
                            "top_k": {"type": "integer", "description": "Max results (default: 5)"},
                            "min_tier": {"type": "integer", "description": "Only return memories with tier >= this value"},
                            "max_tier": {"type": "integer", "description": "Only return memories with tier <= this value"},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "mnemonics_update_summary",
                    "description": "Set or clear the summary field of an existing memory. The summary is a short gist indexed by BM25 alongside the raw text. Pass summary=null to clear. Returns error if the id doesn't exist.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "Memory id to update"},
                            "summary": {"type": ["string", "null"], "description": "New summary text, or null to clear"},
                        },
                        "required": ["id"],
                    },
                },
                {
                    "name": "mnemonics_list",
                    "description": "Browse memories in a namespace, newest first. Returns id, text (first 200 chars), tier, summary, and timestamps. Useful for 'what's in this namespace?' audits without semantic search.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Namespace to list (default: 'default')"},
                            "limit": {"type": "integer", "description": "Max rows to return (default: 20, max: 100)"},
                            "offset": {"type": "integer", "description": "Pagination offset (default: 0)"},
                            "tier": {"type": "integer", "enum": [0, 1, 2], "description": "Filter to a specific tier: 0=pinned, 1=default, 2=ambient (omit for all)"},
                        },
                    },
                },
                {
                    "name": "mnemonics_get",
                    "description": "Fetch a single memory by id. Returns its text, ns, tier, summary, timestamps, and access count. Useful for inspecting a specific memory before pinning, tiering, or deleting it.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"id": {"type": "integer", "description": "Memory id to fetch"}},
                        "required": ["id"],
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
                    "name": "mnemonics_forget_ns",
                    "description": "Bulk delete all memories in a namespace, optionally filtered by date (before) or tier. Dry-run by default — set dry_run=false to actually delete. Pinned (tier=0) rows are excluded unless tier=0 is explicitly passed.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Namespace to clean up (required)"},
                            "before": {"type": "string", "description": "ISO 8601 date string — only delete rows created before this date"},
                            "tier": {"type": "integer", "enum": [0, 1, 2], "description": "Filter to a specific tier (omit to delete all non-pinned rows)"},
                            "dry_run": {"type": "boolean", "description": "If true, return candidate count without deleting (default: true)"},
                        },
                        "required": ["ns"],
                    },
                },
                {
                    "name": "mnemonics_rebuild_index",
                    "description": "Rebuild the hnswlib vector index for a specific namespace from the SQL source of truth. Use when doctor reports 'orphan vectors' (idx > sql) for a namespace. Reads stored vectors by ID — no re-encoding needed. Returns old and new vector counts.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Namespace whose index to rebuild (required)"},
                        },
                        "required": ["ns"],
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
                {
                    "name": "mnemonics_search_by_meta",
                    "description": "Find memories where metadata matches all key=value filters (AND logic). Uses SQLite json_extract — efficient for scalar values.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "filters": {"type": "object", "description": "Key-value pairs to match in meta (all must match)"},
                            "ns": {"type": "string", "description": "Namespace to search (default: 'default')"},
                            "limit": {"type": "integer", "description": "Max results to return (default: 100)"},
                        },
                        "required": ["filters"],
                    },
                },
                {
                    "name": "mnemonics_delete_many",
                    "description": "Delete multiple memories by ID in a single operation. Returns count of actually deleted rows. Missing IDs are silently skipped.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ids": {"type": "array", "items": {"type": "integer"}, "description": "Memory IDs to delete"},
                        },
                        "required": ["ids"],
                    },
                },
                {
                    "name": "mnemonics_update_meta",
                    "description": "Replace the metadata dict of a single memory. Returns changed=true if the ID was found.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "Memory ID"},
                            "meta": {"type": "object", "description": "New metadata to store"},
                        },
                        "required": ["id", "meta"],
                    },
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
                meta_arg = args.get("meta")
                if meta_arg is not None and not isinstance(meta_arg, dict):
                    err("meta must be a JSON object")
                    continue
                metas = [meta_arg] * len(texts) if meta_arg is not None else None
                n = _ingest(
                    texts=texts,
                    store=_get_store(),
                    ns=args.get("ns", "default"),
                    summaries=summaries,
                    meta=metas,
                )
                ok({"content": [{"type": "text", "text": f"Stored {n} chunks."}]})

            elif name == "mnemonics_retrieve":
                candidate_k = int(args.get("candidate_k", 50))
                if candidate_k < 1:
                    err("candidate_k must be >= 1")
                    continue
                try:
                    min_tier_v = args.get("min_tier")
                    max_tier_v = args.get("max_tier")
                    result = _retrieve(
                        query=args["query"],
                        store=_get_store(),
                        ns=args.get("ns", "default"),
                        top_k=int(args.get("top_k", 5)),
                        decay=bool(args.get("decay", True)),
                        hybrid=bool(args.get("hybrid", True)),
                        candidate_k=candidate_k,
                        rerank=bool(args.get("rerank", False)),
                        min_tier=int(min_tier_v) if min_tier_v is not None else None,
                        max_tier=int(max_tier_v) if max_tier_v is not None else None,
                    )
                except RuntimeError as e:
                    err(str(e))
                    continue
                tier_label = {0: "pin", 1: "def", 2: "amb"}
                lines = []
                for r in result["results"]:
                    header = (
                        f"[{r['score']:.3f}] [id={r['id']} raw={r['raw_score']:.3f} decay={r['decay_factor']:.2f} "
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

            elif name == "mnemonics_bm25":
                query = args.get("query", "").strip()
                if not query:
                    err("mnemonics_bm25: 'query' is required")
                    continue
                ns_val = args.get("ns", "default")
                top_k = int(args.get("top_k", 5))
                mt_min = args.get("min_tier")
                mt_max = args.get("max_tier")
                hits = _get_store().search_bm25(
                    query, ns=ns_val, top_k=top_k,
                    min_tier=int(mt_min) if mt_min is not None else None,
                    max_tier=int(mt_max) if mt_max is not None else None,
                )
                if not hits:
                    ok({"content": [{"type": "text", "text": f"No BM25 results for {query!r} in ns={ns_val!r}."}]})
                else:
                    tier_label = {0: "pin", 1: "def", 2: "amb"}
                    lines = []
                    for r in hits:
                        snippet = (r["text"] or "")[:200].replace("\n", " ")
                        tl = tier_label.get(r["tier"], "?")
                        lines.append(f"[{r['score']:.3f}] [{tl}] id={r['id']}  {snippet}")
                        if r.get("summary"):
                            lines.append(f"           summary: {r['summary'][:120]}")
                    ok({"content": [{"type": "text", "text": "\n".join(lines)}]})

            elif name == "mnemonics_update_summary":
                mid = args.get("id")
                if mid is None:
                    err("mnemonics_update_summary: 'id' is required")
                    continue
                summary_val = args.get("summary")  # None clears it
                changed = _get_store().update_summary(int(mid), summary_val)
                if changed:
                    action = "cleared" if summary_val is None else "updated"
                    ok({"content": [{"type": "text", "text": f"Summary {action} for id={mid}."}]})
                else:
                    err(f"mnemonics_update_summary: memory {mid} not found")

            elif name == "mnemonics_list":
                ns_val = args.get("ns", "default")
                limit = min(int(args.get("limit", 20)), 100)
                offset = int(args.get("offset", 0))
                tier_arg = args.get("tier")
                tier_filter = int(tier_arg) if tier_arg is not None else None
                rows = _get_store().list_memories(ns=ns_val, limit=limit, offset=offset, tier=tier_filter)
                if not rows:
                    ok({"content": [{"type": "text", "text": f"No memories in ns={ns_val!r} (offset={offset})."}]})
                else:
                    lines = []
                    for r in rows:
                        snippet = (r["text"] or "")[:200].replace("\n", " ")
                        summary = f"  summary: {r['summary'][:80]}" if r["summary"] else ""
                        lines.append(
                            f"[{r['id']}] tier={r['tier']} created={r['created']}\n"
                            f"  {snippet}{summary}"
                        )
                    header = f"ns={ns_val!r} — showing {len(rows)} row(s) (offset={offset})"
                    ok({"content": [{"type": "text", "text": header + "\n\n" + "\n\n".join(lines)}]})

            elif name == "mnemonics_get":
                mid = args.get("id")
                if mid is None:
                    err("mnemonics_get: 'id' is required")
                    continue
                row = _get_store().get(int(mid))
                if row is None:
                    err(f"mnemonics_get: memory {mid} not found")
                else:
                    import json as _json
                    ok({"content": [{"type": "text", "text": _json.dumps(row, default=str, indent=2)}]})

            elif name == "mnemonics_forget":
                deleted = _get_store().delete(int(args["id"]))
                ok({"content": [{"type": "text", "text": f"Deleted: {deleted}"}]})

            elif name == "mnemonics_forget_ns":
                ns_val = args.get("ns", "").strip()
                if not ns_val:
                    err("mnemonics_forget_ns: 'ns' is required")
                    continue
                store = _get_store()
                before_val = args.get("before") or None
                tier_val = args.get("tier")
                tier_int = int(tier_val) if tier_val is not None else None
                dry_run = bool(args.get("dry_run", True))
                candidates = store.forget_candidates(ns=ns_val, before=before_val, tier=tier_int)
                if dry_run:
                    ok({"content": [{"type": "text", "text": f"Would delete {len(candidates)} row(s) from ns={ns_val!r} (dry-run). Pass dry_run=false to delete."}]})
                else:
                    n = store.forget(ns=ns_val, before=before_val, tier=tier_int)
                    ok({"content": [{"type": "text", "text": f"Deleted {n} row(s) from ns={ns_val!r}."}]})

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

            elif name == "mnemonics_rebuild_index":
                ns_val = args.get("ns", "").strip()
                if not ns_val:
                    err("mnemonics_rebuild_index: 'ns' is required")
                    continue
                try:
                    old_n, new_n = _get_store().rebuild_ns_index(ns_val)
                    removed = old_n - new_n
                    ok({"content": [{"type": "text", "text": f"ns={ns_val}: {old_n} → {new_n} vectors ({removed} orphan(s) removed)"}]})
                except RuntimeError as e:
                    err(str(e))

            elif name == "mnemonics_health":
                report = _get_store().health_check()
                ok({"content": [{"type": "text", "text": json.dumps(report, indent=2)}]})

            elif name == "mnemonics_repair":
                fix = _get_store().repair()
                ok({"content": [{"type": "text", "text": json.dumps(fix, indent=2)}]})

            elif name == "mnemonics_search_by_meta":
                filters = args.get("filters")
                if not isinstance(filters, dict) or not filters:
                    err("mnemonics_search_by_meta: 'filters' must be a non-empty object")
                    continue
                ns_val = args.get("ns", "default")
                limit = int(args.get("limit", 100))
                results = _get_store().search_by_meta(filters, ns=ns_val, limit=limit)
                if not results:
                    ok({"content": [{"type": "text", "text": f"No results for filters {filters} in ns={ns_val!r}."}]})
                else:
                    import json as _json
                    lines = [f"Found {len(results)} result(s):"]
                    for r in results:
                        lines.append(f"  id={r['id']} tier={r['tier']} {r['text'][:120]}")
                    ok({"content": [{"type": "text", "text": "\n".join(lines)}]})

            elif name == "mnemonics_delete_many":
                ids = args.get("ids")
                if not isinstance(ids, list):
                    err("mnemonics_delete_many: 'ids' must be an array of integers")
                    continue
                deleted = _get_store().delete_many([int(i) for i in ids])
                ok({"content": [{"type": "text", "text": f"Deleted {deleted} row(s)."}]})

            elif name == "mnemonics_update_meta":
                mid = args.get("id")
                meta = args.get("meta")
                if mid is None or not isinstance(meta, dict):
                    err("mnemonics_update_meta: 'id' and 'meta' (object) are required")
                    continue
                changed = _get_store().update_meta(int(mid), meta)
                ok({"content": [{"type": "text", "text": f"Meta updated: {changed}"}]})

            elif name == "mnemonics_stats":
                store = _get_store()
                rows = store._db.execute(
                    "SELECT ns, tier, COUNT(*) FROM memories GROUP BY ns, tier ORDER BY ns, tier"
                ).fetchall()
                ns_data: dict[str, dict[int, int]] = {}
                for ns_name, tier, cnt in rows:
                    ns_data.setdefault(ns_name, {})[tier] = cnt
                lines = []
                for ns_name, tiers in sorted(ns_data.items()):
                    total = sum(tiers.values())
                    pin = tiers.get(0, 0)
                    def_ = tiers.get(1, 0)
                    amb = tiers.get(2, 0)
                    lines.append(f"  {ns_name}: {total} chunks  (pin={pin} def={def_} amb={amb})")
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
