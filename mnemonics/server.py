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
        # /import-jsonl reads raw NDJSON — skip the JSON-parse step for it
        if self.path == "/import-jsonl":
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                self._json(400, {"error": "empty body"})
                return
            raw = self.rfile.read(length).decode("utf-8", errors="replace")
            imported = skipped = 0
            errors: list[str] = []
            for lineno, line in enumerate(raw.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"line {lineno}: invalid JSON: {e}")
                    skipped += 1
                    continue
                text = obj.get("text")
                if not text or not isinstance(text, str):
                    errors.append(f"line {lineno}: missing or invalid 'text' field")
                    skipped += 1
                    continue
                ns = obj.get("ns", "default")
                tier = obj.get("tier", 1)
                if tier not in (0, 1, 2):
                    tier = 1
                meta = obj.get("meta") or {}
                summary = obj.get("summary")
                n = _ingest(
                    texts=[text],
                    store=_get_store(),
                    ns=ns,
                    meta=[meta] if meta else None,
                    summaries=[summary],
                    tier=int(tier),
                )
                imported += n
            self._json(200, {"imported": imported, "skipped": skipped,
                             "errors": errors[:10]})
            return

        try:
            body = self._body()
        except Exception:
            self._json(400, {"error": "invalid JSON"})
            return

        if self.path == "/repair":
            self._json(200, _get_store().repair())

        elif self.path == "/reindex-all":
            results_ra = _get_store().reindex_all()
            self._json(200, {"namespaces": results_ra, "count": len(results_ra)})

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

        elif self.path == "/touch-many":
            ids_arg_t = body.get("ids")
            if not isinstance(ids_arg_t, list):
                self._json(400, {"error": "'ids' (list of ints) is required"})
                return
            updated_t = _get_store().touch_many([int(i) for i in ids_arg_t])
            self._json(200, {"touched": updated_t})

        elif self.path == "/bulk-tier":
            ids_arg = body.get("ids")
            tier_arg = body.get("tier")
            if not isinstance(ids_arg, list) or tier_arg is None:
                self._json(400, {"error": "'ids' (list) and 'tier' (0/1/2) are required"})
                return
            try:
                updated = _get_store().set_tier_many([int(i) for i in ids_arg], int(tier_arg))
            except ValueError as e:
                self._json(400, {"error": str(e)})
                return
            self._json(200, {"updated": updated, "tier": int(tier_arg)})

        elif self.path == "/rename-ns":
            old_ns = body.get("old_ns", "").strip()
            new_ns = body.get("new_ns", "").strip()
            if not old_ns or not new_ns:
                self._json(400, {"error": "'old_ns' and 'new_ns' are required"})
                return
            try:
                moved = _get_store().rename_ns(old_ns, new_ns)
            except ValueError as e:
                self._json(409, {"error": str(e)})
                return
            self._json(200, {"old_ns": old_ns, "new_ns": new_ns, "moved": moved})

        elif self.path == "/merge-ns":
            src_ns_m = body.get("src_ns", "").strip()
            dst_ns_m = body.get("dst_ns", "").strip()
            if not src_ns_m or not dst_ns_m:
                self._json(400, {"error": "'src_ns' and 'dst_ns' are required"})
                return
            moved_m = _get_store().merge_ns(src_ns_m, dst_ns_m)
            self._json(200, {"src_ns": src_ns_m, "dst_ns": dst_ns_m, "moved": moved_m})

        elif self.path == "/copy-ns":
            src_ns = body.get("src_ns", "").strip()
            dst_ns = body.get("dst_ns", "").strip()
            if not src_ns or not dst_ns:
                self._json(400, {"error": "'src_ns' and 'dst_ns' are required"})
                return
            try:
                copied = _get_store().copy_ns(src_ns, dst_ns)
            except ValueError as e:
                self._json(409, {"error": str(e)})
                return
            self._json(200, {"src_ns": src_ns, "dst_ns": dst_ns, "copied": copied})

        elif self.path == "/search-bm25":
            query = body.get("query", "").strip()
            if not query:
                self._json(400, {"error": "query is required"})
                return
            ns_val = body.get("ns", "default")
            top_k = int(body.get("top_k", 5))
            bmt_min = body.get("min_tier")
            bmt_max = body.get("max_tier")
            hits = _get_store().search_bm25(
                query, ns=ns_val, top_k=top_k,
                min_tier=int(bmt_min) if bmt_min is not None else None,
                max_tier=int(bmt_max) if bmt_max is not None else None,
            )
            self._json(200, {"query": query, "ns": ns_val, "results": hits})

        elif self.path == "/move-to-ns":
            mtn_ids = body.get("ids")
            mtn_ns = body.get("ns", "").strip()
            if not isinstance(mtn_ids, list) or not mtn_ns:
                self._json(400, {"error": "'ids' (list of ints) and 'ns' (target namespace) are required"})
                return
            n_mtn = _get_store().move_to_ns([int(i) for i in mtn_ids], mtn_ns)
            self._json(200, {"moved": n_mtn, "target_ns": mtn_ns})

        elif self.path == "/touch":
            tc_id = body.get("id")
            if tc_id is None:
                self._json(400, {"error": "'id' (int) is required"})
                return
            ok_tc = _get_store().touch(int(tc_id))
            if not ok_tc:
                self._json(404, {"error": f"memory id={tc_id!r} not found"})
            else:
                self._json(200, {"id": int(tc_id), "touched": True})

        elif self.path == "/bulk-untag":
            but_ids = body.get("ids")
            but_tags = body.get("tags")
            if not isinstance(but_ids, list) or not isinstance(but_tags, list) or not but_tags:
                self._json(400, {"error": "'ids' (list) and 'tags' (non-empty list) are required"})
                return
            n_but = _get_store().bulk_untag([int(i) for i in but_ids], [str(t) for t in but_tags])
            self._json(200, {"updated": n_but, "tags_removed": but_tags})

        elif self.path == "/bulk-tag":
            bt_ids = body.get("ids")
            bt_tags = body.get("tags")
            if not isinstance(bt_ids, list) or not isinstance(bt_tags, list) or not bt_tags:
                self._json(400, {"error": "'ids' (list of ints) and 'tags' (non-empty list of str) are required"})
                return
            n_bt = _get_store().bulk_tag([int(i) for i in bt_ids], [str(t) for t in bt_tags])
            self._json(200, {"updated": n_bt, "tags": bt_tags})

        elif self.path == "/tag":
            tg_id = body.get("id")
            tg_tag = body.get("tag", "").strip()
            if tg_id is None or not tg_tag:
                self._json(400, {"error": "'id' (int) and 'tag' (str) are required"})
                return
            ok_tg = _get_store().tag(int(tg_id), tg_tag)
            if not ok_tg:
                self._json(404, {"error": f"memory id={tg_id!r} not found"})
            else:
                self._json(200, {"id": int(tg_id), "tag": tg_tag, "action": "added"})

        elif self.path == "/untag":
            utg_id = body.get("id")
            utg_tag = body.get("tag", "").strip()
            if utg_id is None or not utg_tag:
                self._json(400, {"error": "'id' (int) and 'tag' (str) are required"})
                return
            ok_utg = _get_store().untag(int(utg_id), utg_tag)
            if not ok_utg:
                self._json(404, {"error": f"memory id={utg_id!r} not found"})
            else:
                self._json(200, {"id": int(utg_id), "tag": utg_tag, "action": "removed"})

        elif self.path == "/update-text":
            ut_id = body.get("id")
            ut_text = body.get("text", "").strip()
            ut_vec = body.get("vec")
            if ut_id is None or not ut_text or not isinstance(ut_vec, list):
                self._json(400, {"error": "'id' (int), 'text' (str), and 'vec' (float list) are required"})
                return
            import numpy as _np_srv_ut
            ok_ut = _get_store().update_text(int(ut_id), ut_text, _np_srv_ut.array(ut_vec, dtype="float32"))
            if not ok_ut:
                self._json(404, {"error": f"memory id={ut_id!r} not found"})
            else:
                self._json(200, {"id": int(ut_id), "updated": True})

        elif self.path == "/clone":
            cl_id = body.get("id")
            cl_ns = body.get("ns", "").strip()
            if cl_id is None or not cl_ns:
                self._json(400, {"error": "'id' (int) and 'ns' (target namespace) are required"})
                return
            new_id = _get_store().clone(int(cl_id), cl_ns)
            if new_id is None:
                self._json(404, {"error": f"memory id={cl_id!r} not found or vector unreadable"})
            else:
                self._json(201, {"cloned_id": new_id, "source_id": int(cl_id), "target_ns": cl_ns})

        elif self.path == "/sample":
            smp_ns = body.get("ns", "default")
            smp_n = min(int(body.get("n", 5)), 100)
            smp_tier = body.get("tier")
            hits_smp = _get_store().sample(
                ns=smp_ns, n=smp_n,
                tier=int(smp_tier) if smp_tier is not None else None,
            )
            self._json(200, {"ns": smp_ns, "n": len(hits_smp), "results": hits_smp})

        elif self.path == "/deduplicate":
            ded_ns = body.get("ns", "default")
            ded_threshold = float(body.get("threshold", 0.98))
            ded_dry = bool(body.get("dry_run", True))
            ded_keep = body.get("keep", "newest")
            result_ded = _get_store().deduplicate(
                ns=ded_ns, threshold=ded_threshold, dry_run=ded_dry, keep=ded_keep
            )
            self._json(200, result_ded)

        elif self.path == "/expire":
            exp_ns = body.get("ns") or None
            exp_age = int(body.get("age_days", 30))
            exp_min_age = body.get("min_age_days")
            demoted = _get_store().expire(
                ns=exp_ns, age_days=exp_age,
                min_age_days=int(exp_min_age) if exp_min_age is not None else None,
            )
            self._json(200, {"demoted": demoted, "age_days": exp_age})

        elif self.path == "/similar-to":
            sim_id = body.get("id")
            if sim_id is None:
                self._json(400, {"error": "'id' (int) is required"})
                return
            top_k_sim = int(body.get("top_k", 5))
            sim_min_tier = body.get("min_tier")
            sim_max_tier = body.get("max_tier")
            sim_hits = _get_store().similar_to(
                int(sim_id), top_k=top_k_sim,
                min_tier=int(sim_min_tier) if sim_min_tier is not None else None,
                max_tier=int(sim_max_tier) if sim_max_tier is not None else None,
            )
            self._json(200, {"id": int(sim_id), "results": sim_hits})

        elif self.path == "/hybrid-search":
            query_hs = body.get("query", "").strip()
            vector_hs = body.get("vector")
            if not query_hs or not isinstance(vector_hs, list):
                self._json(400, {"error": "'query' (str) and 'vector' (list of floats) are required"})
                return
            import numpy as _np_hs
            ns_hs = body.get("ns", "default")
            top_k_hs = int(body.get("top_k", 5))
            rrf_k_hs = int(body.get("rrf_k", 60))
            hs_min_tier = body.get("min_tier")
            hs_max_tier = body.get("max_tier")
            vec_arr = _np_hs.array(vector_hs, dtype="float32")
            hits_hs = _get_store().hybrid_search(
                vec_arr, query_hs, ns=ns_hs, top_k=top_k_hs, rrf_k=rrf_k_hs,
                min_tier=int(hs_min_tier) if hs_min_tier is not None else None,
                max_tier=int(hs_max_tier) if hs_max_tier is not None else None,
            )
            self._json(200, {"query": query_hs, "ns": ns_hs, "results": hits_hs})

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
            tier_val = body.get("tier", 1)
            if tier_val not in (0, 1, 2):
                self._json(400, {"error": "tier must be 0, 1, or 2"})
                return
            n = _ingest(
                texts=texts,
                store=_get_store(),
                ns=body.get("ns", "default"),
                meta=body.get("meta"),
                summaries=summaries,
                tier=int(tier_val),
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
                mt_min = body.get("min_tier")
                mt_max = body.get("max_tier")
                result = _retrieve(
                    query=query,
                    store=_get_store(),
                    ns=body.get("ns", "default"),
                    top_k=int(body.get("top_k", 5)),
                    decay=bool(body.get("decay", True)),
                    hybrid=hybrid,
                    candidate_k=candidate_k,
                    rerank=bool(body.get("rerank", False)),
                    min_tier=int(mt_min) if mt_min is not None else None,
                    max_tier=int(mt_max) if mt_max is not None else None,
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

        elif self.path == "/bulk-update-summary":
            raw_updates = body.get("updates")
            if not isinstance(raw_updates, dict):
                self._json(400, {"error": "'updates' (object mapping id→summary) is required"})
                return
            updates: dict[int, str | None] = {}
            for k, v in raw_updates.items():
                updates[int(k)] = v if isinstance(v, str) else None
            n = _get_store().bulk_update_summary(updates)
            self._json(200, {"updated": n})

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
        elif path.startswith("/count-by-tier"):
            from urllib.parse import urlparse, parse_qs, unquote_plus as _uqp_cbt
            qs_cbt = parse_qs(urlparse(self.path).query)
            ns_cbt_raw = qs_cbt.get("ns", [None])[0]
            ns_cbt = _uqp_cbt(ns_cbt_raw) if ns_cbt_raw is not None else None
            self._json(200, {"ns": ns_cbt, "by_tier": _get_store().count_by_tier(ns_cbt)})
        elif path.startswith("/export-ns/"):
            from urllib.parse import unquote_plus as _uqp_exp
            raw_ns_exp = path[len("/export-ns/"):]
            if not raw_ns_exp:
                self._json(400, {"error": "namespace name required: /export-ns/<ns>"})
                return
            ns_exp = _uqp_exp(raw_ns_exp)
            records = _get_store().export_ns(ns_exp)
            self._json(200, {"ns": ns_exp, "count": len(records), "records": records})

        elif path.startswith("/get-tags/"):
            from urllib.parse import unquote_plus as _uqp_gtag
            raw_id_gt = path[len("/get-tags/"):]
            if not raw_id_gt.isdigit():
                self._json(400, {"error": "memory id must be a positive integer"})
                return
            tags_gt = _get_store().get_tags(int(raw_id_gt))
            if tags_gt is None:
                self._json(404, {"error": f"memory id={raw_id_gt!r} not found"})
            else:
                self._json(200, {"id": int(raw_id_gt), "tags": tags_gt})
        elif path.startswith("/search-date-range"):
            from urllib.parse import urlparse, parse_qs, unquote_plus as _uqp_sdr
            qs_sdr = parse_qs(urlparse(self.path).query)
            ns_sdr_raw = qs_sdr.get("ns", [None])[0]
            ns_sdr = _uqp_sdr(ns_sdr_raw) if ns_sdr_raw is not None else None
            after_sdr = qs_sdr.get("after", [None])[0]
            before_sdr = qs_sdr.get("before", [None])[0]
            limit_sdr = int(qs_sdr.get("limit", ["100"])[0])
            tier_sdr_raw = qs_sdr.get("tier", [None])[0]
            tier_sdr = int(tier_sdr_raw) if tier_sdr_raw is not None else None
            hits_sdr = _get_store().search_date_range(
                ns=ns_sdr, after=after_sdr, before=before_sdr,
                limit=limit_sdr, tier=tier_sdr,
            )
            self._json(200, {"ns": ns_sdr, "after": after_sdr, "before": before_sdr,
                              "results": hits_sdr})
        elif path.startswith("/word-frequency"):
            from urllib.parse import urlparse, parse_qs, unquote_plus as _uqp_wf
            qs_wf = parse_qs(urlparse(self.path).query)
            ns_wf_raw = qs_wf.get("ns", [None])[0]
            ns_wf = _uqp_wf(ns_wf_raw) if ns_wf_raw is not None else None
            top_n_wf = int(qs_wf.get("top_n", ["20"])[0])
            self._json(200, {"ns": ns_wf, "words": _get_store().word_frequency(ns_wf, top_n_wf)})
        elif path.startswith("/find-by-tag"):
            from urllib.parse import urlparse, parse_qs, unquote_plus as _uqp_fbt
            qs_fbt = parse_qs(urlparse(self.path).query)
            tag_fbt = _uqp_fbt(qs_fbt.get("tag", [""])[0]).strip()
            ns_fbt_raw = qs_fbt.get("ns", [None])[0]
            ns_fbt = _uqp_fbt(ns_fbt_raw) if ns_fbt_raw is not None else None
            limit_fbt = int(qs_fbt.get("limit", ["100"])[0])
            if not tag_fbt:
                self._json(400, {"error": "'tag' query param is required"})
                return
            hits_fbt = _get_store().find_by_tag(tag_fbt, ns=ns_fbt, limit=limit_fbt)
            self._json(200, {"tag": tag_fbt, "ns": ns_fbt, "results": hits_fbt})
        elif path.startswith("/list-tags"):
            from urllib.parse import urlparse, parse_qs, unquote_plus as _uqp_lt
            qs_lt = parse_qs(urlparse(self.path).query)
            ns_lt_raw = qs_lt.get("ns", [None])[0]
            ns_lt = _uqp_lt(ns_lt_raw) if ns_lt_raw is not None else None
            self._json(200, {"ns": ns_lt, "tags": _get_store().list_tags(ns_lt)})
        elif path.startswith("/access-stats"):
            from urllib.parse import urlparse, parse_qs, unquote_plus as _uqp_as
            qs_as = parse_qs(urlparse(self.path).query)
            ns_as_raw = qs_as.get("ns", [None])[0]
            ns_as = _uqp_as(ns_as_raw) if ns_as_raw is not None else None
            self._json(200, _get_store().access_stats(ns_as))
        elif path == "/count":
            from urllib.parse import urlparse, parse_qs, unquote_plus
            qs = parse_qs(urlparse(self.path).query)
            ns_raw = qs.get("ns", [None])[0]
            ns = unquote_plus(ns_raw) if ns_raw is not None else None
            count = _get_store().count(ns)
            self._json(200, {"ns": ns, "count": count})
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
            since_param = params.get("since")
            before_param = params.get("before")
            from urllib.parse import unquote_plus
            since_val = unquote_plus(since_param) if since_param else None
            before_val = unquote_plus(before_param) if before_param else None
            rows = _get_store().list_memories(ns=ns_val, limit=limit, offset=offset, tier=tier_val, since=since_val, before=before_val)
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
        elif path == "/recent":
            from urllib.parse import urlparse, parse_qs, unquote_plus
            qs_parsed = parse_qs(urlparse(self.path).query)
            ns_raw = qs_parsed.get("ns", ["default"])[0]
            ns_val = None if ns_raw == "all" else unquote_plus(ns_raw)
            tier_raw = qs_parsed.get("tier", [None])[0]
            limit = min(int(qs_parsed.get("limit", ["20"])[0]), 100)
            hits = _get_store().recent_accessed(
                ns=ns_val, limit=limit,
                tier=int(tier_raw) if tier_raw is not None else None,
            )
            self._json(200, {"count": len(hits), "results": hits})
        elif path == "/top-accessed":
            from urllib.parse import urlparse, parse_qs, unquote_plus
            qs_parsed = parse_qs(urlparse(self.path).query)
            ns_raw = qs_parsed.get("ns", ["default"])[0]
            ns_val = None if ns_raw == "all" else unquote_plus(ns_raw)
            tier_raw = qs_parsed.get("tier", [None])[0]
            limit = min(int(qs_parsed.get("limit", ["20"])[0]), 100)
            hits = _get_store().top_accessed(
                ns=ns_val, limit=limit,
                tier=int(tier_raw) if tier_raw is not None else None,
            )
            self._json(200, {"count": len(hits), "results": hits})
        elif path == "/stats-by-ns":
            result = _get_store().stats_by_ns()
            self._json(200, {"namespaces": result})
        elif path == "/text-search":
            from urllib.parse import urlparse, parse_qs, unquote_plus
            qs_parsed = parse_qs(urlparse(self.path).query)
            q_raw = qs_parsed.get("q", [None])[0]
            if not q_raw:
                self._json(400, {"error": "'q' query parameter is required"})
                return
            q = unquote_plus(q_raw)
            ns_raw = qs_parsed.get("ns", ["default"])[0]
            ns_val = unquote_plus(ns_raw) if ns_raw != "all" else None
            tier_raw = qs_parsed.get("tier", [None])[0]
            limit = min(int(qs_parsed.get("limit", ["20"])[0]), 100)
            hits = _get_store().text_search(
                q, ns=ns_val, limit=limit,
                tier=int(tier_raw) if tier_raw is not None else None,
            )
            self._json(200, {"query": q, "count": len(hits), "results": hits})
        elif path == "/export-jsonl":
            from urllib.parse import urlparse, parse_qs, unquote_plus
            import json as _json
            qs_parsed = parse_qs(urlparse(self.path).query)
            ns_raw = qs_parsed.get("ns", [None])[0]
            tier_raw = qs_parsed.get("tier", [None])[0]
            since_raw = qs_parsed.get("since", [None])[0]
            before_raw = qs_parsed.get("before", [None])[0]
            where_parts = ["1=1"]
            params: list = []
            if ns_raw is not None:
                where_parts.append("ns = ?")
                params.append(unquote_plus(ns_raw))
            if tier_raw is not None:
                where_parts.append("tier = ?")
                params.append(int(tier_raw))
            if since_raw is not None:
                where_parts.append("created >= ?")
                params.append(unquote_plus(since_raw))
            if before_raw is not None:
                where_parts.append("created < ?")
                params.append(unquote_plus(before_raw))
            where = " AND ".join(where_parts)
            rows = _get_store()._db.execute(
                f"SELECT id, ns, text, summary, meta, created, tier, last_accessed, access_count "
                f"FROM memories WHERE {where} ORDER BY id",
                params,
            ).fetchall()
            lines = []
            for r in rows:
                lines.append(_json.dumps({
                    "id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
                    "meta": _json.loads(r[4]), "created": r[5], "tier": r[6],
                    "last_accessed": r[7], "access_count": r[8],
                }, ensure_ascii=False))
            body_bytes = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)
        elif path.startswith("/namespace/"):
            from urllib.parse import unquote_plus
            raw_ns = path[len("/namespace/"):]
            if not raw_ns:
                self._json(400, {"error": "namespace name is required in path: /namespace/<ns>"})
                return
            ns_q = unquote_plus(raw_ns)
            info = _get_store().namespace_info(ns_q)
            if info is None:
                self._json(404, {"error": f"namespace {ns_q!r} not found"})
            else:
                self._json(200, info)
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
            if "summary" not in body and "meta" not in body:
                self._json(400, {"error": "at least one of 'summary' or 'meta' is required"})
                return
            store = _get_store()
            if "summary" in body:
                if not store.update_summary(mid, body.get("summary")):
                    self._json(404, {"error": f"memory {mid} not found"})
                    return
            if "meta" in body:
                meta_val = body.get("meta")
                if not isinstance(meta_val, dict):
                    self._json(400, {"error": "'meta' must be a JSON object"})
                    return
                merge = bool(body.get("merge", True))
                if not store.update_meta(mid, meta_val, merge=merge):
                    self._json(404, {"error": f"memory {mid} not found"})
                    return
            self._json(200, {"id": mid, "updated": True})
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
                            "tier": {
                                "type": "integer",
                                "enum": [0, 1, 2],
                                "description": "Initial tier for all ingested chunks: 0=pinned, 1=default (default), 2=ambient",
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
                    "name": "mnemonics_expire",
                    "description": "Demote stale tier-1 memories to tier-2 (ambient) based on last-access age. Memories not accessed within age_days are downgraded so gc can later remove them. Pinned memories (tier=0) are never touched. Returns count demoted.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Namespace to target (omit for all namespaces)"},
                            "age_days": {"type": "integer", "description": "Memories not accessed within this many days are demoted (default: 30)"},
                            "min_age_days": {"type": "integer", "description": "Only demote memories older than this many days (based on created)"},
                        },
                    },
                },
                {
                    "name": "mnemonics_similar_to",
                    "description": "Find memories most similar to an existing memory by ID. Loads the stored vector for that memory and returns the top_k nearest neighbors (excluding itself). Useful for 'more like this' queries without re-embedding.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "ID of the reference memory"},
                            "top_k": {"type": "integer", "description": "Max similar results to return (default: 5)"},
                            "min_tier": {"type": "integer", "description": "Filter: tier >= this value"},
                            "max_tier": {"type": "integer", "description": "Filter: tier <= this value"},
                        },
                        "required": ["id"],
                    },
                },
                {
                    "name": "mnemonics_hybrid_search",
                    "description": "Hybrid search: combine vector (semantic) and BM25 (keyword) results with Reciprocal Rank Fusion (RRF). Returns the best of both worlds — semantically similar AND keyword-matching results. Each result includes rrf_score, vector_rank, and bm25_rank.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Text query for BM25 matching"},
                            "vector": {"type": "array", "items": {"type": "number"}, "description": "Query vector (same dimension as stored embeddings)"},
                            "ns": {"type": "string", "description": "Namespace to search (default: 'default')"},
                            "top_k": {"type": "integer", "description": "Max results (default: 5)"},
                            "rrf_k": {"type": "integer", "description": "RRF constant — larger values down-weight rank differences (default: 60)"},
                            "min_tier": {"type": "integer", "description": "Only return memories with tier >= this value"},
                            "max_tier": {"type": "integer", "description": "Only return memories with tier <= this value"},
                        },
                        "required": ["query", "vector"],
                    },
                },
                {
                    "name": "mnemonics_touch",
                    "description": "Update a memory's last_accessed timestamp to now without incrementing access_count. Use for 'mark as viewed' semantics that should not bias access statistics.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "Memory ID to touch"},
                        },
                        "required": ["id"],
                    },
                },
                {
                    "name": "mnemonics_bulk_untag",
                    "description": "Remove one or more tags from multiple memories at once. IDs or tags that are absent are silently skipped. Returns count of memories updated.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ids": {"type": "array", "items": {"type": "integer"}},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["ids", "tags"],
                    },
                },
                {
                    "name": "mnemonics_count_by_tier",
                    "description": "Return a {tier: count} breakdown for a namespace (or all namespaces when ns is omitted). Tiers not present are omitted. 0=pinned, 1=default, 2=ambient.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Namespace (omit for all)"},
                        },
                    },
                },
                {
                    "name": "mnemonics_export_ns",
                    "description": "Export all memories in a namespace as a JSON-serialisable list of dicts (no vectors). Each record includes id, ns, text, summary, meta (dict), tier, created, last_accessed, and access_count.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Namespace to export"},
                        },
                        "required": ["ns"],
                    },
                },
                {
                    "name": "mnemonics_bulk_tag",
                    "description": "Add one or more tags to multiple memories at once. Tags already present on a memory are not duplicated. Returns count of memories updated.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ids": {"type": "array", "items": {"type": "integer"}, "description": "Memory IDs to tag"},
                            "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags to add"},
                        },
                        "required": ["ids", "tags"],
                    },
                },
                {
                    "name": "mnemonics_get_tags",
                    "description": "Return the tags list from a memory's meta JSON. Returns an empty list if the memory has no tags, or an error if the memory does not exist.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "Memory ID"},
                        },
                        "required": ["id"],
                    },
                },
                {
                    "name": "mnemonics_search_date_range",
                    "description": "Return memories whose created timestamp falls within an optional date range (ISO-8601 strings for 'after' and 'before'). Both bounds are optional. Optionally filter by tier. Results ordered by created DESC.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Namespace (omit for all)"},
                            "after": {"type": "string", "description": "Lower bound, e.g. '2026-01-01'"},
                            "before": {"type": "string", "description": "Upper bound, e.g. '2026-12-31'"},
                            "limit": {"type": "integer", "default": 100},
                            "tier": {"type": "integer", "description": "Filter by tier (0=pinned, 1=default, 2=ambient)"},
                        },
                    },
                },
                {
                    "name": "mnemonics_word_frequency",
                    "description": "Return the most frequent words across all memory texts in a namespace (or all namespaces). Tokenises on whitespace and punctuation; minimum word length is 2. Returns {word, count} list sorted by count descending.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Namespace to analyse (omit for all namespaces)"},
                            "top_n": {"type": "integer", "default": 20, "description": "Max words to return (max 500)"},
                        },
                    },
                },
                {
                    "name": "mnemonics_find_by_tag",
                    "description": "Find memories that have a specific tag in their meta.tags list. Returns id, ns, text, summary, tier, and created for each match.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "tag": {"type": "string", "description": "Tag to search for"},
                            "ns": {"type": "string", "description": "Namespace to search (omit for all namespaces)"},
                            "limit": {"type": "integer", "default": 100, "description": "Max results"},
                        },
                        "required": ["tag"],
                    },
                },
                {
                    "name": "mnemonics_list_tags",
                    "description": "List all distinct tags used in a namespace (or all namespaces) with their occurrence counts, sorted by count descending.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Namespace to query (omit for all namespaces)"},
                        },
                    },
                },
                {
                    "name": "mnemonics_tag",
                    "description": "Add a tag string to the 'tags' list inside a memory's meta JSON. Idempotent — adding an existing tag is a no-op.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "Memory ID"},
                            "tag": {"type": "string", "description": "Tag to add"},
                        },
                        "required": ["id", "tag"],
                    },
                },
                {
                    "name": "mnemonics_untag",
                    "description": "Remove a tag string from the 'tags' list inside a memory's meta JSON. Idempotent — removing an absent tag is a no-op.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "Memory ID"},
                            "tag": {"type": "string", "description": "Tag to remove"},
                        },
                        "required": ["id", "tag"],
                    },
                },
                {
                    "name": "mnemonics_access_stats",
                    "description": "Return access-count and last-accessed statistics for a namespace (or all namespaces when ns is omitted). Fields: total, total_accesses, avg_accesses, max_accesses, never_accessed, most_recent_access.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Namespace to query (omit to query all namespaces)"},
                        },
                    },
                },
                {
                    "name": "mnemonics_update_text",
                    "description": "Replace the text and embedding vector of an existing memory. The caller must supply the pre-computed embedding (float list, same dimension as the store's index). The old vector is marked deleted in the index and the new one is inserted under the same memory id. Returns error if the memory does not exist.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "Memory ID to update"},
                            "text": {"type": "string", "description": "New text content"},
                            "vec": {"type": "array", "items": {"type": "number"}, "description": "New embedding vector (list of floats, must match store dimension)"},
                        },
                        "required": ["id", "text", "vec"],
                    },
                },
                {
                    "name": "mnemonics_clone",
                    "description": "Clone a single memory (identified by id) to a different namespace. Creates a new row in target_ns with the same text, summary, meta, and tier, and copies the vector into the target namespace index. Returns the new memory id. Run mnemonics_rebuild_index on both namespaces afterward to fully sync vector indexes.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "Memory ID to clone"},
                            "ns": {"type": "string", "description": "Target namespace for the clone"},
                        },
                        "required": ["id", "ns"],
                    },
                },
                {
                    "name": "mnemonics_move_to_ns",
                    "description": "Move specific memories to a different namespace by updating their ns field. Does not update vector indexes — run mnemonics_rebuild_index on both source and target namespaces afterward to keep indexes consistent. Returns count of rows moved.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ids": {"type": "array", "items": {"type": "integer"}, "description": "Memory IDs to move"},
                            "ns": {"type": "string", "description": "Target namespace to move memories into"},
                        },
                        "required": ["ids", "ns"],
                    },
                },
                {
                    "name": "mnemonics_namespace_info",
                    "description": "Return a detailed summary for a single namespace: total memory count, count per tier, oldest/newest timestamps, average text length, total word count, and count of memories with summaries. Returns an error if the namespace doesn't exist.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Namespace to inspect (required)"},
                        },
                        "required": ["ns"],
                    },
                },
                {
                    "name": "mnemonics_sample",
                    "description": "Return up to n randomly sampled memories from a namespace. Useful for spot-checking, building review queues, or seeding evaluation sets without semantic bias.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Namespace to sample from (default: 'default')"},
                            "n": {"type": "integer", "description": "Number of memories to return (default: 5, max: 100)"},
                            "tier": {"type": "integer", "enum": [0, 1, 2], "description": "Filter to a specific tier (omit for all tiers)"},
                        },
                    },
                },
                {
                    "name": "mnemonics_deduplicate",
                    "description": "Find near-duplicate memories in a namespace using cosine similarity on stored vectors. Two memories are duplicates when similarity >= threshold. dry_run=true (default) returns the pairs list without deleting; set dry_run=false to delete. keep='newest' (default) retains the higher ID; keep='oldest' retains the lower ID.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Namespace to deduplicate (default: 'default')"},
                            "threshold": {"type": "number", "description": "Cosine similarity threshold (0-1, default: 0.98). Lower = more aggressive dedup."},
                            "dry_run": {"type": "boolean", "description": "If true (default), only list pairs without deleting"},
                            "keep": {"type": "string", "enum": ["newest", "oldest"], "description": "Which duplicate to keep (default: 'newest')"},
                        },
                    },
                },
                {
                    "name": "mnemonics_bulk_update_summary",
                    "description": "Update (or clear) summaries for multiple memories in a single transaction. Pass a dict mapping memory_id (as string or int) to a summary string or null to clear. Returns the count of rows actually updated — missing IDs are silently skipped.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "updates": {
                                "type": "object",
                                "description": "Map of memory_id → summary string (or null to clear). Keys can be integers or string representations of integers.",
                                "additionalProperties": {"type": ["string", "null"]},
                            },
                        },
                        "required": ["updates"],
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
                            "since": {"type": "string", "description": "ISO date string (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS). Only return memories created on or after this date."},
                            "before": {"type": "string", "description": "ISO date string. Only return memories created before this date. Combine with since for date-range queries."},
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
                    "name": "mnemonics_reindex_all",
                    "description": "Rebuild the hnswlib vector index for every namespace in the store. Useful after bulk imports, data corruption, or when health_check reports orphan vectors across multiple namespaces. Returns per-namespace old/new element counts.",
                    "inputSchema": {"type": "object", "properties": {}},
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
                    "name": "mnemonics_top_accessed",
                    "description": "Return the most frequently accessed memories, ordered by access count descending. Complements mnemonics_recent (time-based) with frequency-based retrieval — useful for surfacing long-term valuable memories.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Namespace to filter (default: 'default', use 'all' for all namespaces)"},
                            "tier": {"type": "integer", "enum": [0, 1, 2], "description": "Filter by tier (omit for all tiers)"},
                            "limit": {"type": "integer", "description": "Max results (default: 20, capped at 100)"},
                        },
                    },
                },
                {
                    "name": "mnemonics_recent",
                    "description": "Return recently accessed memories, ordered by last retrieval time (most recent first). Useful for surfacing active context in AI sessions.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Namespace to filter (default: 'default', use 'all' for all namespaces)"},
                            "tier": {"type": "integer", "enum": [0, 1, 2], "description": "Filter by tier (omit for all tiers)"},
                            "limit": {"type": "integer", "description": "Max results to return (default: 20, capped at 100)"},
                        },
                    },
                },
                {
                    "name": "mnemonics_update_meta",
                    "description": "Update (or replace) the metadata dict on a single memory. With merge=true (default), provided keys are merged into the existing meta — existing keys not mentioned are preserved. With merge=false, the whole meta dict is replaced.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "Memory ID to update"},
                            "meta": {"type": "object", "description": "Key-value pairs to set on the memory's metadata"},
                            "merge": {"type": "boolean", "description": "If true (default), merge into existing meta. If false, replace entirely."},
                        },
                        "required": ["id", "meta"],
                    },
                },
                {
                    "name": "mnemonics_touch_many",
                    "description": "Mark multiple memories as accessed right now: updates last_accessed = now() and increments access_count for each ID. Useful after get_many() or any retrieval path that doesn't auto-touch. Returns the number of rows actually updated.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ids": {"type": "array", "items": {"type": "integer"}, "description": "Memory IDs to mark as accessed"},
                        },
                        "required": ["ids"],
                    },
                },
                {
                    "name": "mnemonics_bulk_tier",
                    "description": "Set tier for multiple memories in a single operation. Useful for bulk-pinning or bulk-archiving a set of results. Returns how many rows were actually updated.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ids": {"type": "array", "items": {"type": "integer"}, "description": "Memory IDs to update"},
                            "tier": {"type": "integer", "enum": [0, 1, 2], "description": "Target tier (0=pinned, 1=default, 2=ambient)"},
                        },
                        "required": ["ids", "tier"],
                    },
                },
                {
                    "name": "mnemonics_count",
                    "description": "Count memories in a namespace. Omit ns (or pass null) to count across all namespaces.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Namespace to count (default: 'default', null for all)"},
                        },
                    },
                },
                {
                    "name": "mnemonics_get_many",
                    "description": "Fetch multiple memories by ID in a single call. Returns a list of memory objects. Missing IDs are silently omitted.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ids": {"type": "array", "items": {"type": "integer"}, "description": "Memory IDs to fetch"},
                        },
                        "required": ["ids"],
                    },
                },
                {
                    "name": "mnemonics_export",
                    "description": "Export memories as a JSONL string. Useful for snapshots, cross-store migration, or backing up a namespace. Filters by ns, tier, since, and before. Returns JSONL where each line is one memory object.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "ns": {"type": "string", "description": "Namespace to export (omit for all namespaces)"},
                            "tier": {"type": "integer", "enum": [0, 1, 2], "description": "Filter to specific tier (omit for all)"},
                            "since": {"type": "string", "description": "Only export memories created on or after this date (YYYY-MM-DD)"},
                            "before": {"type": "string", "description": "Only export memories created before this date (YYYY-MM-DD)"},
                            "limit": {"type": "integer", "description": "Max rows to export (default: 500)"},
                        },
                    },
                },
                {
                    "name": "mnemonics_stats_by_ns",
                    "description": "Return a lightweight per-namespace stats summary: total memory count, count per tier (pinned/default/ambient), and oldest/newest creation timestamps. Faster than health_check — SQL only, no index I/O.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "mnemonics_namespaces",
                    "description": "List all namespaces that exist in the store, ordered alphabetically.",
                    "inputSchema": {"type": "object", "properties": {}},
                },
                {
                    "name": "mnemonics_merge_ns",
                    "description": "Move all memories from src_ns into dst_ns, then remove src_ns. Unlike rename_ns, dst_ns is allowed to already contain memories — the source is appended. The source index file is removed; the destination index is rebuilt on next search.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "src_ns": {"type": "string", "description": "Source namespace to merge from (will be deleted)"},
                            "dst_ns": {"type": "string", "description": "Destination namespace to merge into (may already exist)"},
                        },
                        "required": ["src_ns", "dst_ns"],
                    },
                },
                {
                    "name": "mnemonics_copy_ns",
                    "description": "Copy all memories from one namespace into a new namespace, leaving the source intact. access_count and last_accessed are reset on the copies. Fails if the destination namespace already exists.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "src_ns": {"type": "string", "description": "Source namespace to copy from"},
                            "dst_ns": {"type": "string", "description": "Destination namespace to copy into (must not exist)"},
                        },
                        "required": ["src_ns", "dst_ns"],
                    },
                },
                {
                    "name": "mnemonics_rename_ns",
                    "description": "Rename a namespace — moves all its memories and renames the vector index file. Fails with an error if the target namespace already exists (prevents silent merges).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "old_ns": {"type": "string", "description": "Current namespace name"},
                            "new_ns": {"type": "string", "description": "New namespace name"},
                        },
                        "required": ["old_ns", "new_ns"],
                    },
                },
                {
                    "name": "mnemonics_text_search",
                    "description": "Case-insensitive substring search over memory text and summary. Faster than vector search for exact keyword lookups. Returns rows newest-first.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Substring to search for"},
                            "ns": {"type": "string", "description": "Namespace to search (default: 'default', use 'all' for all namespaces)"},
                            "tier": {"type": "integer", "enum": [0, 1, 2], "description": "Filter to specific tier (omit for all tiers)"},
                            "limit": {"type": "integer", "description": "Max results (default: 20, capped at 100)"},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "mnemonics_import",
                    "description": "Import memories from a JSONL string (counterpart to mnemonics_export). Each line must be a JSON object with at least a 'text' field. Optional fields: ns, tier, summary, meta. ns/tier overrides apply to all rows when set.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "jsonl": {"type": "string", "description": "JSONL string — one JSON object per line, each with at least a 'text' field"},
                            "ns": {"type": "string", "description": "Override namespace for all imported rows (uses per-row ns otherwise, defaulting to 'default')"},
                            "tier": {"type": "integer", "enum": [0, 1, 2], "description": "Override tier for all imported rows (uses per-row tier otherwise, defaulting to 1)"},
                            "dry_run": {"type": "boolean", "description": "If true, parse and count without writing to store"},
                        },
                        "required": ["jsonl"],
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
                tier_arg = args.get("tier", 1)
                if tier_arg not in (0, 1, 2):
                    err("tier must be 0, 1, or 2")
                    continue
                n = _ingest(
                    texts=texts,
                    store=_get_store(),
                    ns=args.get("ns", "default"),
                    summaries=summaries,
                    meta=metas,
                    tier=int(tier_arg),
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

            elif name == "mnemonics_expire":
                exp_ns_m = args.get("ns") or None
                exp_age_m = int(args.get("age_days", 30))
                exp_min_m = args.get("min_age_days")
                demoted_m = _get_store().expire(
                    ns=exp_ns_m, age_days=exp_age_m,
                    min_age_days=int(exp_min_m) if exp_min_m is not None else None,
                )
                ok({"content": [{"type": "text", "text": f"Demoted {demoted_m} memories to tier-2 (age_days={exp_age_m})."}]})

            elif name == "mnemonics_similar_to":
                sim_id_m = args.get("id")
                if sim_id_m is None:
                    err("mnemonics_similar_to: 'id' (int) is required")
                    continue
                sim_top_k = int(args.get("top_k", 5))
                sim_min = args.get("min_tier")
                sim_max = args.get("max_tier")
                sim_hits = _get_store().similar_to(
                    int(sim_id_m), top_k=sim_top_k,
                    min_tier=int(sim_min) if sim_min is not None else None,
                    max_tier=int(sim_max) if sim_max is not None else None,
                )
                if not sim_hits:
                    ok({"content": [{"type": "text", "text": f"No similar memories found for id={sim_id_m}."}]})
                else:
                    tier_label_s = {0: "pin", 1: "def", 2: "amb"}
                    lines_s = []
                    for r in sim_hits:
                        tl = tier_label_s.get(r["tier"], "?")
                        snippet = (r["text"] or "")[:200].replace("\n", " ")
                        lines_s.append(f"[{r['score']:.3f}] [{tl}] id={r['id']}  {snippet}")
                        if r.get("summary"):
                            lines_s.append(f"           summary: {r['summary'][:120]}")
                    ok({"content": [{"type": "text", "text": "\n".join(lines_s)}]})

            elif name == "mnemonics_hybrid_search":
                import numpy as _np_mcp
                hs_query = args.get("query", "").strip()
                hs_vector = args.get("vector")
                if not hs_query or not isinstance(hs_vector, list):
                    err("mnemonics_hybrid_search: 'query' (str) and 'vector' (list) are required")
                    continue
                hs_ns = args.get("ns", "default")
                hs_top_k = int(args.get("top_k", 5))
                hs_rrf_k = int(args.get("rrf_k", 60))
                hs_min_tier = args.get("min_tier")
                hs_max_tier = args.get("max_tier")
                hs_vec_arr = _np_mcp.array(hs_vector, dtype="float32")
                hs_hits = _get_store().hybrid_search(
                    hs_vec_arr, hs_query, ns=hs_ns, top_k=hs_top_k, rrf_k=hs_rrf_k,
                    min_tier=int(hs_min_tier) if hs_min_tier is not None else None,
                    max_tier=int(hs_max_tier) if hs_max_tier is not None else None,
                )
                if not hs_hits:
                    ok({"content": [{"type": "text", "text": f"No hybrid results for {hs_query!r} in ns={hs_ns!r}."}]})
                else:
                    tier_label_hs = {0: "pin", 1: "def", 2: "amb"}
                    lines_hs = []
                    for r in hs_hits:
                        snippet = (r["text"] or "")[:200].replace("\n", " ")
                        tl = tier_label_hs.get(r["tier"], "?")
                        vr = r.get("vector_rank") or "-"
                        br = r.get("bm25_rank") or "-"
                        lines_hs.append(f"[rrf={r['rrf_score']:.4f}] [{tl}] id={r['id']} v={vr} b={br}  {snippet}")
                        if r.get("summary"):
                            lines_hs.append(f"           summary: {r['summary'][:120]}")
                    ok({"content": [{"type": "text", "text": "\n".join(lines_hs)}]})

            elif name == "mnemonics_touch":
                tc_id_m = args.get("id")
                if tc_id_m is None:
                    err("mnemonics_touch: 'id' is required")
                    continue
                ok_tc_m = _get_store().touch(int(tc_id_m))
                if not ok_tc_m:
                    err(f"mnemonics_touch: id={tc_id_m!r} not found")
                    continue
                ok({"content": [{"type": "text", "text": f"Touched id={tc_id_m} (last_accessed updated)."}]})

            elif name == "mnemonics_bulk_untag":
                but_ids_m = args.get("ids")
                but_tags_m = args.get("tags")
                if not isinstance(but_ids_m, list) or not isinstance(but_tags_m, list) or not but_tags_m:
                    err("mnemonics_bulk_untag: 'ids' (list) and 'tags' (non-empty list) are required")
                    continue
                n_but_m = _get_store().bulk_untag([int(i) for i in but_ids_m], [str(t) for t in but_tags_m])
                ok({"content": [{"type": "text", "text": f"Removed tags {but_tags_m} from {n_but_m} memories."}]})

            elif name == "mnemonics_count_by_tier":
                cbt_ns = args.get("ns")
                cbt_counts = _get_store().count_by_tier(cbt_ns)
                tier_labels = {0: "pinned", 1: "default", 2: "ambient"}
                cbt_ns_label = repr(cbt_ns) if cbt_ns is not None else "(all)"
                lines_cbt = [f"Tier counts in ns={cbt_ns_label}:"]
                for t, c in sorted(cbt_counts.items()):
                    lines_cbt.append(f"  {tier_labels.get(t, str(t)):<10}: {c}")
                if not cbt_counts:
                    lines_cbt.append("  (empty)")
                ok({"content": [{"type": "text", "text": "\n".join(lines_cbt)}]})

            elif name == "mnemonics_export_ns":
                exp_ns = args.get("ns", "").strip()
                if not exp_ns:
                    err("mnemonics_export_ns: 'ns' is required")
                    continue
                exp_records = _get_store().export_ns(exp_ns)
                import json as _j_mcp_exp
                ok({"content": [{"type": "text", "text":
                    f"Exported {len(exp_records)} memories from ns={exp_ns!r}.\n" +
                    _j_mcp_exp.dumps(exp_records, default=str, ensure_ascii=False)}]})

            elif name == "mnemonics_bulk_tag":
                bt_ids_m = args.get("ids")
                bt_tags_m = args.get("tags")
                if not isinstance(bt_ids_m, list) or not isinstance(bt_tags_m, list) or not bt_tags_m:
                    err("mnemonics_bulk_tag: 'ids' (list) and 'tags' (non-empty list) are required")
                    continue
                n_bt_m = _get_store().bulk_tag([int(i) for i in bt_ids_m], [str(t) for t in bt_tags_m])
                ok({"content": [{"type": "text", "text":
                    f"Added tags {bt_tags_m} to {n_bt_m} memories."}]})

            elif name == "mnemonics_get_tags":
                gt_id = args.get("id")
                if gt_id is None:
                    err("mnemonics_get_tags: 'id' is required")
                    continue
                gt_tags = _get_store().get_tags(int(gt_id))
                if gt_tags is None:
                    err(f"mnemonics_get_tags: id={gt_id!r} not found")
                    continue
                ok({"content": [{"type": "text", "text":
                    f"Tags for id={gt_id}: {gt_tags if gt_tags else '(none)'}"}]})

            elif name == "mnemonics_search_date_range":
                sdr_ns = args.get("ns")
                sdr_after = args.get("after")
                sdr_before = args.get("before")
                sdr_limit = int(args.get("limit", 100))
                sdr_tier = args.get("tier")
                if sdr_tier is not None:
                    sdr_tier = int(sdr_tier)
                sdr_hits = _get_store().search_date_range(
                    ns=sdr_ns, after=sdr_after, before=sdr_before,
                    limit=sdr_limit, tier=sdr_tier,
                )
                sdr_ns_label = repr(sdr_ns) if sdr_ns is not None else "(all)"
                if not sdr_hits:
                    ok({"content": [{"type": "text", "text":
                        f"No memories found in ns={sdr_ns_label} for that date range."}]})
                    continue
                lines_sdr = [f"Found {len(sdr_hits)} memor{'y' if len(sdr_hits)==1 else 'ies'} in ns={sdr_ns_label}:"]
                tier_labels = {0: "pinned", 1: "default", 2: "ambient"}
                for h in sdr_hits[:20]:
                    tl = tier_labels.get(h["tier"], str(h["tier"]))
                    snippet = h["text"][:70].replace("\n", " ")
                    lines_sdr.append(f"  [{h['created'][:10]}] [{tl}] id={h['id']} {snippet}")
                ok({"content": [{"type": "text", "text": "\n".join(lines_sdr)}]})

            elif name == "mnemonics_word_frequency":
                wf_ns = args.get("ns")
                wf_top = int(args.get("top_n", 20))
                wf_words = _get_store().word_frequency(wf_ns, wf_top)
                wf_ns_label = repr(wf_ns) if wf_ns is not None else "(all)"
                if not wf_words:
                    ok({"content": [{"type": "text", "text": f"No words found in ns={wf_ns_label}."}]})
                    continue
                lines_wf = [f"Top {len(wf_words)} words in ns={wf_ns_label}:"]
                for w in wf_words:
                    lines_wf.append(f"  {w['word']:25s} {w['count']}")
                ok({"content": [{"type": "text", "text": "\n".join(lines_wf)}]})

            elif name == "mnemonics_find_by_tag":
                fbt_tag = args.get("tag", "").strip()
                fbt_ns = args.get("ns")
                fbt_limit = int(args.get("limit", 100))
                if not fbt_tag:
                    err("mnemonics_find_by_tag: 'tag' is required")
                    continue
                fbt_hits = _get_store().find_by_tag(fbt_tag, ns=fbt_ns, limit=fbt_limit)
                if not fbt_hits:
                    ok({"content": [{"type": "text", "text": f"No memories found with tag {fbt_tag!r}."}]})
                    continue
                lines_fbt = [f"Found {len(fbt_hits)} memor{'y' if len(fbt_hits)==1 else 'ies'} with tag {fbt_tag!r}:"]
                for h in fbt_hits[:20]:
                    snippet = h["text"][:80].replace("\n", " ")
                    lines_fbt.append(f"  id={h['id']} ns={h['ns']} [{h['tier']}] {snippet}")
                ok({"content": [{"type": "text", "text": "\n".join(lines_fbt)}]})

            elif name == "mnemonics_list_tags":
                lt_ns = args.get("ns")
                lt_tags = _get_store().list_tags(lt_ns)
                lt_ns_label = repr(lt_ns) if lt_ns is not None else "(all)"
                if not lt_tags:
                    ok({"content": [{"type": "text", "text": f"No tags found in ns={lt_ns_label}."}]})
                    continue
                lines_lt = [f"Tags in ns={lt_ns_label}:"]
                for t in lt_tags:
                    lines_lt.append(f"  {t['tag']:30s} {t['count']}")
                ok({"content": [{"type": "text", "text": "\n".join(lines_lt)}]})

            elif name == "mnemonics_tag":
                tg_id_m = args.get("id")
                tg_tag_m = args.get("tag", "").strip()
                if tg_id_m is None or not tg_tag_m:
                    err("mnemonics_tag: 'id' and 'tag' are required")
                    continue
                ok_tg_m = _get_store().tag(int(tg_id_m), tg_tag_m)
                if not ok_tg_m:
                    err(f"mnemonics_tag: id={tg_id_m!r} not found")
                    continue
                ok({"content": [{"type": "text", "text": f"Added tag {tg_tag_m!r} to id={tg_id_m}."}]})

            elif name == "mnemonics_untag":
                utg_id_m = args.get("id")
                utg_tag_m = args.get("tag", "").strip()
                if utg_id_m is None or not utg_tag_m:
                    err("mnemonics_untag: 'id' and 'tag' are required")
                    continue
                ok_utg_m = _get_store().untag(int(utg_id_m), utg_tag_m)
                if not ok_utg_m:
                    err(f"mnemonics_untag: id={utg_id_m!r} not found")
                    continue
                ok({"content": [{"type": "text", "text": f"Removed tag {utg_tag_m!r} from id={utg_id_m}."}]})

            elif name == "mnemonics_access_stats":
                ns_as_m = args.get("ns")  # None = all
                stats_as = _get_store().access_stats(ns_as_m)
                lines_as = [
                    f"Namespace        : {stats_as['ns'] or '(all)'}",
                    f"Total memories   : {stats_as['total']}",
                    f"Total accesses   : {stats_as['total_accesses']}",
                    f"Avg accesses     : {stats_as['avg_accesses']:.3f}",
                    f"Max accesses     : {stats_as['max_accesses']}",
                    f"Never accessed   : {stats_as['never_accessed']}",
                    f"Most recent      : {stats_as['most_recent_access'] or 'none'}",
                ]
                ok({"content": [{"type": "text", "text": "\n".join(lines_as)}]})

            elif name == "mnemonics_update_text":
                ut_id_m = args.get("id")
                ut_text_m = args.get("text", "").strip()
                ut_vec_m = args.get("vec")
                if ut_id_m is None or not ut_text_m or not isinstance(ut_vec_m, list):
                    err("mnemonics_update_text: 'id', 'text', and 'vec' (float list) are required")
                    continue
                import numpy as _np_mcp_ut
                ok_ut_m = _get_store().update_text(
                    int(ut_id_m), ut_text_m,
                    _np_mcp_ut.array(ut_vec_m, dtype="float32"),
                )
                if not ok_ut_m:
                    err(f"mnemonics_update_text: id={ut_id_m!r} not found")
                    continue
                ok({"content": [{"type": "text", "text": f"Updated id={ut_id_m} text and vector."}]})

            elif name == "mnemonics_clone":
                cl_id_m = args.get("id")
                cl_ns_m = args.get("ns", "").strip()
                if cl_id_m is None or not cl_ns_m:
                    err("mnemonics_clone: 'id' (int) and 'ns' (str) are required")
                    continue
                new_cl_id = _get_store().clone(int(cl_id_m), cl_ns_m)
                if new_cl_id is None:
                    err(f"mnemonics_clone: id={cl_id_m!r} not found or vector unreadable")
                    continue
                ok({"content": [{"type": "text", "text": f"Cloned id={cl_id_m} → new id={new_cl_id} in ns={cl_ns_m!r}."}]})

            elif name == "mnemonics_move_to_ns":
                mtn_ids_m = args.get("ids")
                mtn_ns_m = args.get("ns", "").strip()
                if not isinstance(mtn_ids_m, list) or not mtn_ns_m:
                    err("mnemonics_move_to_ns: 'ids' (list) and 'ns' (str) are required")
                    continue
                n_mtn_m = _get_store().move_to_ns([int(i) for i in mtn_ids_m], mtn_ns_m)
                ok({"content": [{"type": "text", "text": f"Moved {n_mtn_m} memories to ns={mtn_ns_m!r}."}]})

            elif name == "mnemonics_namespace_info":
                ns_ni = args.get("ns", "").strip()
                if not ns_ni:
                    err("mnemonics_namespace_info: 'ns' is required")
                    continue
                info_ni = _get_store().namespace_info(ns_ni)
                if info_ni is None:
                    err(f"mnemonics_namespace_info: namespace {ns_ni!r} not found")
                    continue
                tier_labels = {0: "pinned", 1: "default", 2: "ambient"}
                lines_ni = [f"Namespace: {ns_ni}"]
                lines_ni.append(f"  Total memories : {info_ni['total']}")
                for t, c in sorted(info_ni["by_tier"].items()):
                    lines_ni.append(f"  {tier_labels.get(t, str(t)):<10}: {c}")
                lines_ni.append(f"  Oldest         : {info_ni['oldest']}")
                lines_ni.append(f"  Newest         : {info_ni['newest']}")
                lines_ni.append(f"  Avg text len   : {info_ni['avg_text_len']:.0f} chars")
                lines_ni.append(f"  Total words    : {info_ni['total_words']}")
                lines_ni.append(f"  With summary   : {info_ni['with_summary']}")
                ok({"content": [{"type": "text", "text": "\n".join(lines_ni)}]})

            elif name == "mnemonics_sample":
                smp_ns_m = args.get("ns", "default")
                smp_n_m = min(int(args.get("n", 5)), 100)
                smp_tier_m = args.get("tier")
                results_smp = _get_store().sample(
                    ns=smp_ns_m, n=smp_n_m,
                    tier=int(smp_tier_m) if smp_tier_m is not None else None,
                )
                if not results_smp:
                    ok({"content": [{"type": "text", "text": f"No memories found in ns={smp_ns_m!r}."}]})
                else:
                    lines_smp = [f"Sample of {len(results_smp)} from ns={smp_ns_m!r}:"]
                    for r in results_smp:
                        lines_smp.append(f"  id={r['id']} tier={r['tier']}  {(r['text'] or '')[:120]}")
                    ok({"content": [{"type": "text", "text": "\n".join(lines_smp)}]})

            elif name == "mnemonics_deduplicate":
                ded_ns_m = args.get("ns", "default")
                ded_thr_m = float(args.get("threshold", 0.98))
                ded_dry_m = bool(args.get("dry_run", True))
                ded_keep_m = args.get("keep", "newest")
                res_ded = _get_store().deduplicate(
                    ns=ded_ns_m, threshold=ded_thr_m, dry_run=ded_dry_m, keep=ded_keep_m
                )
                lines_ded = [f"Found {len(res_ded['pairs'])} duplicate pair(s). Removed: {res_ded['removed']}."]
                for p in res_ded["pairs"][:20]:
                    lines_ded.append(f"  kept={p['kept_id']} removed={p['removed_id']} sim={p['similarity']:.4f}")
                ok({"content": [{"type": "text", "text": "\n".join(lines_ded)}]})

            elif name == "mnemonics_bulk_update_summary":
                raw_upd = args.get("updates")
                if not isinstance(raw_upd, dict):
                    err("mnemonics_bulk_update_summary: 'updates' (object) is required")
                    continue
                upd: dict[int, str | None] = {}
                for k, v in raw_upd.items():
                    upd[int(k)] = v if isinstance(v, str) else None
                n_bus = _get_store().bulk_update_summary(upd)
                ok({"content": [{"type": "text", "text": f"Updated summaries for {n_bus} memories."}]})

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
                since_arg = args.get("since")
                before_arg = args.get("before")
                rows = _get_store().list_memories(ns=ns_val, limit=limit, offset=offset, tier=tier_filter, since=since_arg, before=before_arg)
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

            elif name == "mnemonics_reindex_all":
                results_ra_m = _get_store().reindex_all()
                lines_ra = [f"Rebuilt {len(results_ra_m)} namespace(s):"]
                for r in results_ra_m:
                    if "error" in r:
                        lines_ra.append(f"  {r['ns']}: ERROR — {r['error']}")
                    else:
                        lines_ra.append(f"  {r['ns']}: {r['old_count']} → {r['new_count']}")
                ok({"content": [{"type": "text", "text": "\n".join(lines_ra)}]})

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
                merge_flag = bool(args.get("merge", True))
                changed = _get_store().update_meta(int(mid), meta, merge=merge_flag)
                if changed:
                    ok({"content": [{"type": "text", "text": f"Memory {mid} meta updated (merge={merge_flag})."}]})
                else:
                    err(f"memory {mid} not found")

            elif name == "mnemonics_export":
                ns_val = args.get("ns")
                tier_arg = args.get("tier")
                tier_filter = int(tier_arg) if tier_arg is not None else None
                since_arg = args.get("since")
                before_arg = args.get("before")
                limit = min(int(args.get("limit", 500)), 2000)
                where_parts = ["1=1"]
                sql_params: list = []
                if ns_val is not None:
                    where_parts.append("ns = ?")
                    sql_params.append(ns_val)
                if tier_filter is not None:
                    where_parts.append("tier = ?")
                    sql_params.append(tier_filter)
                if since_arg is not None:
                    where_parts.append("created >= ?")
                    sql_params.append(since_arg)
                if before_arg is not None:
                    where_parts.append("created < ?")
                    sql_params.append(before_arg)
                where = " AND ".join(where_parts)
                sql_params.append(limit)
                rows = _get_store()._db.execute(
                    f"SELECT id, ns, text, summary, meta, created, tier, last_accessed, access_count "
                    f"FROM memories WHERE {where} ORDER BY id LIMIT ?",
                    sql_params,
                ).fetchall()
                lines = []
                for r in rows:
                    lines.append(json.dumps({
                        "id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
                        "meta": json.loads(r[4]), "created": r[5], "tier": r[6],
                        "last_accessed": r[7], "access_count": r[8],
                    }, ensure_ascii=False))
                result_text = "\n".join(lines) if lines else "(no memories matched)"
                ok({"content": [{"type": "text", "text": result_text}]})

            elif name == "mnemonics_top_accessed":
                ns_arg_ta = args.get("ns", "default")
                ns_val_ta = None if ns_arg_ta == "all" else ns_arg_ta
                tier_arg_ta = args.get("tier")
                limit_ta = min(int(args.get("limit", 20)), 100)
                hits_ta = _get_store().top_accessed(
                    ns=ns_val_ta, limit=limit_ta,
                    tier=int(tier_arg_ta) if tier_arg_ta is not None else None,
                )
                if hits_ta:
                    lines_ta = [
                        f"id={h['id']} ns={h['ns']} tier={h['tier']} "
                        f"count={h['access_count']}: {h['text'][:100]}"
                        for h in hits_ta
                    ]
                    ok({"content": [{"type": "text", "text": "\n".join(lines_ta)}]})
                else:
                    ok({"content": [{"type": "text", "text": "(no accessed memories found)"}]})

            elif name == "mnemonics_recent":
                ns_arg = args.get("ns", "default")
                ns_val = None if ns_arg == "all" else ns_arg
                tier_arg_r = args.get("tier")
                limit = min(int(args.get("limit", 20)), 100)
                hits = _get_store().recent_accessed(
                    ns=ns_val, limit=limit,
                    tier=int(tier_arg_r) if tier_arg_r is not None else None,
                )
                if hits:
                    lines_r = [
                        f"id={h['id']} ns={h['ns']} tier={h['tier']} "
                        f"accessed={h['last_accessed'] or 'never'}: {h['text'][:100]}"
                        for h in hits
                    ]
                    ok({"content": [{"type": "text", "text": "\n".join(lines_r)}]})
                else:
                    ok({"content": [{"type": "text", "text": "(no recently accessed memories)"}]})

            elif name == "mnemonics_touch_many":
                ids_arg_tm = args.get("ids")
                if not isinstance(ids_arg_tm, list):
                    err("mnemonics_touch_many: 'ids' (list of ints) is required")
                    continue
                touched = _get_store().touch_many([int(i) for i in ids_arg_tm])
                ok({"content": [{"type": "text", "text": f"Touched {touched} memory/memories (last_accessed + access_count updated)."}]})

            elif name == "mnemonics_bulk_tier":
                ids_arg = args.get("ids", [])
                tier_arg = args.get("tier")
                if not isinstance(ids_arg, list) or tier_arg is None:
                    err("mnemonics_bulk_tier: 'ids' (list) and 'tier' (0/1/2) are required")
                    continue
                try:
                    updated = _get_store().set_tier_many([int(i) for i in ids_arg], int(tier_arg))
                except ValueError as e:
                    err(str(e))
                    continue
                ok({"content": [{"type": "text", "text": f"Updated {updated} memory/memories to tier {tier_arg}."}]})

            elif name == "mnemonics_count":
                ns_arg = args.get("ns", "default")
                ns_val = None if ns_arg is None or ns_arg == "null" else ns_arg
                count = _get_store().count(ns=ns_val)
                label = "all namespaces" if ns_val is None else f"ns={ns_val!r}"
                ok({"content": [{"type": "text", "text": f"{count} memories ({label})"}]})

            elif name == "mnemonics_get_many":
                ids_arg = args.get("ids", [])
                if not isinstance(ids_arg, list):
                    err("mnemonics_get_many: 'ids' must be a list")
                    continue
                rows = _get_store().get_many([int(i) for i in ids_arg])
                if rows:
                    lines_gm = [json.dumps(r, default=str, ensure_ascii=False) for r in rows]
                    ok({"content": [{"type": "text", "text": "\n".join(lines_gm)}]})
                else:
                    ok({"content": [{"type": "text", "text": "(no memories found)"}]})

            elif name == "mnemonics_stats_by_ns":
                stats = _get_store().stats_by_ns()
                if stats:
                    lines_s = [
                        f"ns={s['ns']} total={s['total']} pin={s['pinned']} def={s['default']} amb={s['ambient']} "
                        f"oldest={s['oldest']} newest={s['newest']}"
                        for s in stats
                    ]
                    ok({"content": [{"type": "text", "text": "\n".join(lines_s)}]})
                else:
                    ok({"content": [{"type": "text", "text": "(no namespaces)"}]})

            elif name == "mnemonics_namespaces":
                ns_list = _get_store().list_namespaces()
                if ns_list:
                    ok({"content": [{"type": "text", "text": "\n".join(ns_list)}]})
                else:
                    ok({"content": [{"type": "text", "text": "(no namespaces)"}]})

            elif name == "mnemonics_merge_ns":
                src_ns_mg = args.get("src_ns", "").strip()
                dst_ns_mg = args.get("dst_ns", "").strip()
                if not src_ns_mg or not dst_ns_mg:
                    err("mnemonics_merge_ns: 'src_ns' and 'dst_ns' are required")
                    continue
                moved_mg = _get_store().merge_ns(src_ns_mg, dst_ns_mg)
                ok({"content": [{"type": "text", "text": f"Merged {src_ns_mg!r} → {dst_ns_mg!r}: {moved_mg} memories moved."}]})

            elif name == "mnemonics_copy_ns":
                src_ns_c = args.get("src_ns", "").strip()
                dst_ns_c = args.get("dst_ns", "").strip()
                if not src_ns_c or not dst_ns_c:
                    err("mnemonics_copy_ns: 'src_ns' and 'dst_ns' are required")
                    continue
                try:
                    copied_c = _get_store().copy_ns(src_ns_c, dst_ns_c)
                except ValueError as e:
                    err(str(e))
                    continue
                ok({"content": [{"type": "text", "text": f"Copied {src_ns_c!r} → {dst_ns_c!r}: {copied_c} memories copied."}]})

            elif name == "mnemonics_rename_ns":
                old_ns = args.get("old_ns", "").strip()
                new_ns = args.get("new_ns", "").strip()
                if not old_ns or not new_ns:
                    err("mnemonics_rename_ns: 'old_ns' and 'new_ns' are required")
                    continue
                try:
                    moved = _get_store().rename_ns(old_ns, new_ns)
                except ValueError as e:
                    err(str(e))
                    continue
                ok({"content": [{"type": "text", "text": f"Renamed {old_ns!r} → {new_ns!r}: {moved} memories moved."}]})

            elif name == "mnemonics_text_search":
                q = args.get("query", "").strip()
                if not q:
                    err("mnemonics_text_search: 'query' is required")
                    continue
                ns_arg = args.get("ns", "default")
                ns_val = None if ns_arg == "all" else ns_arg
                tier_arg = args.get("tier")
                limit = min(int(args.get("limit", 20)), 100)
                hits = _get_store().text_search(
                    q, ns=ns_val, limit=limit,
                    tier=int(tier_arg) if tier_arg is not None else None,
                )
                if hits:
                    lines_ts = [
                        f"id={h['id']} ns={h['ns']} tier={h['tier']}: {h['text'][:120]}"
                        for h in hits
                    ]
                    ok({"content": [{"type": "text", "text": "\n".join(lines_ts)}]})
                else:
                    ok({"content": [{"type": "text", "text": f"No results for: {q!r}"}]})

            elif name == "mnemonics_import":
                jsonl_str = args.get("jsonl", "")
                if not isinstance(jsonl_str, str) or not jsonl_str.strip():
                    err("mnemonics_import: 'jsonl' must be a non-empty string")
                    continue
                ns_override = args.get("ns")
                tier_override = args.get("tier")
                dry_run = bool(args.get("dry_run", False))
                imported = skipped = 0
                import_errors: list[str] = []
                for lineno, raw_line in enumerate(jsonl_str.splitlines(), start=1):
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue
                    try:
                        obj = json.loads(raw_line)
                    except json.JSONDecodeError as e:
                        import_errors.append(f"line {lineno}: invalid JSON: {e}")
                        skipped += 1
                        continue
                    text = obj.get("text")
                    if not text or not isinstance(text, str):
                        import_errors.append(f"line {lineno}: missing or invalid 'text' field")
                        skipped += 1
                        continue
                    ns = ns_override if ns_override is not None else obj.get("ns", "default")
                    raw_tier = tier_override if tier_override is not None else obj.get("tier", 1)
                    tier = int(raw_tier) if raw_tier in (0, 1, 2) else 1
                    meta = obj.get("meta") or {}
                    summary = obj.get("summary")
                    if not dry_run:
                        n = _ingest(texts=[text], store=_get_store(), ns=ns,
                                    meta=[meta] if meta else None,
                                    summaries=[summary], tier=tier)
                        imported += n
                    else:
                        imported += 1
                summary_parts = [f"imported={imported}", f"skipped={skipped}"]
                if import_errors:
                    summary_parts.append("errors=" + "; ".join(import_errors[:5]))
                if dry_run:
                    summary_parts.insert(0, "[dry-run]")
                ok({"content": [{"type": "text", "text": ", ".join(summary_parts)}]})

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
