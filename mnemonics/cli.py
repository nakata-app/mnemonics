"""mnemonics CLI."""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    p = argparse.ArgumentParser(prog="mnemonics")
    sub = p.add_subparsers(dest="cmd")

    # serve
    s = sub.add_parser("serve", help="Start REST server")
    s.add_argument("--port", type=int, default=7810)
    s.add_argument("--path", default="~/.mnemonics")

    # mcp
    sub.add_parser("mcp", help="Start MCP stdio server")

    # ingest
    i = sub.add_parser("ingest", help="Add text to memory")
    i.add_argument("text", nargs="+")
    i.add_argument("--ns", default="default")
    i.add_argument("--path", default="~/.mnemonics")
    i.add_argument("--dedup", action="store_true", help="Check for near-duplicate memories before saving")
    i.add_argument("--dedup-threshold", type=float, default=0.92, help="Cosine threshold for near-duplicate (default 0.92)")
    i.add_argument("--force-new", action="store_true", help="With --dedup: save anyway, don't prompt")
    i.add_argument("--skip-similar", action="store_true", help="With --dedup: cancel ingest if a near-duplicate exists, don't prompt")
    i.add_argument(
        "--summary",
        default=None,
        help="Optional one-line gist stored alongside the raw text. Searched by BM25 in addition to the chunk body.",
    )
    i.add_argument(
        "--meta",
        default=None,
        help="JSON object to attach as metadata to every chunk (e.g. '{\"tag\":\"work\"}').",
    )

    # retrieve
    r = sub.add_parser("retrieve", help="Search memory")
    r.add_argument("query")
    r.add_argument("--ns", default="default")
    r.add_argument("--top-k", type=int, default=5)
    r.add_argument("--no-decay", action="store_true", help="Disable tier-aware decay scoring")
    r.add_argument("--no-hybrid", dest="hybrid", action="store_false", default=True, help="Disable hybrid; fall back to vector-only retrieval")
    r.add_argument("--candidate-k", type=int, default=50, help="Per-channel candidate pool size for hybrid (default 50)")
    r.add_argument("--rerank", action="store_true", help="Cross-encoder rerank via AdaptMem over the widened candidate band (requires adaptmem)")
    r.add_argument("--min-tier", type=int, choices=[0, 1, 2], default=None, help="Only return memories at or above this tier (0=pinned, 1=default, 2=ambient)")
    r.add_argument("--max-tier", type=int, choices=[0, 1, 2], default=None, help="Only return memories at or below this tier")
    r.add_argument("--path", default="~/.mnemonics")

    # bm25 (pure keyword search)
    bm = sub.add_parser("bm25", help="Pure BM25 keyword search (no vector encoding)")
    bm.add_argument("query")
    bm.add_argument("--ns", default="default")
    bm.add_argument("--top-k", type=int, default=5)
    bm.add_argument("--min-tier", type=int, choices=[0, 1, 2], default=None, help="Only return memories at or above this tier")
    bm.add_argument("--max-tier", type=int, choices=[0, 1, 2], default=None, help="Only return memories at or below this tier")
    bm.add_argument("--path", default="~/.mnemonics")

    # stats
    st = sub.add_parser("stats", help="Show memory stats")
    st.add_argument("--path", default="~/.mnemonics")

    # pin
    pin = sub.add_parser("pin", help="Pin a memory (tier=0, no decay)")
    pin.add_argument("memory_id", type=int)
    pin.add_argument("--path", default="~/.mnemonics")

    # tier
    tr = sub.add_parser("tier", help="Set memory tier (0=pinned, 1=default, 2=ambient)")
    tr.add_argument("memory_id", type=int)
    tr.add_argument("level", type=int, choices=[0, 1, 2])
    tr.add_argument("--path", default="~/.mnemonics")

    # gc
    gc = sub.add_parser("gc", help="Garbage-collect ambient (tier 2) memories never accessed and older than N days")
    gc.add_argument("--ns", default=None, help="Limit GC to one namespace (default: all)")
    gc.add_argument("--age-days", type=int, default=30, help="Minimum age before a row is eligible (default: 30)")
    gc.add_argument("--tier", type=int, choices=[1, 2], default=2, help="Tier to GC: 2=ambient (default), 1=default")
    gc.add_argument("--apply", action="store_true", help="Actually delete (default: dry-run, list only)")
    gc.add_argument("--path", default="~/.mnemonics")

    # doctor
    dr = sub.add_parser("doctor", help="Health check: DB integrity, index counts, orphan indexes")
    dr.add_argument("--path", default="~/.mnemonics")
    dr.add_argument("--json", dest="json_out", action="store_true", help="Output as JSON")
    dr.add_argument("--fix", action="store_true", help="Auto-repair orphan vectors and orphan index files")

    # rebuild-index
    rb = sub.add_parser("rebuild-index", help="Rebuild hnswlib index for a namespace from SQL (removes orphan vectors)")
    rb.add_argument("--ns", required=True, help="Namespace whose index to rebuild")
    rb.add_argument("--path", default="~/.mnemonics")

    # forget
    ft = sub.add_parser("forget", help="Bulk-delete memories in a namespace (default: dry-run)")
    ft.add_argument("--ns", required=True, help="Namespace to forget")
    ft.add_argument("--before", default=None, metavar="YYYY-MM-DD", help="Only delete rows created before this date")
    ft.add_argument("--tier", type=int, choices=[0, 1, 2], default=None, help="Limit to tier (default: excludes pinned/tier-0)")
    ft.add_argument("--apply", action="store_true", help="Actually delete (default: dry-run)")
    ft.add_argument("--path", default="~/.mnemonics")

    # count
    cn = sub.add_parser("count", help="Count memories in a namespace (or all namespaces)")
    cn.add_argument("--ns", default=None, help="Namespace to count (default: all namespaces)")
    cn.add_argument("--path", default="~/.mnemonics")

    # set-tier-many
    stm = sub.add_parser("set-tier-many", help="Set tier for multiple memories by ID")
    stm.add_argument("level", type=int, choices=[0, 1, 2], help="Target tier: 0=pinned, 1=default, 2=ambient")
    stm.add_argument("ids", type=int, nargs="+", metavar="ID")
    stm.add_argument("--path", default="~/.mnemonics")

    # get
    gt = sub.add_parser("get", help="Fetch a single memory by ID")
    gt.add_argument("memory_id", type=int)
    gt.add_argument("--json", dest="json_out", action="store_true", help="Output as JSON")
    gt.add_argument("--path", default="~/.mnemonics")

    # get-many
    gm = sub.add_parser("get-many", help="Fetch multiple memories by ID (space-separated)")
    gm.add_argument("ids", type=int, nargs="+", metavar="ID")
    gm.add_argument("--path", default="~/.mnemonics")

    # delete-ids
    di = sub.add_parser("delete-ids", help="Delete specific memories by ID")
    di.add_argument("ids", type=int, nargs="+", metavar="ID")
    di.add_argument("--path", default="~/.mnemonics")

    # search-meta
    sm = sub.add_parser("search-meta", help="Find memories matching metadata key=value filters")
    sm.add_argument("filters", nargs="+", metavar="KEY=VALUE", help="Metadata filters (e.g. tag=important)")
    sm.add_argument("--ns", default="default")
    sm.add_argument("--limit", type=int, default=20)
    sm.add_argument("--path", default="~/.mnemonics")

    # update-meta
    um = sub.add_parser("update-meta", help="Replace metadata of a memory with JSON")
    um.add_argument("memory_id", type=int)
    um.add_argument("meta_json", help="New metadata as a JSON object string (e.g. '{\"tag\":\"x\"}')")
    um.add_argument("--path", default="~/.mnemonics")

    # set-summary
    ss = sub.add_parser("set-summary", help="Add or update the summary field of a memory")
    ss.add_argument("memory_id", type=int)
    ss.add_argument("summary", nargs="?", default=None, help="Summary text (omit to clear)")
    ss.add_argument("--path", default="~/.mnemonics")

    # list
    ls = sub.add_parser("list", help="Browse memories in a namespace, newest first")
    ls.add_argument("--ns", default="default", help="Namespace to list (default: 'default')")
    ls.add_argument("--limit", type=int, default=20, help="Max rows to show (default: 20)")
    ls.add_argument("--offset", type=int, default=0, help="Pagination offset (default: 0)")
    ls.add_argument("--tier", type=int, choices=[0, 1, 2], default=None, help="Filter to tier: 0=pinned, 1=default, 2=ambient")
    ls.add_argument("--json", dest="json_out", action="store_true", help="Output as JSONL (one object per line)")
    ls.add_argument("--path", default="~/.mnemonics")

    # sync export / import (peer transport)
    sy = sub.add_parser("sync", help="Export/import portable transport archive between stores")
    sy_sub = sy.add_subparsers(dest="sync_cmd")

    se = sy_sub.add_parser("export", help="Write a portable .sync.tar.gz of this store")
    se.add_argument("--out", default=None, help="Output path (default: ~/.mnemonics-sync/<timestamp>.sync.tar.gz)")
    se.add_argument("--path", default="~/.mnemonics", help="Store directory to export")

    si = sy_sub.add_parser("import", help="Merge a peer's .sync.tar.gz into this store")
    si.add_argument("archive", help="Path to the .sync.tar.gz produced by `mnemonics sync export`")
    si.add_argument("--path", default="~/.mnemonics", help="Destination store directory")
    si.add_argument(
        "--strategy",
        default="skip-existing",
        choices=["skip-existing", "force-new-id", "overwrite"],
        help="Conflict policy when an incoming row's text hash matches an existing row in the same namespace",
    )
    si.add_argument("--only-ns", default=None, help="Restrict import to a single namespace")

    # backup
    bk = sub.add_parser("backup", help="Bundle the store into a .tar.gz")
    bk.add_argument("--out", default=None, help="Output archive path (default: ~/.mnemonics-backups/YYYY-MM-DD_HHMMSS.tar.gz)")
    bk.add_argument("--path", default="~/.mnemonics", help="Store directory to back up")

    # export-jsonl
    ej = sub.add_parser("export-jsonl", help="Dump all memories as JSONL (one JSON object per line)")
    ej.add_argument("--ns", default=None, help="Namespace to export (default: all namespaces)")
    ej.add_argument("--tier", type=int, choices=[0, 1, 2], default=None, help="Filter by tier")
    ej.add_argument("--meta-filter", action="append", default=None, metavar="KEY=VALUE",
                    help="Filter by metadata key=value (repeat for multiple filters, AND logic)")
    ej.add_argument("--out", default=None, help="Output file path (default: stdout)")
    ej.add_argument("--path", default="~/.mnemonics", help="Store directory")

    # restore
    rs = sub.add_parser("restore", help="Extract a backup archive into a store directory")
    rs.add_argument("archive", help="Path to the .tar.gz produced by `mnemonics backup`")
    rs.add_argument("--path", default="~/.mnemonics", help="Destination store directory")
    rs.add_argument("--force", action="store_true", help="Overwrite an existing non-empty store")

    # encrypt-db
    ec = sub.add_parser(
        "encrypt-db",
        help="Migrate a plain memories.db to an encrypted SQLCipher DB (one-shot)",
    )
    ec.add_argument("--path", default="~/.mnemonics", help="Store directory")
    ec.add_argument(
        "--key",
        default=None,
        help="64-char hex key. Omit to auto-generate and store in the system keyring.",
    )
    ec.add_argument(
        "--no-keyring",
        action="store_true",
        help="Skip storing the key in the system keyring (print to stdout instead).",
    )
    ec.add_argument(
        "--force",
        action="store_true",
        help="Proceed even if mnemonics MCP processes are running (NOT recommended).",
    )

    # eval
    ev = sub.add_parser("eval", help="Retrieval eval: MRR / R@5 / R@10 / NDCG@10")
    ev.add_argument("--corpus", required=True, help="Path to corpus.jsonl ({id, text})")
    ev.add_argument("--queries", required=True, help="Path to queries.jsonl ({query, relevant_id})")
    ev.add_argument(
        "--encoder",
        action="append",
        default=None,
        help="minilm | adaptmem | <HF id>. Repeat to compare multiple encoders.",
    )
    ev.add_argument("--model-path", default=None, help="Required when an encoder is 'adaptmem' (or set MNEMONICS_ADAPTMEM_PATH)")
    ev.add_argument(
        "--method",
        action="append",
        default=None,
        choices=["vector", "hybrid"],
        help="Retrieval method (default: vector). Repeat to compare vector vs hybrid.",
    )
    ev.add_argument("--candidate-k", type=int, default=50, help="Per-channel pool size when method=hybrid (default 50)")
    ev.add_argument("--top-k", type=int, default=10)
    ev.add_argument("--out", default=None, help="Directory to write per-encoder JSON results")

    args = p.parse_args()

    if args.cmd == "serve":
        import os
        os.environ["MNEMONICS_PATH"] = args.path
        from mnemonics.server import serve
        serve(port=args.port)

    elif args.cmd == "mcp":
        from mnemonics.server import serve
        serve(mcp=True)

    elif args.cmd == "ingest":
        from mnemonics.store import Store
        from mnemonics.ingest import ingest
        store = Store(args.path)
        joined = " ".join(args.text)
        if args.dedup:
            from mnemonics.dedup import find_similar
            matches = find_similar(joined, store=store, ns=args.ns, threshold=args.dedup_threshold)
            if matches:
                print(f"Found {len(matches)} near-duplicate(s) in ns={args.ns} (>= {args.dedup_threshold}):")
                for m in matches:
                    preview = m["text"][:120].replace("\n", " ")
                    print(f"  [sim={m['similarity']:.3f}] id={m['id']}: {preview}")
                if args.skip_similar:
                    print("Skip-similar flag set, not ingesting.")
                    sys.exit(0)
                if args.force_new:
                    decision = "n"
                elif sys.stdin.isatty():
                    decision = input("Save as new memory anyway? [y/N] ").strip().lower()
                    decision = "n" if decision in ("y", "yes") else "c"
                else:
                    # Non-interactive: refuse to ingest silently. Caller decides.
                    print("Non-interactive run, not ingesting. Re-run with --force-new or --skip-similar.")
                    sys.exit(0)
                if decision == "c":
                    print("Cancelled.")
                    sys.exit(0)
        summaries = [args.summary] if args.summary else None
        meta_dict: dict | None = None
        if args.meta is not None:
            try:
                meta_dict = json.loads(args.meta)
            except json.JSONDecodeError as e:
                print(f"--meta: invalid JSON: {e}", file=sys.stderr)
                sys.exit(2)
            if not isinstance(meta_dict, dict):
                print("--meta: must be a JSON object", file=sys.stderr)
                sys.exit(2)
        metas = [meta_dict] if meta_dict is not None else None
        n = ingest(texts=[joined], store=store, ns=args.ns, summaries=summaries, meta=metas)
        print(f"Stored {n} chunk(s).")

    elif args.cmd == "retrieve":
        from mnemonics.store import Store
        from mnemonics.retrieve import retrieve
        store = Store(args.path)
        result = retrieve(
            query=args.query,
            store=store,
            ns=args.ns,
            top_k=args.top_k,
            decay=not args.no_decay,
            hybrid=args.hybrid,
            candidate_k=args.candidate_k,
            rerank=args.rerank,
            min_tier=args.min_tier,
            max_tier=args.max_tier,
        )
        for r in result["results"]:
            tier_label = {0: "pin", 1: "def", 2: "amb"}.get(r["tier"], "?")
            header = (
                f"  [{r['score']:.3f}] "
                f"[id={r['id']} raw={r['raw_score']:.3f} decay={r['decay_factor']:.2f} "
                f"boost={r['boost']:.2f} age={r['age_days']:.0f}d "
                f"tier={tier_label}]"
            )
            summary = r.get("summary")
            if summary:
                print(f"{header} {summary[:120]}")
                print(f"      └─ raw: {r['text'][:120]}")
            else:
                print(f"{header} {r['text'][:120]}")

    elif args.cmd == "bm25":
        from mnemonics.store import Store
        store = Store(args.path)
        hits = store.search_bm25(args.query, ns=args.ns, top_k=args.top_k,
                                   min_tier=args.min_tier, max_tier=args.max_tier)
        if not hits:
            print(f"No BM25 results for {args.query!r} in ns={args.ns!r}")
        else:
            tier_label = {0: "pin", 1: "def", 2: "amb"}
            for r in hits:
                tl = tier_label.get(r["tier"], "?")
                snippet = r["text"][:120].replace("\n", " ")
                line = f"  [{r['score']:.3f}] [{tl}] id={r['id']}  {snippet}"
                print(line)
                if r.get("summary"):
                    print(f"           summary: {r['summary']}")

    elif args.cmd == "count":
        from mnemonics.store import Store
        store = Store(args.path)
        n = store.count(ns=args.ns)
        label = f"ns={args.ns!r}" if args.ns is not None else "all namespaces"
        print(f"{n} memories ({label})")

    elif args.cmd == "set-tier-many":
        from mnemonics.store import Store
        store = Store(args.path)
        n = store.update_tier_many(args.ids, args.level)
        label = {0: "pinned", 1: "default", 2: "ambient"}[args.level]
        print(f"Updated {n} of {len(args.ids)} ID(s) to tier {args.level} ({label}).")

    elif args.cmd == "get":
        from mnemonics.store import Store
        store = Store(args.path)
        row = store.get(args.memory_id)
        if row is None:
            print(f"id={args.memory_id} not found")
            sys.exit(1)
        if args.json_out:
            print(json.dumps(row, default=str, ensure_ascii=False))
        else:
            tier_label = {0: "pinned", 1: "default", 2: "ambient"}.get(row["tier"], "?")
            print(f"id={row['id']}  ns={row['ns']}  tier={tier_label}  created={row['created']}")
            if row["summary"]:
                print(f"summary: {row['summary']}")
            print(f"text: {row['text']}")

    elif args.cmd == "get-many":
        from mnemonics.store import Store
        store = Store(args.path)
        rows = store.get_many(args.ids)
        tier_label = {0: "pinned", 1: "default", 2: "ambient"}
        for row in rows:
            tl = tier_label.get(row["tier"], "?")
            print(f"id={row['id']}  ns={row['ns']}  tier={tl}  {row['text'][:120]}")
        if not rows:
            print("No memories found for those IDs.")

    elif args.cmd == "delete-ids":
        from mnemonics.store import Store
        store = Store(args.path)
        n = store.delete_many(args.ids)
        print(f"Deleted {n} of {len(args.ids)} requested ID(s).")

    elif args.cmd == "search-meta":
        from mnemonics.store import Store
        store = Store(args.path)
        filters: dict[str, str] = {}
        for kv in args.filters:
            if "=" not in kv:
                print(f"Invalid filter '{kv}': must be KEY=VALUE", file=sys.stderr)
                sys.exit(2)
            k, _, v = kv.partition("=")
            filters[k.strip()] = v.strip()
        results = store.search_by_meta(filters, ns=args.ns, limit=args.limit)
        if not results:
            print(f"No results for {filters} in ns={args.ns!r}.")
        else:
            tier_label = {0: "pinned", 1: "default", 2: "ambient"}
            for r in results:
                tl = tier_label.get(r["tier"], "?")
                print(f"id={r['id']}  tier={tl}  {r['text'][:120]}")

    elif args.cmd == "update-meta":
        from mnemonics.store import Store
        store = Store(args.path)
        try:
            meta = json.loads(args.meta_json)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}", file=sys.stderr)
            sys.exit(2)
        if not isinstance(meta, dict):
            print("meta_json must be a JSON object ({})", file=sys.stderr)
            sys.exit(2)
        ok_ = store.update_meta(args.memory_id, meta)
        print(f"Meta updated for id={args.memory_id}" if ok_ else f"id={args.memory_id} not found")
        sys.exit(0 if ok_ else 1)

    elif args.cmd == "set-summary":
        from mnemonics.store import Store
        store = Store(args.path)
        ok_ = store.update_summary(args.memory_id, args.summary)
        if ok_:
            action = "cleared" if args.summary is None else "updated"
            print(f"Summary {action} for id={args.memory_id}")
        else:
            print(f"id={args.memory_id} not found")
            sys.exit(1)

    elif args.cmd == "list":
        from mnemonics.store import Store
        store = Store(args.path)
        rows = store.list_memories(ns=args.ns, limit=args.limit, offset=args.offset, tier=args.tier)
        if not rows:
            if args.json_out:
                print("[]")
            else:
                print(f"No memories in ns={args.ns!r} (offset={args.offset}).")
        else:
            if args.json_out:
                for r in rows:
                    print(json.dumps(r, default=str, ensure_ascii=False))
            else:
                tier_label = {0: "pin", 1: "def", 2: "amb"}
                print(f"ns={args.ns!r}  offset={args.offset}  showing {len(rows)} row(s)")
                for r in rows:
                    snippet = (r["text"] or "")[:120].replace("\n", " ")
                    tl = tier_label.get(r["tier"], "?")
                    summary = f"  [{r['summary'][:60]}]" if r["summary"] else ""
                    print(f"  [{r['id']}] {tl} {r['created']}  {snippet}{summary}")

    elif args.cmd == "stats":
        from mnemonics.store import Store
        store = Store(args.path)
        rows = store._db.execute(
            "SELECT ns, tier, COUNT(*) FROM memories GROUP BY ns, tier ORDER BY ns, tier"
        ).fetchall()
        ns_data: dict = {}
        for ns_name, tier, cnt in rows:
            ns_data.setdefault(ns_name, {})[tier] = cnt
        if not ns_data:
            print("  (empty)")
        for ns_name, tiers in sorted(ns_data.items()):
            total = sum(tiers.values())
            pin, def_, amb = tiers.get(0, 0), tiers.get(1, 0), tiers.get(2, 0)
            print(f"  {ns_name}: {total} chunks  (pin={pin} def={def_} amb={amb})")

    elif args.cmd == "pin":
        from mnemonics.store import Store
        store = Store(args.path)
        ok_ = store.pin(args.memory_id)
        print(f"Pinned id={args.memory_id}" if ok_ else f"id={args.memory_id} not found")
        sys.exit(0 if ok_ else 1)

    elif args.cmd == "tier":
        from mnemonics.store import Store
        store = Store(args.path)
        ok_ = store.set_tier(args.memory_id, args.level)
        label = {0: "pinned", 1: "default", 2: "ambient"}[args.level]
        print(f"id={args.memory_id} -> tier {args.level} ({label})" if ok_ else f"id={args.memory_id} not found")
        sys.exit(0 if ok_ else 1)

    elif args.cmd == "gc":
        from mnemonics.store import Store
        store = Store(args.path)
        candidates = store.gc_candidates(ns=args.ns, age_days=args.age_days, tier=args.tier)
        tier_desc = "ambient (never accessed)" if args.tier == 2 else "default"
        if not candidates:
            print(f"Nothing to GC (tier={args.tier}/{tier_desc} + age>{args.age_days}d).")
            sys.exit(0)
        for c in candidates[:50]:
            print(f"  id={c['id']:>5} ns={c['ns']:<12} age={c['age_days']}d  {c['preview']}")
        if len(candidates) > 50:
            print(f"  ... and {len(candidates) - 50} more")
        if args.apply:
            n = store.gc(ns=args.ns, age_days=args.age_days, tier=args.tier)
            print(f"\nDeleted: {n} row(s).")
        else:
            print(f"\nDry-run, {len(candidates)} candidate(s). Re-run with --apply to delete.")

    elif args.cmd == "doctor":
        from mnemonics.store import Store
        store = Store(args.path)

        if args.fix:
            fix_report = store.repair()
            fixed_v = fix_report["orphan_vectors_fixed"]
            fixed_i = fix_report["orphan_indexes_removed"]
            missing = fix_report["missing_vectors_reported"]
            if fixed_v:
                for item in fixed_v:
                    if "error" in item:
                        print(f"  ✗ {item['ns']}: {item['error']}")
                    else:
                        print(f"  ✓ {item['ns']}: removed {item['removed']} orphan vector(s)")
            if fixed_i:
                for path in fixed_i:
                    if isinstance(path, dict):
                        print(f"  ✗ {path['path']}: {path['error']}")
                    else:
                        print(f"  ✓ removed orphan index: {path}")
            if missing:
                for item in missing:
                    print(f"  ⚠ {item['ns']}: {item['missing']} missing vector(s) — run: mnem rebuild-index --ns {item['ns']!r}")
            if not fixed_v and not fixed_i and not missing:
                print("✓ Nothing to fix")
            sys.exit(0)

        report = store.health_check()

        if args.json_out:
            print(json.dumps(report, indent=2))
            sys.exit(0)

        # Human-readable output
        integrity = report["db_integrity"]
        integrity_ok = integrity == "ok"
        wal_mb = report["wal_size"] / 1_048_576
        print(f"DB integrity : {'OK' if integrity_ok else 'FAIL: ' + integrity}")
        print(f"WAL size     : {wal_mb:.1f} MB{'  ⚠ consider checkpoint' if wal_mb > 50 else ''}")
        print()

        issues = 0
        print(f"{'Namespace':<28} {'SQL':>6} {'IDX':>6} {'Soft-del':>8}  Status")
        print("-" * 60)
        for ns in report["namespaces"]:
            sql = ns["sql_count"]
            idx = ns["idx_count"]
            sd = ns["soft_deleted"]
            mv = ns.get("missing_vectors", 0)
            if ns["idx_missing"]:
                status = "no index"
            elif mv > 0:
                status = f"⚠ {mv} missing vector(s)"
                issues += 1
            elif sd > 0:
                status = f"⚠ {sd} orphan vector(s)"
                issues += 1
            elif ns.get("capacity_warning"):
                status = f"⚠ {ns['usage_pct']}% full — rebuild-index recommended"
                issues += 1
            else:
                status = "OK"
            idx_str = str(idx) if idx is not None else "—"
            print(f"  {ns['ns']:<26} {sql:>6} {idx_str:>6} {sd:>8}  {status}")

        if report["orphan_indexes"]:
            print()
            print("Orphan indexes (no DB rows):")
            for o in report["orphan_indexes"]:
                size_mb = o["size"] / 1_048_576
                print(f"  {o['ns']:<30} {size_mb:.1f} MB  → rm \"{o['path']}\"")
                issues += 1

        print()
        if issues == 0:
            print(f"✓ All OK ({len(report['namespaces'])} namespaces)")
        else:
            print(f"⚠ {issues} issue(s) found")
        sys.exit(0 if issues == 0 else 1)

    elif args.cmd == "rebuild-index":
        from mnemonics.store import Store
        store = Store(args.path)
        old_n, new_n = store.rebuild_ns_index(args.ns)
        removed = old_n - new_n
        print(f"ns={args.ns}: {old_n} → {new_n} vectors  ({removed} orphan(s) removed)")
        sys.exit(0)

    elif args.cmd == "forget":
        from mnemonics.store import Store
        store = Store(args.path)
        candidates = store.forget_candidates(ns=args.ns, before=args.before, tier=args.tier)
        tier_label = {0: "pin", 1: "def", 2: "amb"}
        if not candidates:
            filters = f"ns={args.ns}"
            if args.before:
                filters += f" before={args.before}"
            if args.tier is not None:
                filters += f" tier={args.tier}"
            print(f"Nothing to forget ({filters}).")
            sys.exit(0)
        for c in candidates[:50]:
            print(f"  id={c['id']:>5} tier={tier_label.get(c['tier'], '?')} created={c['created'][:10]}  {c['preview']}")
        if len(candidates) > 50:
            print(f"  ... and {len(candidates) - 50} more")
        if args.tier is None:
            pinned_count = store._db.execute(
                "SELECT count(*) FROM memories WHERE ns=? AND tier=0", (args.ns,)
            ).fetchone()[0]
            if pinned_count:
                print(f"\n  Note: {pinned_count} pinned (tier-0) row(s) excluded — pass --tier 0 to include.")
        if args.apply:
            n = store.forget(ns=args.ns, before=args.before, tier=args.tier)
            print(f"\nDeleted: {n} row(s) from ns={args.ns}.")
        else:
            print(f"\nDry-run, {len(candidates)} candidate(s). Re-run with --apply to delete.")

    elif args.cmd == "sync":
        if args.sync_cmd == "export":
            from mnemonics.sync import export_store
            out_path = export_store(store_path=args.path, out=args.out)
            print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")
        elif args.sync_cmd == "import":
            from mnemonics.sync import import_store
            summary = import_store(
                archive=args.archive,
                store_path=args.path,
                strategy=args.strategy,
                only_ns=args.only_ns,
            )
            print(f"imported={summary['imported']} skipped={summary['skipped']} overwritten={summary['overwritten']}")
        else:
            sy.print_help()
            sys.exit(1)

    elif args.cmd == "export-jsonl":
        from mnemonics.store import Store
        import sys as _sys
        store = Store(args.path)
        # Parse --meta-filter KEY=VALUE pairs
        meta_filters: dict[str, str] = {}
        if args.meta_filter:
            for kv in args.meta_filter:
                if "=" not in kv:
                    print(f"Invalid --meta-filter '{kv}': must be KEY=VALUE", file=sys.stderr)
                    sys.exit(2)
                k, _, v = kv.partition("=")
                meta_filters[k.strip()] = v.strip()
        where_parts = ["1=1"]
        params: list = []
        if args.ns is not None:
            where_parts.append("ns = ?")
            params.append(args.ns)
        if args.tier is not None:
            where_parts.append("tier = ?")
            params.append(args.tier)
        for mkey, mval in meta_filters.items():
            where_parts.append(f"json_extract(meta, '$.{mkey}') = ?")
            params.append(mval)
        where = " AND ".join(where_parts)
        rows = store._db.execute(
            f"SELECT id, ns, text, summary, meta, created, tier, last_accessed, access_count "
            f"FROM memories WHERE {where} ORDER BY id",
            params,
        ).fetchall()
        out_fd = open(args.out, "w", encoding="utf-8") if args.out else _sys.stdout
        try:
            for row in rows:
                obj = {
                    "id": row[0], "ns": row[1], "text": row[2],
                    "summary": row[3], "meta": json.loads(row[4]),
                    "created": row[5], "tier": row[6],
                    "last_accessed": row[7], "access_count": row[8],
                }
                out_fd.write(json.dumps(obj, ensure_ascii=False) + "\n")
        finally:
            if args.out:
                out_fd.close()
        if args.out:
            print(f"Exported {len(rows)} memories to {args.out}", file=_sys.stderr)

    elif args.cmd == "backup":
        from mnemonics.backup import backup
        out_path = backup(store_path=args.path, out=args.out)
        size = out_path.stat().st_size
        print(f"Wrote {out_path} ({size:,} bytes)")

    elif args.cmd == "restore":
        from mnemonics.backup import restore
        try:
            written = restore(archive=args.archive, store_path=args.path, force=args.force)
        except FileExistsError as exc:
            print(f"Refusing to overwrite: {exc}", file=sys.stderr)
            sys.exit(2)
        if not written:
            print("Archive contained no restorable files.")
        else:
            for name in written:
                print(f"  + {name}")
            print(f"Restored {len(written)} file(s) into {args.path}")

    elif args.cmd == "encrypt-db":
        from mnemonics.migrate import encrypt_db
        try:
            encrypt_db(
                path=args.path,
                key_hex=args.key,
                store_in_keyring=not args.no_keyring,
                force=args.force,
            )
        except RuntimeError as exc:
            print(f"encrypt-db: {exc}", file=sys.stderr)
            sys.exit(2)

    elif args.cmd == "eval":
        import json as _json
        from pathlib import Path as _Path
        from mnemonics.eval import run_eval, compare_table

        encoders = args.encoder or ["minilm"]
        methods = args.method or ["vector"]
        results: dict[str, dict] = {}
        for enc in encoders:
            for method in methods:
                print(f"[eval] encoder={enc} method={method}")
                r = run_eval(
                    corpus_path=args.corpus,
                    queries_path=args.queries,
                    encoder=enc,
                    model_path=args.model_path,
                    top_k=args.top_k,
                    method=method,
                    candidate_k=args.candidate_k,
                )
                results[r["encoder"]] = r
                if args.out:
                    out_dir = _Path(args.out).expanduser()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    slug = r["encoder"].replace(":", "_").replace("/", "_").replace("+", "_")
                    with open(out_dir / f"{slug}.json", "w") as f:
                        _json.dump(r, f, indent=2, ensure_ascii=False)
                    print(f"[eval] wrote {out_dir / f'{slug}.json'}")
        print()
        print(compare_table(results))

    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
