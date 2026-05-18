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

    # retrieve
    r = sub.add_parser("retrieve", help="Search memory")
    r.add_argument("query")
    r.add_argument("--ns", default="default")
    r.add_argument("--top-k", type=int, default=5)
    r.add_argument("--no-decay", action="store_true", help="Disable tier-aware decay scoring")
    r.add_argument("--no-hybrid", dest="hybrid", action="store_false", default=True, help="Disable hybrid; fall back to vector-only retrieval")
    r.add_argument("--candidate-k", type=int, default=50, help="Per-channel candidate pool size for hybrid (default 50)")
    r.add_argument("--rerank", action="store_true", help="Cross-encoder rerank via AdaptMem over the widened candidate band (requires adaptmem)")
    r.add_argument("--path", default="~/.mnemonics")

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
    gc.add_argument("--apply", action="store_true", help="Actually delete (default: dry-run, list only)")
    gc.add_argument("--path", default="~/.mnemonics")

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
        n = ingest(texts=[joined], store=store, ns=args.ns, summaries=summaries)
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
        )
        for r in result["results"]:
            tier_label = {0: "pin", 1: "def", 2: "amb"}.get(r["tier"], "?")
            header = (
                f"  [{r['score']:.3f}] "
                f"[raw={r['raw_score']:.3f} decay={r['decay_factor']:.2f} "
                f"boost={r['boost']:.2f} age={r['age_days']:.0f}d "
                f"tier={tier_label}]"
            )
            summary = r.get("summary")
            if summary:
                print(f"{header} {summary[:120]}")
                print(f"      └─ raw: {r['text'][:120]}")
            else:
                print(f"{header} {r['text'][:120]}")

    elif args.cmd == "stats":
        from mnemonics.store import Store
        store = Store(args.path)
        for ns in store.list_namespaces():
            print(f"  {ns}: {store.count(ns)} chunks")

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
        candidates = store.gc_candidates(ns=args.ns, age_days=args.age_days)
        if not candidates:
            print(f"Nothing to GC (tier=2 + age>{args.age_days}d + access_count=0).")
            sys.exit(0)
        for c in candidates[:50]:
            print(f"  id={c['id']:>5} ns={c['ns']:<12} age={c['age_days']}d  {c['preview']}")
        if len(candidates) > 50:
            print(f"  ... and {len(candidates) - 50} more")
        if args.apply:
            n = store.gc(ns=args.ns, age_days=args.age_days)
            print(f"\nDeleted: {n} row(s).")
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
