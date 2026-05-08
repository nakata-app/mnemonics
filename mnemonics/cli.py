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

    # retrieve
    r = sub.add_parser("retrieve", help="Search memory")
    r.add_argument("query")
    r.add_argument("--ns", default="default")
    r.add_argument("--top-k", type=int, default=5)
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
        n = ingest(texts=[" ".join(args.text)], store=store, ns=args.ns)
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
        )
        for r in result["results"]:
            print(f"  [{r['score']:.3f}] {r['text'][:120]}")

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

    else:
        p.print_help()
        sys.exit(1)
