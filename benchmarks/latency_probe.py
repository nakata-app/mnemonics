"""Would a wide candidate band plus a cross-encoder actually survive production?

The BEAM measurements say the missed evidence sits at rank 116 (100K corpus) to
329 (500K corpus), and that the rank scales with corpus size. The obvious fix is
"widen the band and let the cross-encoder sort it out". That fix is only real if
it runs on the machine mnemonics actually runs on.

So this measures wall-clock, not quality: candidate_k against latency, cross
encoder on and off, on a COPY of the live store. Nothing writes to
~/.mnemonics; the copy lives in a temp dir and is deleted.

The number that decides it: if CE at the candidate_k the misses demand costs
seconds per query, "widen the band" is a benchmark result, not a feature. A
memory layer a session calls on every turn has a latency budget in the hundreds
of milliseconds, not tens of seconds.

Queries are drawn from the store's own rows so they are in-distribution without
needing a query log, and they are truncated to a question-like length. This is
an approximation of real queries, not a substitute for a real query log, which
is the honest gap here.

Usage:
  python benchmarks/latency_probe.py
  python benchmarks/latency_probe.py --ks 50,250,1000 --n-queries 15
  python benchmarks/latency_probe.py --ce BAAI/bge-reranker-base
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sqlite3
import statistics
import tempfile
import time
from pathlib import Path

LIVE_STORE = Path.home() / ".mnemonics"


def sample_queries(db: Path, n: int, seed: int = 42) -> list[str]:
    """Take real stored rows and cut them down to question-length probes."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = [r[0] for r in con.execute(
            "select text from memories where text is not null limit 5000") if r[0]]
    finally:
        con.close()
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows:
        words = r.split()
        if len(words) < 8:
            continue
        out.append(" ".join(words[:14]))
        if len(out) >= n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="50,250,1000,2000")
    ap.add_argument("--n-queries", type=int, default=12)
    ap.add_argument("--ce", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--ce-max-length", type=int, default=512)
    ap.add_argument("--ce-batch", type=int, default=8)
    ap.add_argument("--ns", default="sessions")
    ap.add_argument("--out", default="benchmarks/results/latency_probe.json")
    args = ap.parse_args()

    if not (LIVE_STORE / "memories.db").exists():
        print(f"live store not found at {LIVE_STORE}")
        return 2

    os.environ["MNEMONICS_RERANK_MODEL"] = args.ce
    os.environ["MNEMONICS_RERANK_MAX_LENGTH"] = str(args.ce_max_length)
    os.environ["MNEMONICS_RERANK_BATCH_SIZE"] = str(args.ce_batch)

    ks = [int(x) for x in args.ks.split(",")]
    tmp = Path(tempfile.mkdtemp(prefix="latency-probe-"))
    try:
        # Copy, never open the live store: a probe must not be able to mutate
        # 10k+ real memories, and sqlite locks would fight the running MCP.
        for name in ("memories.db", "embed_manifest.json"):
            src = LIVE_STORE / name
            if src.exists():
                shutil.copy2(src, tmp / name)
        # One HNSW index file per namespace, so copy them all rather than
        # guessing which one --ns maps to.
        for extra in LIVE_STORE.glob("index_*.bin"):
            shutil.copy2(extra, tmp / extra.name)
        copied = sorted(p.name for p in tmp.iterdir())
        print(f"store copy: {tmp}  files={copied}", flush=True)

        queries = sample_queries(tmp / "memories.db", args.n_queries)
        print(f"{len(queries)} in-distribution probe queries", flush=True)

        from mnemonics.retrieve import retrieve
        from mnemonics.store import Store

        store = Store(str(tmp))
        n_rows = sqlite3.connect(f"file:{tmp/'memories.db'}?mode=ro", uri=True) \
            .execute("select count(*) from memories").fetchone()[0]
        print(f"store rows: {n_rows}", flush=True)

        results = []
        for rerank in (False, True):
            for k in ks:
                # Warm-up, discarded. The first configuration measured pays for
                # the HNSW index load, the SQLite page cache and (with rerank)
                # the cross-encoder weights, and that cost landed entirely in
                # whichever cell ran first: an earlier run reported p95 4551ms
                # at k=50 against 69.7ms at k=250, which is backwards and was
                # warm-up, not latency.
                for q in queries[:min(3, len(queries))]:
                    try:
                        retrieve(query=q, store=store, ns=args.ns, top_k=10,
                                 candidate_k=k, hybrid=True, rerank=rerank)
                    except Exception:
                        break

                times: list[float] = []
                failed = None
                for q in queries:
                    t0 = time.perf_counter()
                    try:
                        retrieve(query=q, store=store, ns=args.ns, top_k=10,
                                 candidate_k=k, hybrid=True, rerank=rerank)
                    except Exception as e:
                        failed = f"{type(e).__name__}: {e}"
                        break
                    times.append(time.perf_counter() - t0)
                if failed:
                    print(f"  rerank={rerank} k={k}: FAILED {failed}", flush=True)
                    results.append({"rerank": rerank, "candidate_k": k,
                                    "error": failed})
                    continue
                times.sort()
                row = {
                    "rerank": rerank, "candidate_k": k, "n": len(times),
                    "p50_ms": round(statistics.median(times) * 1000, 1),
                    "p95_ms": round(times[int(len(times) * 0.95)] * 1000, 1),
                    "mean_ms": round(statistics.mean(times) * 1000, 1),
                }
                results.append(row)
                print(f"  rerank={str(rerank):<5} k={k:<5} "
                      f"p50={row['p50_ms']:>9.1f}ms  p95={row['p95_ms']:>9.1f}ms",
                      flush=True)

        out = {
            "store_rows": n_rows, "ns": args.ns, "ce_model": args.ce,
            "ce_max_length": args.ce_max_length, "ce_batch": args.ce_batch,
            "n_queries": len(queries), "results": results,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))

        print("\n=== VERDICT ===")
        ce_rows = [r for r in results if r["rerank"] and "p50_ms" in r]
        for r in ce_rows:
            v = ("interactive" if r["p50_ms"] < 500 else
                 "sluggish" if r["p50_ms"] < 2000 else "not viable")
            print(f"  CE at candidate_k={r['candidate_k']:<5} "
                  f"p50={r['p50_ms']:>9.1f}ms  -> {v}")
        print("  a memory layer called every turn has a few hundred ms to spend")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
