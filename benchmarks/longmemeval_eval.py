"""Mnemonics-vs-MemPalace eval on LongMemEval-500.

Compares three retrievers on the same dataset (Sebastian Smolny's
longmemeval_s_cleaned, 500 questions × ~50 sessions/question):

  - MemPalace baseline    : IDF-weighted entity-graph overlap (R@1=0.354
                            already measured in
                            adaptmem/benchmarks/structural_memory_eval/
                            entity_graph_result.json — reused here)
  - Mnemonics no-rerank   : retrieve(rerank=False) — hybrid HNSW + BM25
                            RRF + tier decay
  - Mnemonics rerank=True : adds AdaptMem cross-encoder rerank over the
                            candidate band

For each question, we:
  1. ingest every haystack session into a fresh namespace
  2. retrieve top_k for the query
  3. for each k in {1, 5, 10}, count hit if any retrieved row's
     source session is in answer_session_ids

Usage:
  python benchmarks/longmemeval_eval.py            # 50-q subset, both modes
  python benchmarks/longmemeval_eval.py --n 500    # full set
  python benchmarks/longmemeval_eval.py --mode no_rerank
  python benchmarks/longmemeval_eval.py --mode rerank
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

DATA = Path("/Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json")
MEMPALACE_BASELINE = Path(
    "/Users/macmini/Projects/adaptmem/benchmarks/structural_memory_eval/"
    "entity_graph_result.json"
)


def _session_text(session: list[dict]) -> str:
    """Flatten an LME session (list of {role, content}) into a single string."""
    return "\n".join(f"[{m.get('role','?')}] {m.get('content','')}" for m in session)


def _session_id_of(meta: str | None) -> str | None:
    """Extract LME session id from the meta we stored at ingest."""
    if not meta:
        return None
    # We prefix each text with "SID=<sid>|" at ingest time.
    if "SID=" in meta:
        return meta.split("SID=", 1)[1].split("|", 1)[0]
    return None


def evaluate_mnemonics(questions: list[dict], rerank: bool, top_k: int = 10,
                       candidate_k: int = 20,
                       augment_preferences: bool = False,
                       augment_assistant_facts: bool = False,
                       chunk_size: int = 200, chunk_overlap: int = 40,
                       use_doc_filter: bool = False,
                       per_q_out: Path | None = None) -> dict:
    """Run mnemonics retrieve() across every question, return aggregated metrics."""
    from mnemonics.store import Store
    from mnemonics.ingest import ingest
    from mnemonics.retrieve import retrieve

    ks = [1, 5, 10]
    hits = {k: 0 for k in ks}
    by_type = defaultdict(lambda: {"n": 0, **{f"hit@{k}": 0 for k in ks}})
    per_q: list[dict] = []
    t0 = time.time()
    for i, q in enumerate(questions):
        # Fresh store per question — LongMemEval is per-question independent.
        with tempfile.TemporaryDirectory() as td:
            store = Store(td)
            # Ingest each session as one chunk-stream tagged with its sid.
            texts = []
            for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
                texts.append(f"SID={sid}|{_session_text(sess)}")
            if not texts:
                continue
            ingest(texts=texts, store=store, ns="lme",
                   augment_preferences=augment_preferences,
                   augment_assistant_facts=augment_assistant_facts,
                   chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            try:
                result = retrieve(
                    query=q["question"],
                    store=store,
                    ns="lme",
                    top_k=top_k,
                    candidate_k=candidate_k,
                    rerank=rerank,
                    use_doc_filter=use_doc_filter,
                )
            except RuntimeError as e:
                print(f"  q{i} ERROR: {e}", file=sys.stderr)
                continue
            answer_sids = set(q.get("answer_session_ids") or [])
            retrieved_sids: list[str] = []
            for r in result["results"]:
                sid = _session_id_of(r.get("text"))
                if sid and sid not in retrieved_sids:
                    retrieved_sids.append(sid)

            qtype = q.get("question_type", "unknown")
            by_type[qtype]["n"] += 1
            q_hits = {}
            for k in ks:
                hit = any(sid in answer_sids for sid in retrieved_sids[:k])
                q_hits[k] = hit
                if hit:
                    hits[k] += 1
                    by_type[qtype][f"hit@{k}"] += 1
            if per_q_out is not None:
                per_q.append({
                    "qid": q.get("question_id"),
                    "qtype": qtype,
                    "question": q.get("question"),
                    "answer": q.get("answer"),
                    "answer_sids": list(answer_sids),
                    "retrieved_top10_sids": retrieved_sids[:10],
                    "hit@1": q_hits[1], "hit@5": q_hits[5], "hit@10": q_hits[10],
                })

        step = 1 if len(questions) <= 10 else 5
        if (i + 1) % step == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(questions) - i - 1) / max(rate, 1e-6)
            print(
                f"  [{i+1}/{len(questions)}] "
                f"R@1={hits[1]/(i+1):.3f} R@5={hits[5]/(i+1):.3f} "
                f"R@10={hits[10]/(i+1):.3f}  "
                f"rate={rate:.2f} q/s  eta={eta:.0f}s",
                flush=True,
            )

    n = sum(v["n"] for v in by_type.values())
    out = {
        "mode": "rerank" if rerank else "no_rerank",
        "n": n,
        "runtime_s": round(time.time() - t0, 1),
        **{f"R@{k}": round(hits[k] / max(n, 1), 4) for k in ks},
        "by_type": {
            t: {
                "n": v["n"],
                **{f"R@{k}": round(v[f"hit@{k}"] / max(v["n"], 1), 4) for k in ks},
            }
            for t, v in by_type.items()
        },
    }
    if per_q_out is not None:
        per_q_out.write_text(json.dumps(per_q, indent=2))
    return out


def mempalace_baseline_summary() -> dict:
    """Read AdaptMem's entity-graph baseline (already 500-q full)."""
    if not MEMPALACE_BASELINE.exists():
        return {"error": f"missing {MEMPALACE_BASELINE}"}
    d = json.load(open(MEMPALACE_BASELINE))
    s = d.get("summary", {})
    return {
        "mode": "mempalace_entity_graph",
        "n": s.get("n"),
        "R@1": s.get("R@1"),
        "R@5": s.get("R@5"),
        "R@10": s.get("R@10"),
        "by_type": {
            t: {"n": v["n"], "R@1": v["R@1"], "R@5": v["R@5"], "R@10": v["R@10"]}
            for t, v in (s.get("by_type") or {}).items()
        },
        "runtime_s": s.get("runtime_s"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="Question subset size (ignored if --split-file given)")
    ap.add_argument("--split-file", type=Path, default=None,
                    help="JSON with {'dev': [qid,...]} — subset by exact ids (apples-to-apples with MemPalace)")
    ap.add_argument("--mode", choices=["both", "no_rerank", "rerank"], default="both")
    ap.add_argument("--augment-preferences", action="store_true",
                    help="Pass augment_preferences=True to ingest (synth pref docs)")
    ap.add_argument("--augment-assistant-facts", action="store_true",
                    help="Pass augment_assistant_facts=True to ingest (synth numeric-fact docs over assistant turns)")
    ap.add_argument("--candidate-k", type=int, default=20,
                    help="Per-channel candidate band before fusion (default 20, MemPalace uses 50)")
    ap.add_argument("--chunk-size", type=int, default=200, help="Words per ingest chunk (default 200)")
    ap.add_argument("--chunk-overlap", type=int, default=40, help="Overlap words between adjacent chunks (default 40)")
    ap.add_argument("--use-doc-filter", action="store_true",
                    help="Enable Stage-1 session-level doc filter before fusion")
    ap.add_argument("--out", type=Path, default=Path("/tmp/mnemonics_vs_mempalace.json"))
    ap.add_argument("--per-q-out", type=Path, default=None,
                    help="Optional path to dump per-question hit/miss records")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {DATA}...", flush=True)
    all_q = json.load(open(DATA))
    if args.split_file:
        split = json.load(open(args.split_file))
        dev_ids = set(split.get("dev", []))
        questions = [q for q in all_q if q.get("question_id") in dev_ids]
        print(f"Split file: {len(questions)}/{len(dev_ids)} dev ids found in dataset", flush=True)
    elif args.n < len(all_q):
        import random
        rng = random.Random(args.seed)
        questions = rng.sample(all_q, args.n)
    else:
        questions = all_q
    print(f"Eval set: {len(questions)} questions (of {len(all_q)})", flush=True)
    am_path = os.environ.get("MNEMONICS_ADAPTMEM_PATH")
    print(f"MNEMONICS_ADAPTMEM_PATH = {am_path or '<unset, default MiniLM>'}", flush=True)

    results = {"n_questions": len(questions), "mempalace_full": mempalace_baseline_summary()}

    if args.mode in ("both", "no_rerank"):
        print(f"\n=== Mnemonics (no CE rerank) augment_prefs={args.augment_preferences} augment_facts={args.augment_assistant_facts} cand_k={args.candidate_k} chunk={args.chunk_size}/{args.chunk_overlap} doc_filter={args.use_doc_filter} ===", flush=True)
        results["mnemonics_no_rerank"] = evaluate_mnemonics(
            questions, rerank=False, candidate_k=args.candidate_k,
            augment_preferences=args.augment_preferences,
            augment_assistant_facts=args.augment_assistant_facts,
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
            use_doc_filter=args.use_doc_filter,
        )

    if args.mode in ("both", "rerank"):
        print(f"\n=== Mnemonics (CE rerank) augment_prefs={args.augment_preferences} augment_facts={args.augment_assistant_facts} cand_k={args.candidate_k} chunk={args.chunk_size}/{args.chunk_overlap} doc_filter={args.use_doc_filter} ===", flush=True)
        results["mnemonics_rerank"] = evaluate_mnemonics(
            questions, rerank=True, candidate_k=args.candidate_k,
            augment_preferences=args.augment_preferences,
            augment_assistant_facts=args.augment_assistant_facts,
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
            use_doc_filter=args.use_doc_filter,
            per_q_out=args.per_q_out,
        )

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}", flush=True)
    print("\n--- HEAD-TO-HEAD ---", flush=True)
    for key in ("mempalace_full", "mnemonics_no_rerank", "mnemonics_rerank"):
        v = results.get(key)
        if not v:
            continue
        print(
            f"{key:25} n={v.get('n') or 0:<4} "
            f"R@1={v.get('R@1')}  R@5={v.get('R@5')}  R@10={v.get('R@10')}"
        )


if __name__ == "__main__":
    main()
