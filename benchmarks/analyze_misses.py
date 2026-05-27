"""Analyze per-question dump from longmemeval_eval to inspect misses.

Reads:
  - eval/results/lme*_perq.json  (per-question records)
  - /Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json

Prints:
  - For each hit@10=false question: qid, qtype, question, expected answer,
    a brief preview of the actual answer-session assistant turns so we
    can judge whether assistant-fact patterns would have helped.

Usage:
  python benchmarks/analyze_misses.py eval/results/lme50_perq.json
  python benchmarks/analyze_misses.py eval/results/lme50_perq.json --k 1   # show R@1 misses instead
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA = Path("/Users/macmini/Projects/metis-pair/benchmarks/data/longmemeval/longmemeval_s_cleaned.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("perq", type=Path, help="lme50_perq.json from eval --per-q-out")
    ap.add_argument("--k", type=int, default=10, choices=[1, 5, 10],
                    help="Show misses at this k (default 10)")
    ap.add_argument("--show-assistant", action="store_true",
                    help="Print assistant turns from the answer session")
    args = ap.parse_args()

    perq = json.load(open(args.perq))
    misses = [r for r in perq if not r.get(f"hit@{args.k}")]
    all_q = {q["question_id"]: q for q in json.load(open(DATA))}
    print(f"Per-q dump: {args.perq}  total={len(perq)}  misses@{args.k}={len(misses)}", flush=True)
    print(f"Hit@1={sum(r['hit@1'] for r in perq)}/{len(perq)}  Hit@5={sum(r['hit@5'] for r in perq)}/{len(perq)}  Hit@10={sum(r['hit@10'] for r in perq)}/{len(perq)}", flush=True)
    print()

    for i, r in enumerate(misses):
        qid = r["qid"]
        print(f"--- miss {i+1}/{len(misses)}  qid={qid}  type={r['qtype']} ---")
        print(f"Q: {r['question']}")
        print(f"A (expected): {r['answer']}")
        print(f"answer_sids: {r['answer_sids']}")
        print(f"retrieved top10: {r['retrieved_top10_sids']}")
        if args.show_assistant:
            q = all_q.get(qid)
            if q:
                ans_set = set(r["answer_sids"])
                for sid, sess in zip(q["haystack_session_ids"], q["haystack_sessions"]):
                    if sid in ans_set:
                        print(f"  >> answer session [{sid}] assistant turns:")
                        for m in sess:
                            if m["role"] == "assistant":
                                print(f"     {m['content'][:300]}")
                                print()
                        break
        print()


if __name__ == "__main__":
    main()
