"""Build per-question reranker documents: for each of the top-N candidate sessions, its best `--windows` windows
(local ms-marco-MiniLM-L-12-v2 CE vs the question, cached), each prefixed with the session date.

Output JSONL, one line per question: {"qid", "question", "docs": [{"sid", "text"}]}. A GPU kernel then scores
every doc with an open-weights reranker; session score = max over its docs. Splitting data prep (CPU, cached)
from scoring (GPU) keeps the scoring step a pure, replayable function of this file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentence_transformers import CrossEncoder

from dev_llm_rerank import locomo_sessions
from llm_judge_rerank import JsonlSessions, windows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perq", type=Path, required=True)
    ap.add_argument("--sids-key", default="retrieved_all_sids")
    ap.add_argument("--locomo", type=Path)
    ap.add_argument("--sessions-jsonl", type=Path)
    ap.add_argument("--ce-cache", type=Path)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--window-chars", type=int, default=600)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = json.load(open(args.perq))
    per_q = JsonlSessions(args.sessions_jsonl) if args.sessions_jsonl else None
    loco = locomo_sessions(args.locomo) if args.locomo else {}
    cache = json.load(open(args.ce_cache)) if args.ce_cache and args.ce_cache.exists() else {}
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")
    computed = 0
    with open(args.out, "w") as f:
        for r in rows:
            qid = r["qid"]
            sess = per_q[qid] if per_q is not None else loco
            docs = []
            for s in [x for x in r[args.sids_key][: args.top_n] if x in sess]:
                ch = windows(sess[s][1], args.window_chars)
                sc = cache.get(qid, {}).get(f"{s}|{args.window_chars}")
                if sc is None or len(sc) != len(ch):
                    sc = [float(x) for x in ce.predict([(r["question"], c) for c in ch], batch_size=64,
                                                         show_progress_bar=False)]
                    cache.setdefault(qid, {})[f"{s}|{args.window_chars}"] = sc
                    computed += 1
                for j in sorted(sorted(range(len(ch)), key=lambda j: -sc[j])[: args.windows]):
                    docs.append({"sid": s, "text": f"date: {sess[s][0]}\n{ch[j]}"})
            f.write(json.dumps({"qid": qid, "question": r["question"], "docs": docs}) + "\n")
    if computed and args.ce_cache:
        args.ce_cache.write_text(json.dumps(cache))
    print(f"questions {len(rows)}, sessions scored now {computed} -> {args.out}")


if __name__ == "__main__":
    main()
