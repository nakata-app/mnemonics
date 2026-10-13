"""Turn a synthetic LongMemEval-format dev set into the inputs of the deterministic rerank/judge pipeline.

Writes a perq (qid, qtype, question, answer, answer_sids, retrieved_all_sids = dense all-MiniLM-L6-v2 session
order, session = best 2-turn chunk) and a per-question sessions JSONL ({"qid", "sessions": {sid: [date,
text]}}) in the same "[role] content" text layout as extract_m_candidates_jsonl.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from synth_recall_probe import chunks_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--perq", type=Path, required=True)
    ap.add_argument("--sessions-jsonl", type=Path, required=True)
    args = ap.parse_args()

    data = json.load(open(args.data))
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    perq = []
    with open(args.sessions_jsonl, "w") as f:
        for q in data:
            qv = model.encode([q["question"]], normalize_embeddings=True, show_progress_bar=False)[0]
            best = [float((model.encode(chunks_of(s), normalize_embeddings=True, show_progress_bar=False) @ qv).max())
                    for s in q["haystack_sessions"]]
            order = [q["haystack_session_ids"][i] for i in np.argsort(-np.array(best))]
            perq.append({"qid": q["question_id"], "qtype": q["question_type"], "question": q["question"],
                         "answer": q["answer"], "answer_sids": q["answer_session_ids"], "retrieved_all_sids": order})
            sessions = {sid: [d, "\n".join(f'[{m["role"]}] {m["content"]}' for m in s)]
                        for sid, d, s in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"])}
            f.write(json.dumps({"qid": q["question_id"], "sessions": sessions}) + "\n")
    args.perq.write_text(json.dumps(perq))
    n = len(perq)
    hit = lambda k: sum(any(s in set(r["answer_sids"]) for s in r["retrieved_all_sids"][:k]) for r in perq) / n
    print(f"{n} questions | dense base R@1/5/10 {hit(1):.3f}/{hit(5):.3f}/{hit(10):.3f}")


if __name__ == "__main__":
    main()
