"""LongMemEval benchmark for Mnemonics.

Protocol matches MemPalace / adaptmem:
  - Per-question fresh store (haystack_sessions = corpus)
  - User-only session encoding
  - relevant_ids = answer_session_ids
  - Recall@k: 1.0 if any GT id in top-k corpus_ids, else 0.0

Usage:
  python bench_longmemeval.py --dataset /path/to/longmemeval_s_cleaned.json
  python bench_longmemeval.py --dataset ... --n 50 --top-k 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tempfile
from pathlib import Path

import numpy as np


def session_to_text(turns: list[dict]) -> str:
    return "\n".join(t.get("content", "") for t in turns if t.get("role") == "user")


def recall_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int) -> float:
    return 1.0 if any(rid in ranked_ids[:k] for rid in relevant_ids) else 0.0


def run(dataset_path: str, n: int | None, top_k: int, model: str) -> dict:
    from sentence_transformers import SentenceTransformer
    from mnemonics.store import Store, DIM

    print(f"Loading encoder: {model}", flush=True)
    enc = SentenceTransformer(model)

    with open(dataset_path) as f:
        data = json.load(f)

    questions = [q for q in data if q.get("answer_session_ids")]
    if n:
        questions = questions[:n]

    print(f"Running {len(questions)} questions, top_k={top_k}", flush=True)

    recalls = []
    t0 = time.time()

    for i, q in enumerate(questions):
        qid = q["question"]
        answer_ids = q["answer_session_ids"]
        sessions = q.get("haystack_sessions", {})

        # Build corpus: session_id -> text (user-only, drop empty)
        session_ids = q.get("haystack_session_ids", [])
        corpus = {}
        for sid, turns in zip(session_ids, sessions):
            text = session_to_text(turns)
            if text.strip():
                corpus[sid] = text

        if not corpus:
            recalls.append(0.0)
            continue

        # Fresh store per question (dim inferred from encoder)
        dim = enc.get_sentence_embedding_dimension()
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp, dim=dim)

            texts = list(corpus.values())
            sids = list(corpus.keys())

            vecs = enc.encode(texts, batch_size=64, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
            store.add(texts, vecs, meta=[{"session_id": s} for s in sids])

            qvec = enc.encode([qid], normalize_embeddings=True,
                              convert_to_numpy=True)[0]
            results = store.search(qvec, top_k=top_k)

        ranked_ids = [r["meta"]["session_id"] for r in results]
        r = recall_at_k(ranked_ids, answer_ids, top_k)
        recalls.append(r)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(questions)}] R@{top_k}={np.mean(recalls):.4f}  "
                  f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    result = {
        "model": model,
        "n_questions": len(questions),
        "top_k": top_k,
        "recall_at_k": round(float(np.mean(recalls)), 4),
        "elapsed_s": round(elapsed, 1),
    }

    print(f"\nR@{top_k} = {result['recall_at_k']}  "
          f"({result['n_questions']} questions, {elapsed:.0f}s)", flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--n", type=int, default=None, help="Limit questions (default: all)")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--out", default=None, help="Save results JSON")
    args = p.parse_args()

    result = run(args.dataset, args.n, args.top_k, args.model)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
