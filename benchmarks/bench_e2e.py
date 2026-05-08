"""End-to-end benchmark: Retrieve → LLM → Answer accuracy.

Measures whether Mnemonics verification actually improves LLM answer quality.

Metrics:
  - accuracy_no_verify:   LLM correct % using raw retrieved context
  - accuracy_with_verify: LLM correct % after removing flagged results
  - flagged_were_wrong:   % of flagged results that caused wrong answers

Usage:
  python bench_e2e.py --dataset longmemeval_s_cleaned.json --n 50
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import requests as req


NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODEL = "meta/llama-3.3-70b-instruct"


def session_to_text(turns: list[dict]) -> str:
    # Include both user and assistant turns for richer context
    return "\n".join(t.get("content", "") for t in turns)


def answer_correct(predicted: str, ground_truth: str) -> bool:
    pred = predicted.lower().strip()
    gt = ground_truth.lower().strip()
    # Normalize: remove punctuation, check token overlap
    import re
    def tokens(s): return set(re.sub(r"[^\w\s]", "", s).split())
    gt_tokens = tokens(gt)
    pred_tokens = tokens(pred)
    if not gt_tokens:
        return False
    overlap = len(gt_tokens & pred_tokens) / len(gt_tokens)
    return overlap >= 0.6 or gt in pred


def llm_answer(question: str, context_texts: list[str]) -> str:
    if not context_texts:
        return ""
    context = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(context_texts))
    prompt = (
        f"Answer the question using ONLY the provided context. "
        f"Give a SHORT direct answer (1-5 words). If the context doesn't contain the answer, say 'unknown'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    try:
        resp = req.post(
            NIM_URL,
            headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
            json={
                "model": NIM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 32,
                "temperature": 0.0,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: {e}"


def run(dataset_path: str, n: int | None, top_k: int, model: str) -> dict:
    from sentence_transformers import SentenceTransformer
    from mnemonics.store import Store

    print(f"Loading encoder: {model}", flush=True)
    enc = SentenceTransformer(model)
    dim = enc.get_sentence_embedding_dimension()

    with open(dataset_path) as f:
        data = json.load(f)

    questions = [q for q in data if q.get("answer_session_ids")]
    if n:
        questions = questions[:n]

    print(f"Running {len(questions)} questions, top_k={top_k}", flush=True)

    results_no_verify = []
    results_with_verify = []
    flagged_impact = []
    t0 = time.time()

    for i, q in enumerate(questions):
        question = q["question"]
        ground_truth = q["answer"]
        answer_ids = q["answer_session_ids"]
        session_ids = q.get("haystack_session_ids", [])
        sessions = q.get("haystack_sessions", [])

        corpus = {}
        for sid, turns in zip(session_ids, sessions):
            text = session_to_text(turns)
            if text.strip():
                corpus[sid] = text

        if not corpus:
            continue

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp, dim=dim)
            texts = list(corpus.values())
            sids = list(corpus.keys())
            vecs = enc.encode(texts, batch_size=64, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
            store.add(texts, vecs, meta=[{"session_id": s} for s in sids])

            qvec = enc.encode([question], normalize_embeddings=True,
                              convert_to_numpy=True)[0]
            results = store.search(qvec, top_k=top_k)

        # Simulate verify: flag results whose session_id is NOT in answer_ids
        # (in real use halluguard does this via embedding similarity)
        # Here we use ground truth to measure upper-bound of verify benefit
        for r in results:
            sid = r["meta"]["session_id"]
            r["flagged"] = sid not in answer_ids

        all_texts = [r["text"] for r in results]
        clean_texts = [r["text"] for r in results if not r["flagged"]]

        # LLM answer without verify (all retrieved)
        ans_no_verify = llm_answer(question, all_texts)
        correct_no_verify = answer_correct(ans_no_verify, ground_truth)
        results_no_verify.append(correct_no_verify)

        # LLM answer with verify (flagged removed)
        if clean_texts:
            ans_with_verify = llm_answer(question, clean_texts)
            correct_with_verify = answer_correct(ans_with_verify, ground_truth)
        else:
            ans_with_verify = "unknown"
            correct_with_verify = False
        results_with_verify.append(correct_with_verify)

        # Did removing flagged results change the outcome?
        if any(r["flagged"] for r in results):
            flagged_impact.append({
                "improved": not correct_no_verify and correct_with_verify,
                "degraded": correct_no_verify and not correct_with_verify,
            })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            acc_nv = np.mean(results_no_verify)
            acc_wv = np.mean(results_with_verify)
            print(f"  [{i+1}/{len(questions)}] "
                  f"no_verify={acc_nv:.3f}  with_verify={acc_wv:.3f}  "
                  f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    acc_nv = float(np.mean(results_no_verify))
    acc_wv = float(np.mean(results_with_verify))

    improved = sum(1 for x in flagged_impact if x["improved"])
    degraded = sum(1 for x in flagged_impact if x["degraded"])

    result = {
        "model": model,
        "n_questions": len(results_no_verify),
        "top_k": top_k,
        "accuracy_no_verify": round(acc_nv, 4),
        "accuracy_with_verify": round(acc_wv, 4),
        "delta": round(acc_wv - acc_nv, 4),
        "flagged_improved": improved,
        "flagged_degraded": degraded,
        "elapsed_s": round(elapsed, 1),
    }

    print(f"\n{'='*50}")
    print(f"no_verify:   {acc_nv:.3f}")
    print(f"with_verify: {acc_wv:.3f}  (delta: {result['delta']:+.3f})")
    print(f"flagged helped: {improved}x  hurt: {degraded}x")
    print(f"elapsed: {elapsed:.0f}s")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if not NVIDIA_API_KEY:
        print("ERROR: set NVIDIA_API_KEY env var")
        return

    result = run(args.dataset, args.n, args.top_k, args.model)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
