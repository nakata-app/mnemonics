"""Dense-retrieval recall probe on the synthetic preference dev set (local, CPU, memory-safe).

Session = best 2-turn chunk vs the question (all-MiniLM-L6-v2, cosine). Reports R@1/5/10/30 over each
question's pooled haystack, for two policies:
  base       : rank sessions by best-chunk similarity (the symptom: preference gold is out-ranked by
               sessions that only echo the question's topic).
  pref-lane  : a dedicated preference lane. Extract first-person preference statements with GENERAL
               grammatical markers (no topic words), score the question against those records, and gate
               preference-bearing sessions above the rest (ordered by preference-record similarity;
               non-preference sessions fall back to base order). Accept only if it improves dev with
               zero regression -- which it does on this set: R@1 0.0->0.909, R@5 0.545->1.0, R@10 1.0.
This is dev, synthetic, proof-of-mechanism only; it is NOT an M result. The production extractor should
be an LLM (robust to arbitrary phrasing); the regex here only demonstrates the mechanism off the exam.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# first-person preference markers: grammatical, domain-independent (no topic/brand words).
PREF = re.compile(
    r"\b(i (can't|cannot|never|always)|i'd (rather|always)|my thing|does nothing for me|"
    r"(isn't|aren't) for me|kill me|drain me|exhaust me|stress me out|steer clear|gave up)\b",
    re.I,
)


def chunks_of(session: list[dict]) -> list[str]:
    lines = [f'[{m["role"]}] {m["content"]}' for m in session]
    return ["\n".join(lines[j:j + 2]) for j in range(0, len(lines), 2)] or [""]


def pref_lines(session: list[dict]) -> list[str]:
    return [m["content"] for m in session if m["role"] == "user" and PREF.search(m["content"])]


def score(data, model):
    out = []
    for q in data:
        qv = model.encode([q["question"]], normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=False)[0]
        base, pscore = [], []
        for sess in q["haystack_sessions"]:
            ce = model.encode(chunks_of(sess), normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)
            base.append(float((ce @ qv).max()))
            pl = pref_lines(sess)
            if pl:
                pe = model.encode(pl, normalize_embeddings=True, convert_to_numpy=True,
                                  show_progress_bar=False)
                pscore.append(float((pe @ qv).max()))
            else:
                pscore.append(None)
        out.append((q["haystack_session_ids"], set(q["answer_session_ids"]), base, pscore))
    return out


def recall(scored, lane: bool):
    hits = {k: 0 for k in (1, 5, 10, 30)}
    ranks = []
    for sids, gold, base, pscore in scored:
        if lane:
            has = sorted((i for i in range(len(base)) if pscore[i] is not None), key=lambda i: -pscore[i])
            non = sorted((i for i in range(len(base)) if pscore[i] is None), key=lambda i: -base[i])
            order = has + non
        else:
            order = list(np.argsort(-np.array(base)))
        ranked = [sids[i] for i in order]
        ranks.append(next((i + 1 for i, s in enumerate(ranked) if s in gold), None))
        for k in hits:
            hits[k] += any(s in gold for s in ranked[:k])
    n = len(scored)
    return {f"R@{k}": round(v / n, 4) for k, v in hits.items()} | {
        "median_gold_rank": int(np.median([r for r in ranks if r]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("benchmarks/data/synth/pref_dev.json"))
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    data = json.load(open(args.data))
    model = SentenceTransformer(args.model, device="cpu")
    scored = score(data, model)
    print(json.dumps({"n": len(data), "base": recall(scored, False),
                      "pref_lane": recall(scored, True)}, indent=2))


if __name__ == "__main__":
    main()
