"""Preference-lane post-hoc rerank on the M top-30 pool dump (local, label-free, pre-registered).

Honest M test of the preference lane proven on the synthetic dev set. Operates only on the already
dumped candidates (m_poolall_perq retrieved_all_sids + m_poolall_top30 session texts), so it runs on
the Mac and never re-runs M retrieval.

FROZEN, developed ONLY on benchmarks/data/synth (never on M):
- ROUTER (label-free): a question is treated as a preference question from its own wording, NOT from the
  LongMemEval question_type label. If it does not match, the base order is kept unchanged.
- EXTRACTOR: first-person preference statements by general grammatical markers (no topic/brand words).
- GATE: among the top-N candidates, sessions that contain a preference statement are ordered first by
  the similarity of that statement to the question; the rest keep their base order.

Non-preference questions are untouched by construction, so other types cannot regress. Reports R@1/5/10
overall and by type, base vs reranked, and how often the router fired. Run once, log the result as is.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from sentence_transformers import SentenceTransformer

# FROZEN from synthetic dev. Label-free preference-QUESTION wording (general, no topic words).
ROUTER = re.compile(
    r"\b(prefer|favou?rite|like to|enjoy|go for|my (go-to|ideal|approach)|"
    r"what (kind|sort|type) of .* do i|how (do|would) i (like|prefer)|reach me)\b",
    re.I,
)
# FROZEN from synthetic dev. First-person preference markers (grammatical, no topic words).
PREF = re.compile(
    r"\b(i (can't|cannot|never|always)|i'd (rather|always)|my thing|does nothing for me|"
    r"(isn't|aren't) for me|kill me|drain me|exhaust me|stress me out|steer clear|gave up)\b",
    re.I,
)


def pref_statements(text: str) -> list[str]:
    # the session text is "[user] ...\n[assistant] ...\n..."; keep user lines with a preference marker
    lines = [ln[len("[user] "):] for ln in text.split("\n") if ln.startswith("[user] ")]
    return [ln for ln in lines if PREF.search(ln)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perq", type=Path, required=True)
    ap.add_argument("--sessions", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    perq = json.load(open(args.perq))
    sess_by_q = json.load(open(args.sessions))
    model = SentenceTransformer(args.model, device="cpu")

    ks = (1, 5, 10)
    H = {"base": {k: 0 for k in ks}, "rr": {k: 0 for k in ks}}
    BT = defaultdict(lambda: {"n": 0, "base": {k: 0 for k in ks}, "rr": {k: 0 for k in ks}})
    fired = 0

    for r in perq:
        qid, q = r["qid"], r["question"]
        gold = set(r["answer_sids"])
        base_order = r["retrieved_all_sids"][: args.top_n]
        sess = sess_by_q.get(qid, {})

        order = base_order
        if ROUTER.search(q):
            fired += 1
            qv = model.encode([q], normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)[0]
            has, non = [], []
            for sid in base_order:
                st = pref_statements(sess[sid][1]) if sid in sess else []
                if st:
                    pe = model.encode(st, normalize_embeddings=True, convert_to_numpy=True,
                                      show_progress_bar=False)
                    has.append((sid, float((pe @ qv).max())))
                else:
                    non.append(sid)
            has.sort(key=lambda x: -x[1])
            order = [s for s, _ in has] + non

        qt = r["qtype"]
        BT[qt]["n"] += 1
        for k in ks:
            hb = any(s in gold for s in base_order[:k])
            hr = any(s in gold for s in order[:k])
            H["base"][k] += hb; H["rr"][k] += hr
            BT[qt]["base"][k] += hb; BT[qt]["rr"][k] += hr

    n = len(perq)
    out = {
        "n": n, "router_fired": fired,
        "base": {f"R@{k}": round(H["base"][k] / n, 4) for k in ks},
        "reranked": {f"R@{k}": round(H["rr"][k] / n, 4) for k in ks},
        "by_type": {qt: {"n": v["n"],
                         "base": {f"R@{k}": round(v["base"][k] / v["n"], 3) for k in ks},
                         "reranked": {f"R@{k}": round(v["rr"][k] / v["n"], 3) for k in ks}}
                    for qt, v in sorted(BT.items())},
    }
    print(json.dumps(out, indent=2))
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
