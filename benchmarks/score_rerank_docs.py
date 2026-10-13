"""Turn kaggle_qwen_rerank_docs.py scores into session rankings and R@k.

Session score = max over its documents. Sessions without documents keep their base order after the scored
ones. Writes a calls.jsonl with `base` and `order` so the judge stages can use it as --base-calls.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perq", type=Path, required=True)
    ap.add_argument("--sids-key", default="retrieved_all_sids")
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    perq = {r["qid"]: r for r in json.load(open(args.perq))}
    docs = {json.loads(l)["qid"]: json.loads(l)["docs"] for l in open(args.docs) if l.strip()}
    scores = {json.loads(l)["qid"]: json.loads(l)["scores"] for l in open(args.scores) if l.strip()}
    args.out.mkdir(parents=True, exist_ok=True)
    ks = (1, 5, 10)
    H = {"base": [0, 0, 0], "order": [0, 0, 0]}
    BT = defaultdict(lambda: {"n": 0, "base": [0, 0, 0], "order": [0, 0, 0]})
    with open(args.out / "calls.jsonl", "w") as f:
        for qid, sc in scores.items():
            r = perq[qid]
            base = r[args.sids_key]
            best = {}
            for d, v in zip(docs[qid], sc):
                best[d["sid"]] = max(best.get(d["sid"], -1.0), v)
            ranked = sorted(best, key=lambda s: (-best[s], base.index(s)))
            order = ranked + [s for s in base if s not in best]
            gold = set(r["answer_sids"])
            BT[r["qtype"]]["n"] += 1
            for i, k in enumerate(ks):
                for key, lst in (("base", base), ("order", order)):
                    h = any(s in gold for s in lst[:k])
                    H[key][i] += h
                    BT[r["qtype"]][key][i] += h
            f.write(json.dumps({"qid": qid, "qtype": r["qtype"], "base": base[:75], "order": order[:75],
                                "gold": sorted(gold)}) + "\n")
    n = len(scores)
    summary = {"n": n, "base": [round(x / n, 4) for x in H["base"]], "reranked": [round(x / n, 4) for x in H["order"]],
               "by_type": {t: {"n": v["n"], "base": [round(x / v["n"], 3) for x in v["base"]],
                               "reranked": [round(x / v["n"], 3) for x in v["order"]]} for t, v in sorted(BT.items())}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
