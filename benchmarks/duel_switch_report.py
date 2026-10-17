"""Offline switch report for a finished final-duel run (llm_judge_duel.py calls.jsonl).

Answers "did the duel switch help, and how well does it decide WHEN to switch" without a GPU or an LLM
call. Reads one or more calls.jsonl files and prints, per run:

  R@1/R@5/R@10 for base, judge and duel          (the headline numbers)
  fixes / breaks / neutral                       (paired against the judge, same rows, same order)
  McNemar exact two-sided p on those pairs
  switches: how many rows moved the judge's #1, and on which kind of row
    - judge #1 wrong, judge #2 gold  -> "rescuable" rows; switching there is the point of the stage
    - judge #1 already gold          -> switching there can only lose (or stay neutral via a second gold)
  switch precision = fixes / (fixes + breaks)   (of the switches that decided a row either way)
  switch yield     = fixes / switched_rows      (of every row the duel actually moved; this is the ledger's
                                                 "switch precision 10/26 = 0.38" statistic -- keep both)

Usage:
  python benchmarks/duel_switch_report.py OUT_DIR [OUT_DIR ...] [--json]
Each OUT_DIR must contain calls.jsonl (written by llm_judge_duel.py).
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def report(out: Path) -> dict:
    rows = [json.loads(l) for l in open(out / "calls.jsonl") if l.strip()]
    keys = {"base": "base", "judge": "judge", "duel": "order"}  # llm_judge_duel.py logs the duel as "order"
    hits = {k: [0, 0, 0] for k in keys}
    fixes = breaks = neutral = 0
    switched = 0
    rescuable = rescued = 0        # judge #1 wrong, judge #2 gold  -> did the duel take #2?
    already_right = lost = 0       # judge #1 gold                -> did the duel move off it?
    no_candidate = 0               # neither of the judge's two picks is gold
    for o in rows:
        gold = set(o["gold"])
        judge1 = o["judge"][0]
        duel1 = o["order"][0]
        judge_ok = judge1 in gold
        duel_ok = duel1 in gold
        for i, k in enumerate((1, 5, 10)):
            for key, field in keys.items():
                hits[key][i] += any(s in gold for s in o[field][:k])
        if duel1 != judge1:
            switched += 1
        if judge_ok and duel_ok:
            neutral += 1
        elif duel_ok:
            fixes += 1
        elif judge_ok:
            breaks += 1
        else:
            neutral += 1  # still wrong; counted so fixes+breaks+neutral == n only with the split below
        if not judge_ok and len(o["judge"]) > 1 and o["judge"][1] in gold:
            rescuable += 1
            rescued += int(duel_ok)
        elif judge_ok:
            already_right += 1
            lost += int(not duel_ok)
        elif not duel_ok:
            no_candidate += 1
    n = len(rows)
    denom = fixes + breaks
    return {
        "run": str(out),
        "n": n,
        "base": [round(x / n, 4) for x in hits["base"]],
        "judge": [round(x / n, 4) for x in hits["judge"]],
        "duel": [round(x / n, 4) for x in hits["duel"]],
        "net": fixes - breaks,
        "fixes": fixes,
        "breaks": breaks,
        "mcnemar_p": round(mcnemar_exact(fixes, breaks), 4),
        "switched_rows": switched,
        "switch_yield": round(fixes / switched, 3) if switched else None,
        "switch_precision": round(fixes / denom, 3) if denom else None,
        "rescuable_rows": rescuable,
        "rescued": rescued,
        "rescue_rate": round(rescued / rescuable, 3) if rescuable else None,
        "judge_right_rows": already_right,
        "lost_when_right": lost,
        "neither_pick_gold": no_candidate,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", type=Path, nargs="+")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    for out in args.out_dirs:
        r = report(out)
        if args.json:
            print(json.dumps(r, indent=2))
        else:
            print(f"{out.name:28s} n={r['n']}  base {r['base'][0]:.3f}  judge {r['judge'][0]:.3f}  "
                  f"duel {r['duel'][0]:.3f}  ({r['fixes']}F/{r['breaks']}B, net {r['net']:+d}, "
                  f"p={r['mcnemar_p']})")
            print(f"{'':28s} switched {r['switched_rows']}  yield {r['switch_yield']}  "
                  f"precision {r['switch_precision']}  rescued {r['rescued']}/{r['rescuable_rows']} "
                  f"({r['rescue_rate']})  lost when judge right {r['lost_when_right']}/{r['judge_right_rows']}")


if __name__ == "__main__":
    main()
