"""Run the final-duel stage only where the CE and the judge DISAGREE.

The duel loses more than it gains because it also moves rows the judge already had right (see
duel_switch_report.py: 32B switches 26 times, 10 fixes, 7 breaks, 12 switches on already-right rows).
The judge and the cheap CE are two independent opinions on the same two candidates, so a natural,
gold-free gate is: take the duel's move only when the CE ranks the judge's #2 ABOVE its #1. Where the
CE agrees with the judge, keep the judge order.

No question label, no gold, no qid is used. Session CE score = max window score from the
locomo_ce_cache_w600.json cache (the same number llm_judge_rerank.py uses to pick windows).

Writes a raw JSONL ({qid, raw}) that llm_judge_duel.py --replay-raw scores exactly like any generation
run, so the instrument and its parse rules stay untouched:

  python benchmarks/duel_switch_gate.py --calls OUT/calls.jsonl --ce-cache CE.json --out gated_raw.jsonl
  python benchmarks/llm_judge_duel.py ... --reason-prompt --replay-raw gated_raw.jsonl --out GATED_OUT
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def session_ce(cache: dict, qid: str, sid: str) -> float | None:
    ws = cache.get(qid, {}).get(f"{sid}|600")
    return max(ws) if ws else None


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Exact two-sided Fisher test on [[a, b], [c, d]]."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    def p(x: int) -> float:
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    obs = p(a)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs + 1e-12))


def gate_passes(cache: dict, judge: list, qid: str, margin: float) -> bool:
    if len(judge) < 2:
        return False
    a, b = session_ce(cache, qid, judge[0]), session_ce(cache, qid, judge[1])
    return a is not None and b is not None and (b - a) > margin



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=Path, required=True)
    ap.add_argument("--ce-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="require CE(judge#2) - CE(judge#1) > margin (exploratory only; 0 is the rule)")
    ap.add_argument("--mode", choices=("veto", "swap"), default="veto",
                    help="veto: keep the duel's move only where the gate passes (default). "
                         "swap: ignore the duel and swap the judge's top-2 where the gate passes -- the "
                         "negative control that shows CE disagreement is not a signal on its own.")
    args = ap.parse_args()

    ce = json.load(open(args.ce_cache))
    rows = [json.loads(l) for l in open(args.calls) if l.strip()]
    stats = {"switch": 0, "keep": 0, "no_ce": 0}
    # label crosstab, computed only for reporting AFTER the decision (the rule never sees gold)
    conf = {"RESCUE": [0, 0], "BREAK": [0, 0], "OTHER": [0, 0]}  # [passed gate, blocked]
    out = []
    for r in rows:
        raw = r.get("raw")
        chosen = None
        if raw:
            import re
            m = re.search(r"(?i)answer:\s*(\d+)", raw)
            chosen = int(m.group(1)) if m else None
            if chosen is None:
                b = re.fullmatch(r"\s*[\[(]?\s*(\d{1,3})\s*[\])]?\s*[.!]?\s*", raw)
                chosen = int(b.group(1)) if b else None
        # duel calls files carry the judge order in "judge"; a plain judge-run file carries it in "order"
        judge = r.get("judge") or (r.get("order") if args.mode == "swap" else []) or []
        allow = False
        margin = None
        if args.mode == "swap":
            target = 1 if gate_passes(ce, judge, r["qid"], args.margin) else 0
            stats["switch" if target else "keep"] += 1
            out.append({"qid": r["qid"], "raw": str(target)})
            if len(judge) > 1:
                gold = set(r["gold"])
                lab = ("RESCUE" if (judge[0] not in gold and judge[1] in gold)
                       else "BREAK" if (judge[0] in gold and judge[1] not in gold) else "OTHER")
                conf[lab][0 if target else 1] += 1
            continue
        if chosen not in (None, 0) and len(judge) > 1:
            a, b = session_ce(ce, r["qid"], judge[0]), session_ce(ce, r["qid"], judge[1])
            if a is None or b is None:
                stats["no_ce"] += 1
            else:
                margin = b - a  # >0 means the CE prefers the judge's #2
                allow = margin > args.margin
        stats["switch" if allow else "keep"] += 1
        out.append({"qid": r["qid"], "raw": str(chosen if allow else 0)})
        if raw and chosen is not None and chosen != 0 and judge:
            gold = set(r["gold"])
            j1ok = judge[0] in gold
            d1ok = judge[chosen] in gold if chosen < len(judge) else False
            lab = "RESCUE" if (not j1ok and d1ok) else "BREAK" if (j1ok and not d1ok) else "OTHER"
            conf[lab][0 if allow else 1] += 1

    args.out.write_text("".join(json.dumps(o) + "\n" for o in out))
    print(json.dumps({"calls": str(args.calls), "margin": args.margin, "stats": stats,
                      "passed_gate_vs_blocked": conf}, indent=2))
    r_, b_ = conf["RESCUE"], conf["BREAK"]
    if sum(r_) and sum(b_):
        print("rescues allowed/blocked:", r_[0], "/", r_[1], "  breaks allowed/blocked:", b_[0], "/", b_[1])
        print("Fisher two-sided p (are rescues more likely than breaks to pass the gate):",
              round(fisher_two_sided(r_[0], r_[1], b_[0], b_[1]), 4))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
