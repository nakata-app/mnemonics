"""EXAM 21: recency tiebreaker over S11 near-ties, measured on a frozen held-out M split.

Pre-registered in docs/EXAM21_PREREG.md (committed before this file ran on any score). Baseline = S11, the
persisted EXAM 11 order in results/exam11_m_scored/calls.jsonl. The lever reorders the S11 top-K by session
recency relative to the question date (orthogonal to content similarity, which dense+reranker exhaust).

Frozen grid (9): K in {2,3,5} x arm in {promote1, sort, oldest}; plus baseline S11. `oldest` is a negative
control and is never reported as a result. Selection = best dev-M overall R@1 (ties: smaller K, promote1, sort);
if nothing beats baseline on dev-M, verdict FAIL. Reported result = test-M for the selected config.

No GPU, no cloud. Deterministic (no randomness in apply); the split is regenerated from seed 21 and checked
against the frozen split.json. Run: python3 exam21_recency.py
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

R = Path("/Volumes/WD-Backup/datasets/longmemeval/results")
I = Path("/Volumes/WD-Backup/datasets/longmemeval/rerank_inputs")
D = Path.home() / ".local/share/mnemonics-exam21"
CALLS = R / "exam11_m_scored" / "calls.jsonl"
QDATES = I / "m_qdates.json"
SESSIONS = I / "m_deep_top75_sessions.jsonl"
KS = (2, 3, 5)
ARMS = ("promote1", "sort", "oldest")
INF = 10**9


def norm_qdate(s: str) -> date:
    return datetime.strptime(s.strip(), "%Y/%m/%d (%a) %H:%M").date()


def norm_sdate(s: str) -> date:
    return datetime.strptime(s.strip()[:10], "%Y/%m/%d").date()


def regen_split() -> dict:
    """Reproduce the frozen seed-21 type-stratified 50/50 split and assert it matches split.json."""
    e11 = [json.loads(l) for l in open(CALLS) if l.strip()]
    byt = defaultdict(list)
    for r in e11:
        byt[r["qtype"]].append(r["qid"])
    rng = random.Random(21)
    dev, test = [], []
    for t in sorted(byt):
        ids = sorted(byt[t])
        rng.shuffle(ids)
        h = len(ids) // 2
        dev += ids[:h]
        test += ids[h:]
    split = {"seed": 21, "dev": sorted(dev), "test": sorted(test)}
    saved = json.load(open(D / "split.json"))
    assert saved["dev"] == split["dev"] and saved["test"] == split["test"], "split mismatch — frozen seed 21 broken"
    return split


def reorder(order: list, ages: dict, k: int, arm: str) -> list:
    head = order[:k]
    fin = [s for s in head if ages.get(s, INF) < INF]
    if not fin:
        return order
    if arm == "promote1":
        best = min(fin, key=lambda s: (ages[s], order.index(s)))  # smallest age, S11 order breaks ties
        return [best] + [s for s in order if s != best]
    if arm == "oldest":
        best = max(fin, key=lambda s: (ages[s], -order.index(s)))  # largest finite age (negative control)
        return [best] + [s for s in order if s != best]
    if arm == "sort":
        # stable sort top-K by age ascending (finite first, +inf keeps position)
        ordered = sorted(head, key=lambda s: (ages.get(s, INF), head.index(s)))
        return ordered + order[k:]
    raise ValueError(arm)


def main() -> None:
    split = regen_split()
    dev, test = set(split["dev"]), set(split["test"])
    calls = {r["qid"]: r for r in (json.loads(l) for l in open(CALLS) if l.strip())}
    qdates = json.load(open(QDATES))

    configs = ["S11"] + [f"K{k}_{a}" for k in KS for a in ARMS]
    # r1[cfg][qid] = 0/1 ; also keep r5/r10 for the record
    hit = {c: {} for c in configs}
    hit5 = {c: {} for c in configs}
    hit10 = {c: {} for c in configs}

    def score(order, gold, store1, store5, store10, qid):
        store1[qid] = 1 if any(s in gold for s in order[:1]) else 0
        store5[qid] = 1 if any(s in gold for s in order[:5]) else 0
        store10[qid] = 1 if any(s in gold for s in order[:10]) else 0

    with open(SESSIONS) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row["qid"]
            if qid not in calls:
                continue
            order = calls[qid]["order"]
            gold = set(calls[qid]["gold"])
            qd = norm_qdate(qdates[qid]) if qid in qdates else None
            ages = {}
            if qd is not None:
                for sid, v in row["sessions"].items():
                    try:
                        sd = norm_sdate(v[0])
                    except Exception:
                        continue
                    ages[sid] = (qd - sd).days if sd <= qd else INF
            score(order, gold, hit["S11"], hit5["S11"], hit10["S11"], qid)
            for k in KS:
                for a in ARMS:
                    c = f"K{k}_{a}"
                    o = reorder(order, ages, k, a) if qd is not None else order
                    score(o, gold, hit[c], hit5[c], hit10[c], qid)

    missing = [q for q in calls if q not in hit["S11"]]
    if missing:
        for c in configs:
            for q in missing:
                order = calls[q]["order"]; gold = set(calls[q]["gold"])
                score(order, gold, hit[c], hit5[c], hit10[c], q)

    def r1(c, ids):
        return round(sum(hit[c][q] for q in ids) / len(ids), 4)

    # selection on dev-M (exclude the oldest negative-control arm from being *selected*? prereg: grid includes
    # oldest in selection, but it may not be reported as a result. We follow the frozen rule: select best dev R@1
    # over all 9; if the pick is an `oldest` arm, that is a negative-control win and the verdict is reported as
    # such, not as a usable result.)
    dev_ids = sorted(dev)
    cand = [f"K{k}_{a}" for k in KS for a in ARMS]
    base_dev = r1("S11", dev_ids)
    rank_key = lambda c: (-r1(c, dev_ids), int(c.split("_")[0][1:]), ARMS.index(c.split("_")[1]))
    sel = min(cand, key=rank_key)
    beats = r1(sel, dev_ids) > base_dev

    print(f"split seed 21 OK | dev-M n={len(dev)} test-M n={len(test)}")
    print(f"\ndev-M R@1 (selection set): S11={base_dev}")
    for k in KS:
        for a in ARMS:
            c = f"K{k}_{a}"
            print(f"  {c:12s} {r1(c, dev_ids)}")
    print(f"\ndev-M selection = {sel} (beats S11 on dev: {beats})")

    test_ids = sorted(test)
    sel_is_control = sel.endswith("oldest")
    # paired McNemar on test-M R@1
    f = b = 0
    for q in test_ids:
        s, c = hit["S11"][q], hit[sel][q]
        f += c and not s
        b += s and not c
    net = f - b
    # two-sided exact binomial p on discordant pairs
    from math import comb
    n = f + b
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(0, min(f, b) + 1)) / 2**n)
    bytype_test = {}
    qtypes = {q: calls[q]["qtype"] for q in calls}
    for t in sorted(set(qtypes[q] for q in test_ids)):
        qs = [q for q in test_ids if qtypes[q] == t]
        bytype_test[t] = {"n": len(qs), "S11": round(sum(hit["S11"][q] for q in qs) / len(qs), 3),
                          sel: round(sum(hit[sel][q] for q in qs) / len(qs), 3)}
    g1 = beats and r1(sel, test_ids) > r1("S11", test_ids) and net > 0 and p < 0.10
    g2 = all(bytype_test[t][sel] >= bytype_test[t]["S11"] - 0.005 for t in bytype_test)
    verdict = "FAIL" if (sel_is_control or not beats) else ("PASS" if (g1 and g2) else "FAIL")

    result = {"exam": "EXAM 21", "selected": sel, "selected_is_negative_control": sel_is_control,
              "dev_beats_S11": beats,
              "test_S11_r1_r5_r10": [r1("S11", test_ids),
                                     round(sum(hit5["S11"][q] for q in test_ids) / len(test_ids), 4),
                                     round(sum(hit10["S11"][q] for q in test_ids) / len(test_ids), 4)],
              "test_selected_r1_r5_r10": [r1(sel, test_ids),
                                          round(sum(hit5[sel][q] for q in test_ids) / len(test_ids), 4),
                                          round(sum(hit10[sel][q] for q in test_ids) / len(test_ids), 4)],
              "test_paired_r1": {"fixes": f, "breaks": b, "net": net, "mcnemar_p": round(p, 4)},
              "test_by_type_r1": bytype_test,
              "G1": g1, "G2": g2, "VERDICT": verdict}
    D.mkdir(parents=True, exist_ok=True)
    (D / "exam21_result.json").write_text(json.dumps(result, indent=2))
    print("\nRESULT:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
