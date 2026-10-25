"""EXAM 22: stronger stage-2 reranker (Qwen3-Reranker-8B) over the S11 ex8top20 pool, held-out M split.

Pre-registered in docs/EXAM22_PREREG.md (committed before any 8B score existed). Reuses the exact EXAM 11
fusion (score_rerank_docs.py lines 43-50: session_max per session, equal-weight RRF k=60 over stage-2 scorers,
ties by the ex8 base order, unscored sessions keep base order). Only the stage-2 score set changes:
  C0 = RRF(4B, VL)      = S11 (step 0: must reproduce the persisted EXAM 11 order)
  C1 = 8B               C2 = RRF(8B, VL)   C3 = RRF(8B, 4B)   C4 = RRF(8B, 4B, VL)
Select on dev-M (seed 21), report test-M. No GPU here (reads persisted score files); deterministic.

Run: python3 exam22_stronger_rerank.py        (C0-only validation if 8B scores are not yet present)
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path

R = Path("/Volumes/WD-Backup/datasets/longmemeval/results")
I = Path("/Volumes/WD-Backup/datasets/longmemeval/rerank_inputs")
SPLIT = Path.home() / ".local/share/mnemonics-exam21/split.json"
OUT = Path.home() / ".local/share/mnemonics-exam22"
CALLS = R / "exam11_m_scored" / "calls.jsonl"
PERQ = I / "m_ex8_perq.json"  # base = ex8 order (the frozen EXAM 11 build: --perq m_ex8_perq --sids-key ex8)
DOCS = I / "m_rerank_docs_ex8top20_full3000.jsonl"
SC = {
    "4b": [R / "mnemonics-m-ex11-4b-s0/scores.jsonl", R / "mnemonics-m-ex11-4b-s1/scores.jsonl"],
    "vl": [R / "mnemonics-m-ex11-vl-s0/scores.jsonl", R / "mnemonics-m-ex11-vl-s1/scores.jsonl"],
    "8b": [R / "mnemonics-m-ex22-8b-s0/scores.jsonl", R / "mnemonics-m-ex22-8b-s1/scores.jsonl"],
}


def merge(paths):
    out = {}
    for p in paths:
        if not p.exists():
            return None
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                out[r["qid"]] = r["scores"]
    return out


def session_max(docs, sc):
    # Floor at -1.0, byte-identical to the frozen score_rerank_docs.py (EXAM 11 instrument): VL logits below
    # -1.0 are clamped, so VL is largely washed out in S11. 4B/8B reranker scores are >=0, unaffected.
    if len(docs) != len(sc):
        raise ValueError(f"len mismatch {len(docs)} {len(sc)}")
    best = {}
    for d, v in zip(docs, sc):
        best[d["sid"]] = max(best.get(d["sid"], -1.0), v)
    return best


def fuse(base, docs, score_sets):
    """Replicate score_rerank_docs.py 43-50 for 1..n stage-2 scorers (equal-weight RRF k=60)."""
    bests = [session_max(docs, s) for s in score_sets]
    keys = bests[0]  # all scorers share the same docs -> same session set
    if len(bests) == 1:
        fused = {s: bests[0][s] for s in keys}
    else:
        rmaps = []
        for b in bests:
            srt = sorted(b, key=lambda s: (-b[s], base.index(s)))
            rmaps.append({s: i for i, s in enumerate(srt)})
        fused = {s: sum(1 / (60 + rm[s]) for rm in rmaps) for s in keys}
    ranked = sorted(fused, key=lambda s: (-fused[s], base.index(s)))
    return ranked + [s for s in base if s not in fused]


def hits(order, gold, k):
    return 1 if any(s in gold for s in order[:k]) else 0


def regen_split_assert():
    e11 = [json.loads(l) for l in open(CALLS) if l.strip()]
    byt = defaultdict(list)
    for r in e11:
        byt[r["qtype"]].append(r["qid"])
    rng = random.Random(21)
    dev, test = [], []
    for t in sorted(byt):
        ids = sorted(byt[t]); rng.shuffle(ids); h = len(ids) // 2
        dev += ids[:h]; test += ids[h:]
    saved = json.load(open(SPLIT))
    assert saved["dev"] == sorted(dev) and saved["test"] == sorted(test), "split seed 21 mismatch"
    return set(dev), set(test)


def main():
    dev, test = regen_split_assert()
    calls = {r["qid"]: r for r in (json.loads(l) for l in open(CALLS) if l.strip())}
    perq = {r["qid"]: r for r in json.load(open(PERQ))}
    base_of = {q: perq[q]["ex8"] for q in calls}
    gold_of = {q: set(perq[q]["answer_sids"]) for q in calls}
    docs = {json.loads(l)["qid"]: json.loads(l)["docs"] for l in open(DOCS) if l.strip()}
    sc4b, scvl, sc8b = merge(SC["4b"]), merge(SC["vl"]), merge(SC["8b"])
    assert sc4b and scvl, "4B/VL scores missing"

    # C0 reproduction check (instrument): frozen EXAM 11 build = RRF(4B,VL) over the ex8 base
    mism = 0
    for qid in calls:
        o = fuse(base_of[qid], docs[qid], [sc4b[qid], scvl[qid]])
        if o[:75] != calls[qid]["order"]:
            mism += 1
    print(f"step 0: C0=RRF(4B,VL) reproduces EXAM 11 order: mismatches={mism}/{len(calls)} ({'OK' if mism == 0 else 'FAIL'})")
    if mism:
        raise SystemExit("C0 does not reproduce EXAM 11 — fix instrument before any 8B comparison")
    if sc8b is None:
        print("8B scores not present yet — C0-only validation done.")
        return

    configs = {"C0_4b_vl": [sc4b, scvl], "C1_8b": [sc8b], "C2_8b_vl": [sc8b, scvl],
               "C3_8b_4b": [sc8b, sc4b], "C4_8b_4b_vl": [sc8b, sc4b, scvl]}
    r1 = {c: {} for c in configs}; r5 = {c: {} for c in configs}; r10 = {c: {} for c in configs}
    for qid in calls:
        gold = gold_of[qid]
        for cfg, sets in configs.items():
            o = fuse(base_of[qid], docs[qid], [s[qid] for s in sets])
            r1[cfg][qid] = hits(o, gold, 1); r5[cfg][qid] = hits(o, gold, 5); r10[cfg][qid] = hits(o, gold, 10)

    def R1(cfg, ids): return round(sum(r1[cfg][q] for q in ids) / len(ids), 4)
    dev_ids, test_ids = sorted(dev), sorted(test)
    qtypes = {q: calls[q]["qtype"] for q in calls}

    cand = ["C1_8b", "C2_8b_vl", "C3_8b_4b", "C4_8b_4b_vl"]
    pref = {"C2_8b_vl": 0, "C4_8b_4b_vl": 1, "C3_8b_4b": 2, "C1_8b": 3}
    base_dev = R1("C0_4b_vl", dev_ids)
    sel = min(cand, key=lambda c: (-R1(c, dev_ids), pref[c]))
    beats = R1(sel, dev_ids) > base_dev

    print(f"\ndev-M R@1: C0(S11)={base_dev}")
    for c in cand:
        print(f"  {c:14s} {R1(c, dev_ids)}")
    print(f"dev-M selection = {sel} (beats S11: {beats})")

    f = b = 0
    for q in test_ids:
        s, cc = r1["C0_4b_vl"][q], r1[sel][q]
        f += cc and not s; b += s and not cc
    net = f - b; n = f + b
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(f, b) + 1)) / 2 ** n)
    bytype = {}
    for t in sorted(set(qtypes[q] for q in test_ids)):
        qs = [q for q in test_ids if qtypes[q] == t]
        bytype[t] = {"n": len(qs), "C0": round(sum(r1["C0_4b_vl"][q] for q in qs) / len(qs), 3),
                     sel: round(sum(r1[sel][q] for q in qs) / len(qs), 3)}
    g1 = beats and R1(sel, test_ids) > R1("C0_4b_vl", test_ids) and net > 0 and p < 0.10
    g2 = all(bytype[t][sel] >= bytype[t]["C0"] - 0.005 for t in bytype)
    verdict = "PASS" if (beats and g1 and g2) else "FAIL"
    result = {"exam": "EXAM 22", "selected": sel, "dev_beats_S11": beats,
              "test_C0_r1_r5_r10": [R1("C0_4b_vl", test_ids),
                                    round(sum(r5["C0_4b_vl"][q] for q in test_ids) / len(test_ids), 4),
                                    round(sum(r10["C0_4b_vl"][q] for q in test_ids) / len(test_ids), 4)],
              "test_sel_r1_r5_r10": [R1(sel, test_ids),
                                     round(sum(r5[sel][q] for q in test_ids) / len(test_ids), 4),
                                     round(sum(r10[sel][q] for q in test_ids) / len(test_ids), 4)],
              "test_paired_r1": {"fixes": f, "breaks": b, "net": net, "mcnemar_p": round(p, 4)},
              "test_by_type_r1": bytype, "dev_all": {c: R1(c, dev_ids) for c in configs},
              "VERDICT": verdict}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exam22_result.json").write_text(json.dumps(result, indent=2))
    print("\nRESULT:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
