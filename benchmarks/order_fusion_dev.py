"""EXAM 17 dev instrument: equal-weight RRF of the EXAM 10 and EXAM 11 analogue orders.

Encoder-free and offline: it only reads persisted reranker score files, so no embedding model is loaded.

  S10 = RRF(k=60) of 4B-w3 default and 4B-w3 personalization (session = max over docs, ties by base order)
  S11 = stage 1 the 4B-w3 default order; stage 2 RRF(k=60) of 4B-full3000 and VL-8B-full3000 over the stage-1 top-20,
        rest keeps stage-1 order (same rule as score_rerank_docs.py --sids-key ex8 for EXAM 11)
  F   = RRF(k=60) over the S10 and S11 orders (rank = position in each order), ties by S11 order
  X4  = exploratory only: flat RRF of all four signals over the stage-1 top-20

`--m-calls A B` applies F to two persisted calls.jsonl files (the EXAM 17 M run). It is not used on dev.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

K = 60
KS = (1, 5, 10)
R = Path("/Volumes/WD-Backup/datasets/longmemeval/results")
I = Path("/Volumes/WD-Backup/datasets/longmemeval/rerank_inputs")

SETS = {
    "locomo": {
        "perq": I / "locomo_poolall_perq.json",
        "docs_w3": I / "locomo_rerank_docs_top30_w3.jsonl",
        "docs_full": I / "locomo_rerank_docs_top30_full3000.jsonl",
        "w3": R / "locomo_qrr4b/scores.jsonl",
        "pers": R / "mnemonics-locomo-qrr4b-pers/scores.jsonl",
        "full4b": R / "mnemonics-locomo-qrr4b-full/scores.jsonl",
        "fullvl": R / "mnemonics-locomo-qvlrr8b-full/scores.jsonl",
    },
    "pref2": {
        "perq": I / "pref2_perq.json",
        "docs_w3": I / "pref2_rerank_docs_top40_w3.jsonl",
        "docs_full": I / "pref2_rerank_docs_top40_full3000.jsonl",
        "w3": R / "mnemonics-pref2-qrr4b/scores.jsonl",
        "pers": R / "mnemonics-pref2-qrr4b-pers/scores.jsonl",
        "full4b": R / "mnemonics-pref2-qrr4b-full/scores.jsonl",
        "fullvl": R / "mnemonics-pref2-qvlrr8b-full/scores.jsonl",
    },
}


def jl(path: Path, key: str) -> dict:
    out = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            out[r["qid"]] = r[key]
    return out


def session_max(docs: list, sc: list) -> dict:
    if len(docs) != len(sc):
        raise ValueError(f"docs/scores length mismatch {len(docs)} != {len(sc)}")
    best: dict = {}
    for d, v in zip(docs, sc):
        if isinstance(v, float) and math.isnan(v):
            raise ValueError("NaN score")
        best[d["sid"]] = max(best.get(d["sid"], -1e30), v)
    return best


def ranks(score: dict, tie: list) -> dict:
    pos = {s: i for i, s in enumerate(tie)}
    return {s: i for i, s in enumerate(sorted(score, key=lambda s: (-score[s], pos[s])))}


def rrf(rank_maps: list, tie: list, pool: list | None = None) -> list:
    """RRF over rank maps; a session missing from a map gets no contribution from it. Ties by `tie` order."""
    pool = pool if pool is not None else tie
    pos = {s: i for i, s in enumerate(tie)}
    fused = {s: sum(1 / (K + m[s]) for m in rank_maps if s in m) for s in pool}
    return sorted(pool, key=lambda s: (-fused[s], pos[s]))


def order_from(score: dict, base: list) -> list:
    pos = {s: i for i, s in enumerate(base)}
    return sorted(score, key=lambda s: (-score[s], pos[s])) + [s for s in base if s not in score]


def two_stage(stage1: list, score_dicts: list, n: int = 20) -> list:
    top = stage1[:n]
    scored = [s for s in top if all(s in d for d in score_dicts)]
    rmaps = [ranks({s: d[s] for s in scored}, stage1) for d in score_dicts]
    return rrf(rmaps, stage1, scored) + [s for s in stage1 if s not in scored]


def hits(order: list, gold: set) -> list:
    return [any(s in gold for s in order[:k]) for k in KS]


def mcnemar(f: int, b: int) -> float:
    n = f + b
    if n == 0:
        return 1.0
    x = min(f, b)
    p = sum(math.comb(n, i) for i in range(x + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summarize(name: str, rows: list, systems: list, ref: str) -> dict:
    n = len(rows)
    out = {"set": name, "n": n, "systems": {}, "paired_vs_" + ref: {}}
    for s in systems:
        out["systems"][s] = [round(sum(r[s][i] for r in rows) / n, 4) for i in range(3)]
    for s in systems:
        if s == ref:
            continue
        f = sum(r[s][0] and not r[ref][0] for r in rows)
        b = sum(r[ref][0] and not r[s][0] for r in rows)
        out["paired_vs_" + ref][s] = {"fixes": f, "breaks": b, "net": f - b, "mcnemar_p": round(mcnemar(f, b), 4)}
    bt = defaultdict(list)
    for r in rows:
        bt[r["qtype"]].append(r)
    out["by_type_r1"] = {t: {"n": len(v), **{s: round(sum(r[s][0] for r in v) / len(v), 3) for s in systems}}
                         for t, v in sorted(bt.items())}
    return out


def run_set(name: str, out_dir: Path) -> dict:
    c = SETS[name]
    perq = {r["qid"]: r for r in json.load(open(c["perq"]))}
    docs_w3 = jl(c["docs_w3"], "docs")
    sc_w3, sc_pers = jl(c["w3"], "scores"), jl(c["pers"], "scores")
    have_full = all(Path(c[k]).exists() for k in ("docs_full", "full4b", "fullvl"))
    if have_full:
        docs_full = jl(c["docs_full"], "docs")
        sc_f4, sc_fv = jl(c["full4b"], "scores"), jl(c["fullvl"], "scores")
    rows = []
    for qid in sorted(sc_w3):
        r = perq[qid]
        base = r["retrieved_all_sids"]
        gold = set(r["answer_sids"])
        b_w3 = session_max(docs_w3[qid], sc_w3[qid])
        b_pers = session_max(docs_w3[qid], sc_pers[qid])
        o_w3 = order_from(b_w3, base)
        s10 = rrf([ranks(b_w3, base), ranks(b_pers, base)], base, list(b_w3)) + [s for s in base if s not in b_w3]
        row = {"qid": qid, "qtype": r["qtype"], "base": hits(base, gold), "w3": hits(o_w3, gold), "S10": hits(s10, gold)}
        if have_full:
            b_f4 = session_max(docs_full[qid], sc_f4[qid])
            b_fv = session_max(docs_full[qid], sc_fv[qid])
            s11 = two_stage(o_w3, [b_f4, b_fv])
            f = rrf([{s: i for i, s in enumerate(s10)}, {s: i for i, s in enumerate(s11)}], s11)
            x4 = two_stage(o_w3, [b_w3, b_pers, b_f4, b_fv])
            row.update({"S11": hits(s11, gold), "F": hits(f, gold), "X4_exploratory": hits(x4, gold),
                        "top1": {"S10": s10[0], "S11": s11[0], "F": f[0]}})
        rows.append(row)
    systems = ["base", "w3", "S10"] + (["S11", "F", "X4_exploratory"] if have_full else [])
    summ = summarize(name, rows, systems, "S11" if have_full else "S10")
    if have_full:
        agree = [r for r in rows if r["top1"]["S10"] == r["top1"]["S11"]]
        summ["agreement"] = {
            "S10_S11_agree_at_1": len(agree),
            "hit_rate_when_agree": round(sum(r["S11"][0] for r in agree) / max(1, len(agree)), 4),
            "disagree": len(rows) - len(agree),
            "disagree_S10_right": sum(r["S10"][0] for r in rows if r not in agree),
            "disagree_S11_right": sum(r["S11"][0] for r in rows if r not in agree),
            "disagree_F_right": sum(r["F"][0] for r in rows if r not in agree),
            "union_oracle_r1": round(sum(r["S10"][0] or r["S11"][0] for r in rows) / len(rows), 4),
        }
    else:
        summ["blocked"] = "full-chunk 4B/VL score files missing for this set; S11 and F not computable"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}_rows.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def run_m(a: Path, b: Path, out_dir: Path) -> dict:
    """EXAM 17 M pass: a = EXAM 10 calls, b = EXAM 11 calls. Only for the pre-registered M run."""
    A, B = {}, {}
    for path, dst in ((a, A), (b, B)):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                dst[r["qid"]] = r
    assert set(A) == set(B), "qid sets differ"
    rows = []
    for qid in sorted(B):
        oa, ob = A[qid]["order"], B[qid]["order"]
        gold = set(B[qid]["gold"])
        pool = ob + [s for s in oa if s not in set(ob)]
        f = rrf([{s: i for i, s in enumerate(oa)}, {s: i for i, s in enumerate(ob)}], pool)
        rows.append({"qid": qid, "qtype": B[qid]["qtype"], "EXAM10": hits(oa, gold), "EXAM11": hits(ob, gold),
                     "F": hits(f, gold)})
    summ = summarize("M", rows, ["EXAM10", "EXAM11", "F"], "EXAM11")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "M_summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["locomo", "pref2"])
    ap.add_argument("--out", type=Path, default=Path.home() / ".local/share/mnemonics-exam17")
    ap.add_argument("--m-calls", nargs=2, type=Path, help="EXAM 17 M run only: EXAM10 calls, EXAM11 calls")
    args = ap.parse_args()
    if args.m_calls:
        print(json.dumps(run_m(*args.m_calls, args.out), indent=2))
        return
    for s in args.sets:
        print(json.dumps(run_set(s, args.out), indent=2))


if __name__ == "__main__":
    main()
