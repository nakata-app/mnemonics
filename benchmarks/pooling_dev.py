"""EXAM 19 dev instrument: per-session chunk-score pooling inside the frozen S11 pipeline (docs/EXAM19_PREREG.md).

Offline, encoder-free, deterministic. Grid is fixed by the pre-registration:
  P0 max (incumbent) | P1 top2mean | P2 max + 0.5*s2 | P3 top3sum ; scope A = stage-2 only, B = stage-1 w3 too.
Everything else (stage-1 order, RRF k=60 over stage-1 top-20, tie rule, tail) is order_fusion_dev verbatim.

  python3 -I benchmarks/pooling_dev.py                 # dev: LoCoMo + pref2, prints gate verdict
  python3 -I benchmarks/pooling_dev.py --m P2 A        # M exam: one pooling/scope over the EXAM 11 shards (+ P0 step-0)
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import order_fusion_dev as ofd  # noqa: E402

R, I = ofd.R, ofd.I
POOLS = ("P0", "P1", "P2", "P3")
OUT = Path.home() / ".local/share/mnemonics-exam19"


def pool(docs: list, sc: list, p: str) -> dict:
    if len(docs) != len(sc):
        raise ValueError(f"docs/scores length mismatch {len(docs)} != {len(sc)}")
    per: dict = defaultdict(list)
    for d, v in zip(docs, sc):
        if isinstance(v, float) and math.isnan(v):
            raise ValueError("NaN score")
        per[d["sid"]].append(v)
    out = {}
    for sid, vs in per.items():
        vs = sorted(vs, reverse=True)
        s1 = vs[0]
        s2 = vs[1] if len(vs) > 1 else None
        s3 = vs[2] if len(vs) > 2 else None
        if p == "P0":
            out[sid] = s1
        elif p == "P1":
            out[sid] = (s1 + (s2 if s2 is not None else s1)) / 2
        elif p == "P2":
            out[sid] = s1 + 0.5 * (s2 if s2 is not None else 0.0)
        elif p == "P3":
            out[sid] = s1 + (s2 or 0.0) + (s3 or 0.0)
        else:
            raise ValueError(p)
    return out


def s11_order(base: list, docs_w3: list, sc_w3: list, docs_full: list, sc_f4: list, sc_fv: list, p: str, scope: str) -> list:
    p1 = p if scope == "B" else "P0"
    o_w3 = ofd.order_from(pool(docs_w3, sc_w3, p1), base)
    return ofd.two_stage(o_w3, [pool(docs_full, sc_f4, p), pool(docs_full, sc_fv, p)])


def r_at(rows: list, key: str) -> list:
    n = len(rows)
    return [round(sum(x[key][i] for x in rows) / n, 4) for i in range(3)] if n else [0, 0, 0]


def paired(rows: list, a: str, b: str) -> dict:
    f = sum(x[b][0] and not x[a][0] for x in rows)
    br = sum(x[a][0] and not x[b][0] for x in rows)
    return {"fixes": f, "breaks": br, "net": f - br, "mcnemar_p": round(ofd.mcnemar(f, br), 4)}


CONFIGS = ["P0A"] + [f"{p}{s}" for p in POOLS[1:] for s in ("A", "B")]


def run_set(name: str) -> dict:
    c = ofd.SETS[name]
    perq = {r["qid"]: r for r in json.load(open(c["perq"]))}
    docs_w3, docs_full = ofd.jl(c["docs_w3"], "docs"), ofd.jl(c["docs_full"], "docs")
    sc_w3 = ofd.jl(c["w3"], "scores")
    sc_f4, sc_fv = ofd.jl(c["full4b"], "scores"), ofd.jl(c["fullvl"], "scores")
    rows = []
    for qid in sorted(sc_w3):
        r = perq[qid]
        gold = set(r["answer_sids"])
        row = {"qid": qid, "qtype": r["qtype"]}
        for cfg in CONFIGS:
            o = s11_order(r["retrieved_all_sids"], docs_w3[qid], sc_w3[qid], docs_full[qid], sc_f4[qid], sc_fv[qid], cfg[:2], cfg[2])
            row[cfg] = ofd.hits(o, gold)
        rows.append(row)
    res = {"n": len(rows), "systems": {cfg: r_at(rows, cfg) for cfg in CONFIGS},
           "paired_vs_P0A": {cfg: paired(rows, "P0A", cfg) for cfg in CONFIGS[1:]},
           "by_type_r1": {t: {cfg: r_at([x for x in rows if x["qtype"] == t], cfg)[0] for cfg in CONFIGS}
                          for t in sorted({x["qtype"] for x in rows})}}
    return res, rows


def dev() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in ("locomo", "pref2"):
        res, rows = run_set(name)
        out[name] = res
        (OUT / f"dev_{name}_rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    L, P = out["locomo"], out["pref2"]
    # selection rule: highest LoCoMo R@1, then R@5, then fewer breaks
    cands = CONFIGS[1:]
    sel = sorted(cands, key=lambda c: (-L["systems"][c][0], -L["systems"][c][1], L["paired_vs_P0A"][c]["breaks"]))[0]
    s11, cs = L["systems"]["P0A"], L["systems"][sel]
    g1 = (cs[0] - s11[0] >= 0.015) and all(cs[i] >= s11[i] - 0.005 for i in range(3))
    g2 = P["systems"][sel][0] >= P["systems"]["P0A"][0] - 0.005 and P["systems"][sel][2] >= P["systems"]["P0A"][2] - 0.005
    pr = L["paired_vs_P0A"][sel]
    g3 = pr["net"] > 0 and pr["mcnemar_p"] < 0.10
    out["gate"] = {"selected": sel, "G1_locomo_gain_no_regress": g1, "G2_pref2_no_harm": g2, "G3_locomo_net_p": g3,
                   "verdict": "PASS" if (g1 and g2 and g3) else "FAIL"}
    (OUT / "dev_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def m_exam(p: str, scope: str) -> None:
    """One offline pass over M with the EXAM 11 persisted shards. Scope B is not available on M here (see prereg)."""
    if scope != "A":
        raise SystemExit("M exam supports scope A only (prereg: falls back to A and says so)")
    perq = {r["qid"]: r for r in json.load(open(I / "m_ex8_perq.json"))}
    docs = ofd.jl(I / "m_rerank_docs_ex8top20_full3000.jsonl", "docs")
    sc4, scv = {}, {}
    for s in ("s0", "s1"):
        sc4.update(ofd.jl(R / f"mnemonics-m-ex11-4b-{s}/scores.jsonl", "scores"))
        scv.update(ofd.jl(R / f"mnemonics-m-ex11-vl-{s}/scores.jsonl", "scores"))
    assert len(sc4) == len(scv) == 500, (len(sc4), len(scv))
    rows = []
    for qid in sorted(sc4):
        r = perq[qid]
        base, gold = r["ex8"], set(r["answer_sids"])
        row = {"qid": qid, "qtype": r["qtype"], "base": ofd.hits(base, gold)}
        for pp in ("P0", p):
            o = ofd.two_stage(base, [pool(docs[qid], sc4[qid], pp), pool(docs[qid], scv[qid], pp)])
            row[pp] = ofd.hits(o, gold)
        rows.append(row)
    out = {"n": len(rows), "pooling": p, "scope": scope,
           "systems": {k: r_at(rows, k) for k in ("base", "P0", p)},
           "paired_P0_vs_cand": paired(rows, "P0", p),
           "by_type": {t: {k: r_at([x for x in rows if x["qtype"] == t], k) for k in ("base", "P0", p)}
                       for t in sorted({x["qtype"] for x in rows})}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"m_{p}{scope}_rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (OUT / f"m_{p}{scope}_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", nargs=2, metavar=("POOL", "SCOPE"))
    a = ap.parse_args()
    if a.m:
        m_exam(*a.m)
    else:
        dev()
