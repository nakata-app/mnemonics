"""EXAM 20 instrument (docs/EXAM20_PREREG.md): deterministic hypothetical-memory (HyDE) query view.

  python3 -I benchmarks/exam20_hyde.py prompts   # rewrite prompts for LoCoMo+pref2+M -> ~/.local/share/mnemonics-exam20/prompts.jsonl
  python3 -I benchmarks/exam20_hyde.py docs      # dev rerank docs with question := H / Q+H (qid suffix __H / __QH)
  python3 -I benchmarks/exam20_hyde.py dev       # score configs H3 / QH2 vs S11 on LoCoMo + pref2, print gate
  python3 -I benchmarks/exam20_hyde.py mdocs CFG # M rerank docs for the selected config (only after dev PASS)
  python3 -I benchmarks/exam20_hyde.py m CFG     # M exam
Frozen prompt text lives in PROMPT below; it must not change after the first rewrite exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import order_fusion_dev as ofd  # noqa: E402
from pooling_dev import paired, r_at  # noqa: E402

D = Path.home() / ".local/share/mnemonics-exam20"
R, I = ofd.R, ofd.I
PROMPT = ("Question asked about past conversations: {question}\n"
          "Write the ONE sentence from a past conversation that would answer it, phrased exactly as the speaker "
          "would have said it at the time (first person, concrete, specific).\nOutput only that sentence.")
SETQ = {"locomo": I / "locomo_poolall_perq.json", "pref2": I / "pref2_perq.json", "m": I / "m_ex8_perq.json"}


def perq(name: str) -> dict:
    return {r["qid"]: r for r in json.load(open(SETQ[name]))}


def prompts() -> None:
    D.mkdir(parents=True, exist_ok=True)
    n = {}
    with open(D / "prompts.jsonl", "w") as f:
        for name in ("locomo", "pref2", "m"):
            qs = perq(name)
            for qid in sorted(qs):
                f.write(json.dumps({"qid": qid, "prompt": PROMPT.format(question=qs[qid]["question"].strip())}) + "\n")
            n[name] = len(qs)
    print(json.dumps(n))


def rewrites() -> dict:
    return {json.loads(l)["qid"]: json.loads(l)["raw"].strip().split("\n")[0].strip() for l in open(D / "kaggle/raw.jsonl") if l.strip()}


def write_docs(names: list, docs_key: str, out: Path, allow_m: bool = False) -> None:
    rw = rewrites()
    with open(out, "w") as f:
        for name in names:
            qs = perq(name)
            src = ofd.SETS[name][docs_key] if name != "m" else I / "m_rerank_docs_ex8top20_full3000.jsonl"
            for line in open(src):
                if not line.strip():
                    continue
                r = json.loads(line)
                q, h = qs[r["qid"]]["question"].strip(), rw[r["qid"]]
                f.write(json.dumps({"qid": r["qid"] + "__H", "question": h, "docs": r["docs"]}) + "\n")
                f.write(json.dumps({"qid": r["qid"] + "__QH", "question": f"{q}\nLikely memory: {h}", "docs": r["docs"]}) + "\n")
    print("wrote", out)


def orders(name: str, sc_h: dict):
    c = ofd.SETS[name]
    qs = perq(name)
    docs_w3, docs_full = ofd.jl(c["docs_w3"], "docs"), ofd.jl(c["docs_full"], "docs")
    sc_w3, sc_f4, sc_fv = ofd.jl(c["w3"], "scores"), ofd.jl(c["full4b"], "scores"), ofd.jl(c["fullvl"], "scores")
    rows = []
    for qid in sorted(sc_w3):
        r = qs[qid]
        gold = set(r["answer_sids"])
        o_w3 = ofd.order_from(ofd.session_max(docs_w3[qid], sc_w3[qid]), r["retrieved_all_sids"])
        b4, bv = ofd.session_max(docs_full[qid], sc_f4[qid]), ofd.session_max(docs_full[qid], sc_fv[qid])
        bh = ofd.session_max(docs_full[qid], sc_h[qid + "__H"])
        bqh = ofd.session_max(docs_full[qid], sc_h[qid + "__QH"])
        rows.append({"qid": qid, "qtype": r["qtype"],
                     "S11": ofd.hits(ofd.two_stage(o_w3, [b4, bv]), gold),
                     "H3": ofd.hits(ofd.two_stage(o_w3, [b4, bv, bh]), gold),
                     "QH2": ofd.hits(ofd.two_stage(o_w3, [bqh, bv]), gold)})
    return rows


def dev() -> None:
    sc_h = ofd.jl(D / "kaggle-dev/scores.jsonl", "scores")
    out = {}
    for name in ("locomo", "pref2"):
        rows = orders(name, sc_h)
        (D / f"dev_{name}_rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        out[name] = {"n": len(rows), "systems": {k: r_at(rows, k) for k in ("S11", "H3", "QH2")},
                     "paired_vs_S11": {k: paired(rows, "S11", k) for k in ("H3", "QH2")},
                     "by_type_r1": {t: {k: r_at([x for x in rows if x["qtype"] == t], k)[0] for k in ("S11", "H3", "QH2")}
                                    for t in sorted({x["qtype"] for x in rows})}}
    L, P = out["locomo"], out["pref2"]
    sel = sorted(("H3", "QH2"), key=lambda k: (-L["systems"][k][0], -L["systems"][k][1], L["paired_vs_S11"][k]["breaks"]))[0]
    s, c = L["systems"]["S11"], L["systems"][sel]
    g1 = c[0] - s[0] >= 0.015 and all(c[i] >= s[i] - 0.005 for i in range(3))
    g2 = P["systems"][sel][0] >= P["systems"]["S11"][0] - 0.005 and P["systems"][sel][2] >= P["systems"]["S11"][2] - 0.005
    pr = L["paired_vs_S11"][sel]
    g3 = pr["net"] > 0 and pr["mcnemar_p"] < 0.10
    out["gate"] = {"selected": sel, "G1": g1, "G2": g2, "G3": g3, "verdict": "PASS" if (g1 and g2 and g3) else "FAIL"}
    (D / "dev_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def m_exam(cfg: str) -> None:
    qs = perq("m")
    docs = ofd.jl(I / "m_rerank_docs_ex8top20_full3000.jsonl", "docs")
    sc4, scv = {}, {}
    for s in ("s0", "s1"):
        sc4.update(ofd.jl(R / f"mnemonics-m-ex11-4b-{s}/scores.jsonl", "scores"))
        scv.update(ofd.jl(R / f"mnemonics-m-ex11-vl-{s}/scores.jsonl", "scores"))
    sc_h = ofd.jl(D / "kaggle/m_scores.jsonl", "scores")
    rows = []
    for qid in sorted(sc4):
        r = qs[qid]
        base, gold = r["ex8"], set(r["answer_sids"])
        b4, bv = ofd.session_max(docs[qid], sc4[qid]), ofd.session_max(docs[qid], scv[qid])
        if cfg == "H3":
            cand = ofd.two_stage(base, [b4, bv, ofd.session_max(docs[qid], sc_h[qid + "__H"])])
        else:
            cand = ofd.two_stage(base, [ofd.session_max(docs[qid], sc_h[qid + "__QH"]), bv])
        rows.append({"qid": qid, "qtype": r["qtype"], "base": ofd.hits(base, gold),
                     "P0": ofd.hits(ofd.two_stage(base, [b4, bv]), gold), cfg: ofd.hits(cand, gold)})
    out = {"n": len(rows), "config": cfg, "systems": {k: r_at(rows, k) for k in ("base", "P0", cfg)},
           "paired_P0_vs_cand": paired(rows, "P0", cfg),
           "by_type": {t: {k: r_at([x for x in rows if x["qtype"] == t], k) for k in ("base", "P0", cfg)}
                       for t in sorted({x["qtype"] for x in rows})}}
    (D / f"m_{cfg}_rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (D / f"m_{cfg}_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prompts":
        prompts()
    elif cmd == "docs":
        write_docs(["locomo", "pref2"], "docs_full", D / "dev_rerank_docs.jsonl")
    elif cmd == "dev":
        dev()
    elif cmd == "mdocs":
        write_docs(["m"], "", D / "m_rerank_docs.jsonl")
    elif cmd == "m":
        m_exam(sys.argv[2])
