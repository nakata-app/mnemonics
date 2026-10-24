"""EXAM 18 part 2: score the Kaggle Qwen3-14B judge replies against the pre-registered dev gate
(docs/EXAM18_PREREG.md). Offline, encoder-free, deterministic.

Inputs (default ~/.local/share/mnemonics-exam18/):
  judge_inputs.jsonl  from exam18_judge_prompts.py  {qid, set, qtype, routed, s11, gold, top}
  kaggle/raw.jsonl    from the kernel               {qid, raw}
Routed queries: the judge permutation of the S11 top-K (parse_rank, frozen in temporal_judge_dev) replaces the
top-K; the S11 tail is kept. Non-routed queries keep S11 byte-for-byte. A routed qid with no reply is counted as
judge_failed and keeps S11 (reported, never hidden).

Gate (verbatim from the pre-registration, evaluated mechanically; PASS requires all):
  G1 LoCoMo R@1(judge) - R@1(S11) >= +0.01 AND no R@k(judge) < R@k(S11) - 0.005
  G2 pref2  R@1(judge) >= R@1(S11) - 0.005
  G3 LoCoMo paired net (fixes - breaks) > 0, fixes concentrated in routed queries (reported per routed/non-routed;
     non-routed is identical to S11 by construction, so all fixes are routed)
  G4 encoder pin + byte-identical re-run (this script is deterministic; the kernel prints its repeat-check)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import order_fusion_dev as ofd  # noqa: E402
from temporal_judge_dev import parse_rank  # noqa: E402

D = Path.home() / ".local/share/mnemonics-exam18"


def r_at(rows: list, key: str) -> list:
    n = len(rows)
    return [round(sum(x[key][i] for x in rows) / n, 4) for i in range(3)] if n else [0, 0, 0]


def paired(rows: list) -> dict:
    f = sum(x["judge"][0] and not x["S11"][0] for x in rows)
    b = sum(x["S11"][0] and not x["judge"][0] for x in rows)
    return {"fixes": f, "breaks": b, "net": f - b, "mcnemar_p": round(ofd.mcnemar(f, b), 4)}


def main() -> None:
    raw = {}
    for line in open(D / "kaggle/raw.jsonl"):
        if line.strip():
            r = json.loads(line)
            raw[r["qid"]] = r["raw"]
    rows = []
    for line in open(D / "judge_inputs.jsonl"):
        q = json.loads(line)
        gold, s11 = set(q["gold"]), q["s11"]
        judged, failed = s11, False
        if q["routed"]:
            if q["qid"] in raw:
                top = q["top"]
                judged = [top[i] for i in parse_rank(raw[q["qid"]], len(top))] + s11[len(top):]
            else:
                failed = True
        rows.append({**{k: q[k] for k in ("qid", "set", "qtype", "routed")}, "judge_failed": failed,
                     "S11": ofd.hits(s11, gold), "judge": ofd.hits(judged, gold)})

    out = {"judge_replies": len(raw), "sets": {}}
    for name in ("locomo", "pref2"):
        rs = [x for x in rows if x["set"] == name]
        rt = [x for x in rs if x["routed"]]
        bytype = {}
        for t in sorted({x["qtype"] for x in rs}):
            sub = [x for x in rs if x["qtype"] == t]
            bytype[t] = {"n": len(sub), "S11_r1": r_at(sub, "S11")[0], "judge_r1": r_at(sub, "judge")[0]}
        out["sets"][name] = {
            "n": len(rs), "routed": len(rt), "judge_failed": sum(x["judge_failed"] for x in rs),
            "S11": r_at(rs, "S11"), "judge": r_at(rs, "judge"),
            "paired_all": paired(rs), "paired_routed": paired(rt), "paired_nonrouted": paired([x for x in rs if not x["routed"]]),
            "by_type_r1": bytype,
        }
    L, P = out["sets"]["locomo"], out["sets"]["pref2"]
    g1 = (L["judge"][0] - L["S11"][0] >= 0.01) and all(L["judge"][i] >= L["S11"][i] - 0.005 for i in range(3))
    g2 = P["judge"][0] >= P["S11"][0] - 0.005
    g3 = L["paired_all"]["net"] > 0
    out["gate"] = {"G1_locomo_r1_plus_no_regress": g1, "G2_pref2_no_harm": g2, "G3_locomo_net_positive": g3,
                   "verdict": "PASS" if (g1 and g2 and g3) else "FAIL",
                   "note": "G4 (encoder pin / byte-identical) is procedural: see kernel repeat-check and ledger"}
    (D / "judge_gate_rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (D / "judge_gate_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
