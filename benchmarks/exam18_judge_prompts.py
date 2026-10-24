"""EXAM 18 part 2 (pre-registered path): build the deterministic judge prompts for the Kaggle Qwen3-14B kernel.

Encoder-free and offline. For LoCoMo (200) and pref2 (80) it recomputes the frozen S11 / EXAM 11 order
(order_fusion_dev, verbatim), applies the frozen router (temporal_router_dev.is_temporal) and, for routed
queries only, emits the judge prompt over the S11 top-K (K=10). The prompt text is the one fixed in
temporal_judge_dev.judge_order BEFORE any judge output was seen; only the execution engine changes
(Evren GLM-5.3 -> Kaggle Qwen3-14B NF4 greedy, see krun-kernels/mnemonics-locomo-qwen14b-judge/run.py).

Outputs (default ~/.local/share/mnemonics-exam18/):
  prompts.jsonl       {qid, prompt}            -> Kaggle dataset input for the kernel
  judge_inputs.jsonl  {qid, set, qtype, routed, s11, gold, top}  -> local sidecar for exam18_judge_gate.py
Non-routed queries are NOT sent to the judge; the gate script keeps their S11 order byte-for-byte.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import order_fusion_dev as ofd  # noqa: E402
from temporal_judge_dev import K, sid_text  # noqa: E402  (frozen prompt pieces)
from temporal_router_dev import is_temporal  # noqa: E402

OUT = Path.home() / ".local/share/mnemonics-exam18"


def build_prompt(question: str, top: list, docs: list) -> str:
    blocks = [f"[{i}] {sid_text(docs, sid)}" for i, sid in enumerate(top)]
    return (
        "You pick which past conversation session answers a time-related question.\n"
        f"Question: {question}\n\nCandidate sessions (each begins with its date):\n"
        + "\n\n".join(blocks)
        + "\n\nRank the candidate numbers best-first by how well the session answers the question's time "
        "constraint. Return ONLY the numbers, comma-separated, best first. Example: 3,0,7"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pf = open(OUT / "prompts.jsonl", "w")
    sf = open(OUT / "judge_inputs.jsonl", "w")
    stats = {}
    for name in ("locomo", "pref2"):
        c = ofd.SETS[name]
        perq = {r["qid"]: r for r in json.load(open(c["perq"]))}
        docs_w3, docs_full = ofd.jl(c["docs_w3"], "docs"), ofd.jl(c["docs_full"], "docs")
        sc_w3 = ofd.jl(c["w3"], "scores")
        sc_f4, sc_fv = ofd.jl(c["full4b"], "scores"), ofd.jl(c["fullvl"], "scores")
        n = routed = 0
        for qid in sorted(sc_w3):
            r = perq[qid]
            o_w3 = ofd.order_from(ofd.session_max(docs_w3[qid], sc_w3[qid]), r["retrieved_all_sids"])
            s11 = ofd.two_stage(o_w3, [ofd.session_max(docs_full[qid], sc_f4[qid]),
                                       ofd.session_max(docs_full[qid], sc_fv[qid])])
            rt = is_temporal(r["question"])
            top = s11[:K]
            n += 1
            routed += rt
            if rt:
                pf.write(json.dumps({"qid": qid, "prompt": build_prompt(r["question"], top, docs_full[qid])}) + "\n")
            sf.write(json.dumps({"qid": qid, "set": name, "qtype": r["qtype"], "routed": rt,
                                 "s11": s11, "gold": sorted(set(r["answer_sids"])), "top": top}) + "\n")
        stats[name] = {"n": n, "routed": routed}
    pf.close(); sf.close()
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
