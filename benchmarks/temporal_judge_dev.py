"""EXAM 18 dev instrument, part 2 of 2: the temporal judge arm (EXPLORATORY, Evren-hosted).

Applies a chat-model judge to the S11 / EXAM 11 order on routed-temporal queries only (router = part 1,
temporal_router_dev.py, frozen). For a routed query the judge re-reads the S11 top-K sessions with the
question's time constraint made explicit and returns a best-first ranking of those K; non-routed queries keep
the S11 order byte-for-byte, and the S11 tail (below K) is always kept.

IMPORTANT -- this is an EXPLORATORY probe of the judge HYPOTHESIS, not the deterministic artifact the M R@1
goal needs. The judge model here is Evren's GLM-5.3, a hosted reasoning model that is NOT reproducible even at
temperature 0 (measured: identical prompt gave "0,1" then "0" on two calls). So:
  - a PASS here does NOT satisfy the deterministic requirement; it only justifies building the local
    deterministic Qwen3-14B judge on GPU.
  - the pre-registered dev gate (docs/EXAM18_PREREG.md) is LoCoMo + pref2; pref2 S11 does not exist yet, so
    this script measures the LoCoMo arm ONLY and MUST NOT be read as the gate passing.
Deviation from the pre-registration, fixed BEFORE seeing any score: judge model Qwen3-14B(Kaggle) -> GLM-5.3
(Evren), K=10, temperature 0, max_tokens 2048. Raw judge replies are cached for audit.

Usage: EVREN_BASE_URL/EVREN_API_KEY set; MNEMONICS_ENCODER_MODEL pinned (not loaded here).
  python3 benchmarks/temporal_judge_dev.py            # LoCoMo routed-temporal queries
Outputs ~/.local/share/mnemonics-exam18/judge_locomo_{summary.json,rows.jsonl,raw.jsonl}.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import order_fusion_dev as ofd  # noqa: E402  (frozen S11 logic reused verbatim)
from temporal_router_dev import is_temporal  # noqa: E402  (frozen router)

MODEL = "glm-5.3"
K = 10
MAXTOK = 2048
OUT = Path.home() / ".local/share/mnemonics-exam18"


def evren(prompt: str) -> str:
    base = os.environ["EVREN_BASE_URL"].rstrip("/")
    body = json.dumps({
        "model": MODEL, "temperature": 0, "max_tokens": MAXTOK,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        base + "/chat/completions", body,
        {"Authorization": "Bearer " + os.environ["EVREN_API_KEY"], "Content-Type": "application/json"},
    )
    d = json.load(urllib.request.urlopen(req, timeout=180))
    return d["choices"][0]["message"].get("content") or ""


def parse_rank(text: str, k: int) -> list[int]:
    """First k distinct integers in 0..k-1 as they appear; missing ones appended in original order."""
    seen, out = set(), []
    tok = ""
    for ch in text:
        if ch.isdigit():
            tok += ch
        else:
            if tok:
                v = int(tok)
                if 0 <= v < k and v not in seen:
                    seen.add(v); out.append(v)
                tok = ""
    if tok:
        v = int(tok)
        if 0 <= v < k and v not in seen:
            seen.add(v); out.append(v)
    out += [i for i in range(k) if i not in seen]
    return out


def sid_text(docs: list, sid: str, cap: int = 1500) -> str:
    parts = [d["text"] for d in docs if d["sid"] == sid]
    return (" ".join(parts))[:cap]


def judge_order(question: str, s11: list, docs: list) -> tuple[list, str]:
    top = s11[:K]
    blocks = [f"[{i}] {sid_text(docs, sid)}" for i, sid in enumerate(top)]
    prompt = (
        "You pick which past conversation session answers a time-related question.\n"
        f"Question: {question}\n\nCandidate sessions (each begins with its date):\n"
        + "\n\n".join(blocks)
        + "\n\nRank the candidate numbers best-first by how well the session answers the question's time "
        "constraint. Return ONLY the numbers, comma-separated, best first. Example: 3,0,7"
    )
    raw = evren(prompt)
    perm = parse_rank(raw, len(top))
    reordered = [top[i] for i in perm]
    return reordered + s11[K:], raw


def main() -> None:
    name = "locomo"
    c = ofd.SETS[name]
    perq = {r["qid"]: r for r in json.load(open(c["perq"]))}
    docs_w3 = ofd.jl(c["docs_w3"], "docs")
    sc_w3, sc_pers = ofd.jl(c["w3"], "scores"), ofd.jl(c["pers"], "scores")
    docs_full = ofd.jl(c["docs_full"], "docs")
    sc_f4, sc_fv = ofd.jl(c["full4b"], "scores"), ofd.jl(c["fullvl"], "scores")

    OUT.mkdir(parents=True, exist_ok=True)
    raw_fh = open(OUT / "judge_locomo_raw.jsonl", "w")
    rows = []
    routed_ct = 0
    for qid in sorted(sc_w3):
        r = perq[qid]
        base = r["retrieved_all_sids"]
        gold = set(r["answer_sids"])
        b_w3 = ofd.session_max(docs_w3[qid], sc_w3[qid])
        b_pers = ofd.session_max(docs_w3[qid], sc_pers[qid])
        o_w3 = ofd.order_from(b_w3, base)
        b_f4 = ofd.session_max(docs_full[qid], sc_f4[qid])
        b_fv = ofd.session_max(docs_full[qid], sc_fv[qid])
        s11 = ofd.two_stage(o_w3, [b_f4, b_fv])
        routed = is_temporal(r["question"])
        if routed:
            routed_ct += 1
            judged, raw = judge_order(r["question"], s11, docs_full[qid])
            raw_fh.write(json.dumps({"qid": qid, "raw": raw}) + "\n")
        else:
            judged = s11
        rows.append({
            "qid": qid, "qtype": r["qtype"], "routed": routed,
            "S11": ofd.hits(s11, gold), "judge": ofd.hits(judged, gold),
        })
    raw_fh.close()

    def r_at(sys_key: str, subset: list) -> list:
        n = len(subset)
        return [round(sum(x[sys_key][i] for x in subset) / n, 4) for i in range(3)] if n else [0, 0, 0]

    routed_rows = [x for x in rows if x["routed"]]
    f = sum(x["judge"][0] and not x["S11"][0] for x in routed_rows)
    b = sum(x["S11"][0] and not x["judge"][0] for x in routed_rows)
    summ = {
        "set": name, "n": len(rows), "routed": routed_ct, "judge_model": MODEL, "K": K,
        "note": "EXPLORATORY, non-deterministic hosted judge; LoCoMo arm only; NOT the pre-registered gate",
        "overall": {"S11": r_at("S11", rows), "judge": r_at("judge", rows)},
        "routed_only": {"S11": r_at("S11", routed_rows), "judge": r_at("judge", routed_rows),
                        "fixes": f, "breaks": b, "net": f - b, "mcnemar_p": round(ofd.mcnemar(f, b), 4)},
    }
    with open(OUT / "judge_locomo_rows.jsonl", "w") as fh:
        for x in rows:
            fh.write(json.dumps(x) + "\n")
    (OUT / "judge_locomo_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
