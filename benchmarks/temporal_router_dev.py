"""EXAM 18 dev instrument, part 1 of 2: the temporal query router.

The EXAM 18 hypothesis is a temporal-routed deterministic judge over the S11 / EXAM 11 order: a 14B judge
re-reads the S11 top-k ONLY for queries the router marks temporal, and the router is a pure query-regex
classifier FIXED BEFORE ANY JUDGE OUTPUT EXISTS. This script is that router, committed on its own so the
routing decision cannot be tuned to the judge's answers later. It is encoder-free, offline and deterministic:
it reads only the perq question text and the qtype label.

Discipline (reward-hacking guard): the regex is designed from temporal LANGUAGE cues and the published
LongMemEval category definitions, never fitted to answer_sids. The only thing measured here is the router's
FIRE RATE PER qtype -- an input-vs-input face-validity check (does the 'temporal' regex fire on the questions
labelled temporal?), not an accuracy against gold. No gold is read.

Routed-temporal means: the question asks to recall WHEN something happened or reason over a past timepoint
or duration. It deliberately does NOT fire on bare future intent ('next week/month') that appears in pref2
preference queries, because that is planning, not memory-temporal recall.

Usage: python3 -I benchmarks/temporal_router_dev.py            # both dev sets
        python3 -I benchmarks/temporal_router_dev.py --sets locomo
Outputs ~/.local/share/mnemonics-exam18/router_{set}.jsonl + router_summary.json.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

I = Path("/Volumes/WD-Backup/datasets/longmemeval/rerank_inputs")
PERQ = {"locomo": I / "locomo_poolall_perq.json", "pref2": I / "pref2_perq.json"}

# The frozen router. Each alternative is a temporal-recall cue. Ordered groups kept for readability; the
# compiled pattern is the single source of truth and is what the committed sha256 pins.
_MONTHS = r"january|february|march|april|may|june|july|august|september|october|november|december"
_WEEKDAYS = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
TEMPORAL = re.compile(
    r"\b(?:"
    r"when|what\s+time|what\s+date|what\s+year|which\s+year|what\s+day|which\s+day"
    r"|how\s+long|how\s+often|how\s+many\s+(?:times|years|months|weeks|days|hours)"
    r"|before|after|since|until|till|during|prior\s+to|by\s+the\s+time|earlier|later\s+than"
    r"|first\s+time|last\s+time|most\s+recent|recently|ago|used\s+to|back\s+when"
    r"|" + _MONTHS +
    r"|" + _WEEKDAYS +
    r"|\b(?:19|20)\d{2}\b"            # a 4-digit year 1900-2099
    r"|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b"  # numeric date 5/1 or 5/1/2022
    r")\b",
    re.IGNORECASE,
)


def is_temporal(q: str) -> bool:
    return bool(TEMPORAL.search(q or ""))


def run_set(name: str, out_dir: Path) -> dict:
    recs = json.load(open(PERQ[name]))
    per_type: dict = defaultdict(lambda: {"n": 0, "temporal": 0})
    rows = []
    for r in recs:
        q = r.get("question", "")
        t = is_temporal(q)
        qt = r["qtype"]
        per_type[qt]["n"] += 1
        per_type[qt]["temporal"] += int(t)
        rows.append({"qid": r["qid"], "qtype": qt, "temporal": t, "question": q})
    n = len(rows)
    routed = sum(x["temporal"] for x in rows)
    summ = {
        "set": name,
        "n": n,
        "routed_temporal": routed,
        "routed_frac": round(routed / n, 4) if n else 0.0,
        "by_qtype": {
            qt: {"n": v["n"], "temporal": v["temporal"], "frac": round(v["temporal"] / v["n"], 4)}
            for qt, v in sorted(per_type.items())
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"router_{name}.jsonl", "w") as fh:
        for x in rows:
            fh.write(json.dumps(x) + "\n")
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["locomo", "pref2"])
    ap.add_argument("--out", type=Path, default=Path.home() / ".local/share/mnemonics-exam18")
    args = ap.parse_args()
    out = {}
    for s in args.sets:
        out[s] = run_set(s, args.out)
    (args.out).mkdir(parents=True, exist_ok=True)
    (args.out / "router_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
