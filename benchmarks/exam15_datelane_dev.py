"""EXAM 15 dev instrument: date-consistency partition over the S11 order (frozen before any temp1 score).

Lever (deterministic, model-free, offline): for a question whose text yields a concrete target time
(mnemonics.memory_route.route_memory_query with reference_time = question date -> target_time, window_days),
take the S11 order, and within its top-K stably move the DATE-CONSISTENT sessions ahead of the inconsistent
ones. A session is consistent when any date in its date set lies in [target - window - TOL, target + window + TOL],
TOL = 1 day. The date set is, by grid arm:
  DS=session   {session date}                               (negative control: timestamp-only routing, which the
                                                             2026-10-07 M arithmetic already falsified 10/20)
  DS=event     {session date} U event dates resolved from the user's turns by the rule-based extractor below
Questions that do not route to a target time are left unchanged (lever inert by construction).

Grid (6 configs, fixed): K in {5, 10, 20} x DS in {session, event}. Selection = best temp1-anchor R@1, ties ->
smaller K, then DS=session. Gate (all must hold for the selected config):
  G1 temp1-anchor R@1 >= S11 + 0.05 and McNemar net > 0 and p < 0.10
  G2 temp1-control (duration questions) R@1/R@5/R@10 each >= S11 - 0.005  (inertness)
  G3 LoCoMo 200 and pref2 80: no R@1/R@5/R@10 below S11 - 0.005
No gold is read while routing or extracting; gold is read only to score. The extractor and the grammar are the
ones in this file at the committed SHA; they are not edited after any temp1 score exists.

Usage:
  python3 -I benchmarks/exam15_datelane_dev.py selftest          # extractor unit checks on toy strings (no data)
  python3 -I benchmarks/exam15_datelane_dev.py dev               # scores all sets, prints gate -> dev_summary.json
"""
from __future__ import annotations

import calendar
import json
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import order_fusion_dev as ofd  # noqa: E402
from mnemonics.memory_route import route_memory_query  # noqa: E402

D = Path.home() / ".local/share/mnemonics-exam15"
R, I = ofd.R, ofd.I
TOL = 1
GRID = [(k, ds) for k in (5, 10, 20) for ds in ("session", "event")]

SETS = {
    # LoCoMo has no question date (ledger 2026-10-23), so no target time can be computed: the lever is inert there
    # by construction and the set is scored only to show that (G3).
    "locomo": {**ofd.SETS["locomo"], "qdates": None, "sessions": None},
    "pref2": {**ofd.SETS["pref2"], "qdates": I / "pref2_qdates.json", "sessions": I / "pref2_sessions.jsonl"},
    "temp1": {"perq": I / "temp1_perq.json", "docs_w3": I / "temp1_rerank_docs_top40_w3.jsonl",
              "docs_full": I / "temp1_rerank_docs_top40_full3000.jsonl", "w3": R / "mnemonics-temp1-qrr4b/scores.jsonl",
              "full4b": R / "mnemonics-temp1-qrr4b-full/scores.jsonl", "fullvl": R / "mnemonics-temp1-qvlrr8b-full/scores.jsonl",
              "qdates": D / "temp1_qdates.json", "sessions": I / "temp1_sessions.jsonl"},
}

# ---- rule-based event-time extractor (user turns only) ------------------------------------------------------
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "couple": 2, "few": 3}
_UNIT = {"day": 1, "week": 7, "month": 30, "year": 365}
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
_RX_AGO = re.compile(r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|couple|few)(?:\s+of)?\s+(day|week|month|year)s?\s+ago\b", re.I)
_RX_TODAY = re.compile(r"\b(today|tonight|this (?:morning|afternoon|evening)|earlier today)\b", re.I)
_RX_YESTERDAY = re.compile(r"\byesterday\b", re.I)
_RX_DAY_BEFORE = re.compile(r"\bthe day before yesterday\b", re.I)
_RX_LAST_WD = re.compile(r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I)
_RX_LAST_UNIT = re.compile(r"\blast\s+(week|month|year|weekend)\b", re.I)
_RX_ABS = re.compile(r"\b(?:on\s+)?(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.I)
_RX_ISO = re.compile(r"\b(20\d\d)-(\d\d)-(\d\d)\b")


def _last_weekday(ref: date, name: str) -> date:
    t = list(calendar.day_name).index(name.capitalize())
    back = (ref.weekday() - t) % 7 or 7
    return ref - timedelta(days=back)


def event_dates(text: str, session_date: date) -> set:
    """Dates of events the USER reports, resolved against the session date. Rule-based, deterministic."""
    out = set()
    for line in text.split("\n"):
        if not line.startswith("[user]"):
            continue
        for m in _RX_AGO.finditer(line):
            n = int(m.group(1)) if m.group(1).isdigit() else _NUM[m.group(1).lower()]
            out.add(session_date - timedelta(days=n * _UNIT[m.group(2).lower()]))
        if _RX_TODAY.search(line):
            out.add(session_date)
        if _RX_DAY_BEFORE.search(line):
            out.add(session_date - timedelta(days=2))
        elif _RX_YESTERDAY.search(line):
            out.add(session_date - timedelta(days=1))
        for m in _RX_LAST_WD.finditer(line):
            out.add(_last_weekday(session_date, m.group(1)))
        for m in _RX_LAST_UNIT.finditer(line):
            u = m.group(1).lower()
            out.add(session_date - timedelta(days={"week": 7, "weekend": 7, "month": 30, "year": 365}[u]))
        for m in _RX_ABS.finditer(line):
            try:
                d = date(session_date.year, _MONTHS[m.group(1).lower()], int(m.group(2)))
                out.add(d if d <= session_date else d.replace(year=session_date.year - 1))
            except ValueError:
                pass
        for m in _RX_ISO.finditer(line):
            try:
                out.add(date(int(m.group(1)), int(m.group(2)), int(m.group(3))))
            except ValueError:
                pass
    return out


def selftest() -> None:
    s = date(2024, 5, 15)  # Wednesday
    cases = [
        ("[user] Yesterday I finally saw the dentist.", {date(2024, 5, 14)}),
        ("[user] Two weeks ago I ran my first 10K.\n[assistant] Great, three weeks ago you were injured.", {date(2024, 5, 1)}),
        ("[user] Last Friday we tried the new ramen place.", {date(2024, 5, 10)}),
        ("[user] Today I picked up the parcel.", {s}),
        ("[user] A month ago I started lessons.", {date(2024, 4, 15)}),
        ("[user] On March 3rd I had the interview.", {date(2024, 3, 3)}),
        ("[user] Nothing happened.", set()),
    ]
    bad = 0
    for text, want in cases:
        got = event_dates(text, s)
        bad += got != want
        print("OK " if got == want else "BAD", repr(text[:45]), sorted(map(str, got)))
    print("selftest failures", bad)


# ---- scoring -------------------------------------------------------------------------------------------------
def target(question: str, qdate: str):
    rt = route_memory_query(question, reference_time=datetime.fromisoformat(qdate))
    if rt.target_time is None:
        return None
    return rt.target_time.date(), (rt.window_days or 0)


def partition(order: list, k: int, consistent: set) -> list:
    head = order[:k]
    return [s for s in head if s in consistent] + [s for s in head if s not in consistent] + order[k:]


def load_sessions(c: dict) -> dict:
    """{qid: {sid: (date, text)}} from a per-question sessions JSONL; empty when the set has none."""
    if c["sessions"] is None:
        return {}
    out = {}
    for line in open(c["sessions"]):
        if line.strip():
            r = json.loads(line)
            out[r["qid"]] = {sid: (v[0], v[1]) for sid, v in r["sessions"].items()}
    return out


def run_set(name: str) -> list:
    c = SETS[name]
    perq = {r["qid"]: r for r in json.load(open(c["perq"]))}
    qdates = json.load(open(c["qdates"])) if c["qdates"] else {}
    docs_w3, docs_full = ofd.jl(c["docs_w3"], "docs"), ofd.jl(c["docs_full"], "docs")
    sc_w3, sc_f4, sc_fv = ofd.jl(c["w3"], "scores"), ofd.jl(c["full4b"], "scores"), ofd.jl(c["fullvl"], "scores")
    sess_all = load_sessions(c) if qdates else {}
    rows = []
    for qid in sorted(sc_w3):
        r = perq[qid]
        gold = set(r["answer_sids"])
        o_w3 = ofd.order_from(ofd.session_max(docs_w3[qid], sc_w3[qid]), r["retrieved_all_sids"])
        s11 = ofd.two_stage(o_w3, [ofd.session_max(docs_full[qid], sc_f4[qid]), ofd.session_max(docs_full[qid], sc_fv[qid])])
        row = {"qid": qid, "qtype": r["qtype"], "S11": ofd.hits(s11, gold), "routed": False}
        tg = target(r["question"], qdates[qid]) if qid in qdates else None
        sess = sess_all.get(qid, {}) if tg else {}
        for k, ds in GRID:
            key = f"K{k}_{ds}"
            if tg is None:
                row[key] = row["S11"]
                continue
            row["routed"] = True
            t, w = tg
            lo, hi = t - timedelta(days=w + TOL), t + timedelta(days=w + TOL)
            cons = set()
            for sid in s11[:k]:
                if sid not in sess:
                    continue
                sdate = date.fromisoformat(sess[sid][0][:10])
                dates = {sdate} | (event_dates(sess[sid][1], sdate) if ds == "event" else set())
                if any(lo <= d <= hi for d in dates):
                    cons.add(sid)
            row[key] = ofd.hits(partition(s11, k, cons), gold)
        rows.append(row)
    return rows


def dev() -> None:
    systems = ["S11"] + [f"K{k}_{ds}" for k, ds in GRID]
    out = {}
    for name in ("temp1", "locomo", "pref2"):
        rows = run_set(name)
        if name == "temp1":
            meta = {m["qid"]: m for m in json.load(open(D / "temp1_meta.json"))}
            anchor = [r for r in rows if not meta[r["qid"]]["control"]]
            control = [r for r in rows if meta[r["qid"]]["control"]]
            out["temp1_anchor"] = ofd.summarize("temp1_anchor", anchor, systems, "S11")
            out["temp1_control"] = ofd.summarize("temp1_control", control, systems, "S11")
            out["temp1_anchor"]["routed"] = sum(r["routed"] for r in anchor)
            out["temp1_control"]["routed"] = sum(r["routed"] for r in control)
            by_kind = defaultdict(list)
            for r in anchor:
                by_kind[meta[r["qid"]]["kind"]].append(r)
            out["temp1_anchor"]["by_kind_r1"] = {k: {s: round(sum(x[s][0] for x in v) / len(v), 3) for s in systems}
                                                 for k, v in sorted(by_kind.items())}
        else:
            out[name] = ofd.summarize(name, rows, systems, "S11")
            out[name]["routed"] = sum(r["routed"] for r in rows)
        (D / f"rows_{name}.json").write_text(json.dumps(rows))
    A = out["temp1_anchor"]
    best = max(GRID, key=lambda g: (A["systems"][f"K{g[0]}_{g[1]}"][0], -g[0], g[1] == "session"))
    sel = f"K{best[0]}_{best[1]}"
    pa = A["paired_vs_S11"][sel]
    g1 = A["systems"][sel][0] >= A["systems"]["S11"][0] + 0.05 and pa["net"] > 0 and pa["mcnemar_p"] < 0.10
    C = out["temp1_control"]
    g2 = all(C["systems"][sel][i] >= C["systems"]["S11"][i] - 0.005 for i in range(3))
    g3 = all(out[n]["systems"][sel][i] >= out[n]["systems"]["S11"][i] - 0.005 for n in ("locomo", "pref2") for i in range(3))
    out["gate"] = {"selected": sel, "G1": g1, "G2": g2, "G3": g3, "verdict": "PASS" if (g1 and g2 and g3) else "FAIL"}
    (D / "dev_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev"
    selftest() if cmd == "selftest" else dev()
