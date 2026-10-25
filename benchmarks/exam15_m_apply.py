"""EXAM 15 M step: apply the dev-selected date-consistency config to the persisted EXAM 11 M order.

Pre-registered in docs/EXAM15_PREREG.md. The dev gate (exam15_datelane_dev.py dev) PASSED and selected a
config (read here from dev_summary.json, never re-selected on M). This script:
  step 0  reproduces EXAM 11 R@1/5/10 = 0.894/0.966/0.984 from results/exam11_m_scored/calls.jsonl `order`
          (the persisted S11 M order) vs `gold`, as an instrument check BEFORE the lever touches anything;
  apply   re-orders each question's top-K with date-consistent sessions first, using the FROZEN functions
          imported verbatim from exam15_datelane_dev (target / event_dates / partition / TOL). No GPU, no cloud.

The only M-specific code is date normalization: m_qdates and m_deep_top75_sessions store dates as
"YYYY/MM/DD (Day) HH:MM"; the frozen functions expect ISO. Normalization is pure data prep and does not touch
the router grammar, extractor, grid, TOL or gate. The 490 MB session file is streamed, one qid per line.

Determinism: offline, byte-identical re-run. Forbidden (prereg): re-selecting a config on M, tuning anything,
reading M per-question results before this run. The full 6-config grid is printed for transparency but the
reported result is the frozen dev selection only.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import exam15_datelane_dev as ex15  # frozen functions: target, event_dates, partition, TOL, GRID
import order_fusion_dev as ofd

R, I = ofd.R, ofd.I
D = Path.home() / ".local/share/mnemonics-exam15"
CALLS = R / "exam11_m_scored" / "calls.jsonl"
PERQ = I / "m_perq.json"
QDATES = I / "m_qdates.json"
SESSIONS = I / "m_deep_top75_sessions.jsonl"
EXAM11_REF = (0.894, 0.966, 0.984)  # prereg instrument check


def norm_dt(s: str) -> str:
    """'2023/05/30 (Tue) 23:31' -> ISO 'YYYY-MM-DDTHH:MM' for datetime.fromisoformat (used by target())."""
    return datetime.strptime(s.strip(), "%Y/%m/%d (%a) %H:%M").isoformat()


def norm_date(s: str) -> date:
    """Leading 'YYYY/MM/DD' of a session timestamp -> date."""
    return datetime.strptime(s.strip()[:10], "%Y/%m/%d").date()


def rk(hitvec: list) -> tuple:
    n = len(hitvec)
    return tuple(round(sum(h[i] for h in hitvec) / n, 4) for i in range(3))


def hits(order: list, gold: set) -> list:
    return [1 if any(s in gold for s in order[:k]) else 0 for k in (1, 5, 10)]


def main() -> None:
    sel = json.load(open(D / "dev_summary.json"))["gate"]["selected"]  # frozen dev pick, e.g. "K10_event"
    sel_k = int(sel.split("_")[0][1:])
    sel_arm = sel.split("_")[1]
    print(f"dev-selected config = {sel} (K={sel_k}, arm={sel_arm}); grid printed for transparency, not re-selected")

    calls = {r["qid"]: r for r in (json.loads(l) for l in open(CALLS) if l.strip())}
    perq = {r["qid"]: r for r in json.load(open(PERQ))}
    qdates = json.load(open(QDATES))

    # step 0: reproduce EXAM 11 from the persisted order before the lever
    s11_hits = {qid: hits(c["order"], set(c["gold"])) for qid, c in calls.items()}
    s11_rk = rk(list(s11_hits.values()))
    ok0 = all(abs(s11_rk[i] - EXAM11_REF[i]) <= 0.001 for i in range(3))
    print(f"step 0  S11 (persisted EXAM 11 order) R@1/5/10 = {s11_rk}  ref {EXAM11_REF}  reproduce={'OK' if ok0 else 'MISMATCH'}")
    if not ok0:
        raise SystemExit("step 0 failed: persisted order does not reproduce EXAM 11; aborting before the lever")

    # apply: stream the 490 MB session file, process each qid once
    grid_hits = {f"K{k}_{ds}": {} for k, ds in ex15.GRID}
    routed = {key: 0 for key in grid_hits}
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
            sess = {sid: (v[0], v[1]) for sid, v in row["sessions"].items()}
            tg = ex15.target(perq[qid]["question"], norm_dt(qdates[qid])) if qid in qdates else None
            for k, ds in ex15.GRID:
                key = f"K{k}_{ds}"
                if tg is None:
                    grid_hits[key][qid] = s11_hits[qid]
                    continue
                routed[key] += 1
                t, w = tg
                lo, hi = t - timedelta(days=w + ex15.TOL), t + timedelta(days=w + ex15.TOL)
                cons = set()
                for sid in order[:k]:
                    if sid not in sess:
                        continue
                    sdate = norm_date(sess[sid][0])
                    dts = {sdate} | (ex15.event_dates(sess[sid][1], sdate) if ds == "event" else set())
                    if any(lo <= d <= hi for d in dts):
                        cons.add(sid)
                grid_hits[key][qid] = hits(ex15.partition(order, k, cons), gold)

    # any qid without a session line keeps S11 (should be none; report if so)
    missing = [qid for qid in calls if qid not in grid_hits[f"K{sel_k}_{sel_arm}"]]
    for key in grid_hits:
        for qid in missing:
            grid_hits[key][qid] = s11_hits[qid]
    if missing:
        print(f"WARNING: {len(missing)} qids had no session line; kept S11 for them")

    # per-type and paired vs S11 for the SELECTED config only (the reported result)
    qtypes = {qid: calls[qid]["qtype"] for qid in calls}
    def paired(key):
        f = b = 0
        for qid in calls:
            s, c = s11_hits[qid][0], grid_hits[key][qid][0]
            f += c and not s
            b += s and not c
        return f, b, f - b

    print("\n grid (all 6, R@1/5/10 | routed):")
    for k, ds in ex15.GRID:
        key = f"K{k}_{ds}"
        mark = "  <== dev selection" if key == sel else ""
        print(f"  {key:12s} {rk(list(grid_hits[key].values()))} routed={routed[key]}{mark}")

    key = sel
    sel_rk = rk(list(grid_hits[key].values()))
    f, b, net = paired(key)
    by_type = {}
    for t in sorted(set(qtypes.values())):
        qs = [qid for qid in calls if qtypes[qid] == t]
        by_type[t] = {"n": len(qs), "S11": round(sum(s11_hits[q][0] for q in qs) / len(qs), 3),
                      sel: round(sum(grid_hits[key][q][0] for q in qs) / len(qs), 3)}

    result = {"exam": "EXAM 15 M apply", "selected": sel, "n": len(calls),
              "step0_s11_r1_r5_r10": s11_rk, "step0_reproduce_exam11": ok0,
              "selected_r1_r5_r10": sel_rk, "routed": routed[key],
              "paired_vs_S11_r1": {"fixes": f, "breaks": b, "net": net},
              "by_type_r1": by_type}
    (D / "m_apply_result.json").write_text(json.dumps(result, indent=2))
    print("\n RESULT:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
