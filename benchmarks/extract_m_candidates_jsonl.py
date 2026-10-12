"""Stream LongMemEval-M and write, one line per question, the texts of its candidate sessions (plus gold).

Like extract_m_candidates.py, but the output is JSONL ({"qid": ..., "sessions": {sid: [date, text]}}) so a
consumer can seek to one question at a time instead of loading every question's sessions at once. Session
ids are the run's anonymized ids ("s" + sha1(original)[:12]).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from extract_m_candidates import anon, iter_questions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--perq", type=Path, required=True)
    ap.add_argument("--sids-key", default="cands")
    ap.add_argument("--top-n", type=int, default=75)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    want = {r["qid"]: set(r[args.sids_key][: args.top_n]) | set(r["answer_sids"]) for r in json.load(open(args.perq))}
    n = unresolved = 0
    with open(args.out, "w") as f:
        for q in iter_questions(args.data):
            qid = q["question_id"]
            if qid not in want:
                continue
            keep = {}
            for sid, date, sess in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
                a = anon(sid)
                if a in want[qid]:
                    keep[a] = [date, "\n".join(f'[{m["role"]}] {m["content"]}' for m in sess)]
            unresolved += len(want[qid] - set(keep))
            f.write(json.dumps({"qid": qid, "sessions": keep}) + "\n")
            n += 1
    print(f"questions {n}/{len(want)}, unresolved {unresolved} -> {args.out}")


if __name__ == "__main__":
    main()
