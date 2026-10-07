"""Convert LoCoMo into LongMemEval question format for dev-only session retrieval.

Dev data for the original-score plan (see ORIGINAL_SCORE_PLAN.md): LongMemEval S/M is the exam
and must never be used for development, so generic mechanisms are tuned here instead.

Each LoCoMo QA (category 5 / unanswerable skipped) becomes one question whose haystack is every
session of its own conversation plus, with --distractors, every session of the other nine
conversations. Gold = sessions named in the QA evidence ("D3:5" -> session 3).
All turns are role "user" with a "Name: " prefix, because both speakers are people.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta
from pathlib import Path


def lme_date(s: str) -> str:
    dt = datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y")
    return dt.strftime("%Y/%m/%d (%a) %H:%M")


def sessions_of(sample: dict) -> list[tuple[str, int, str, list[dict]]]:
    conv = sample["conversation"]
    out = []
    for k in conv:
        m = re.fullmatch(r"session_(\d+)", k)
        if not m or not conv[k]:
            continue
        n = int(m.group(1))
        turns = [{"role": "user", "content": f'{t["speaker"]}: {t["text"]}'} for t in conv[k]]
        out.append((f'{sample["sample_id"]}_s{n}', n, lme_date(conv[f"session_{n}_date_time"]), turns))
    return sorted(out, key=lambda x: x[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", type=Path, default=Path(__file__).parent / "data/locomo/locomo10.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--distractors", action="store_true", help="add other conversations' sessions to every haystack")
    args = ap.parse_args()

    data = json.load(open(args.locomo))
    per_conv = {s["sample_id"]: sessions_of(s) for s in data}
    questions = []
    for s in data:
        own = per_conv[s["sample_id"]]
        hay = [x for sid in per_conv for x in per_conv[sid]] if args.distractors else own
        last = max(datetime.strptime(d, "%Y/%m/%d (%a) %H:%M") for _, _, d, _ in own)
        qdate = (last + timedelta(days=1)).strftime("%Y/%m/%d (%a) %H:%M")
        for i, qa in enumerate(s["qa"]):
            if qa.get("category") == 5:
                continue
            nums = {int(m.group(1)) for e in qa.get("evidence", []) for m in [re.match(r"D(\d+):", e)] if m}
            gold = [sid for sid, n, _, _ in own if n in nums]
            if not gold:
                continue
            questions.append({
                "question_id": f'{s["sample_id"]}_q{i}',
                "question_type": f'locomo-cat{qa["category"]}',
                "question": qa["question"],
                "answer": str(qa.get("answer", "")),
                "question_date": qdate,
                "haystack_session_ids": [x[0] for x in hay],
                "haystack_dates": [x[2] for x in hay],
                "haystack_sessions": [x[3] for x in hay],
                "answer_session_ids": gold,
            })
    args.out.write_text(json.dumps(questions))
    print(f"wrote {len(questions)} questions, haystack {len(questions[0]['haystack_session_ids'])} sessions -> {args.out}")


if __name__ == "__main__":
    main()
