"""Stream LongMemEval-M and keep only the sessions a finished run ranked in its top-N (plus gold), keyed by
question id and the run's anonymized session id ("s" + sha1(original)[:12]).

The M file is 2.7GB. Loading it whole would crash the 8GB Mac, so this reads one question object at a time
with a bounded buffer. Output feeds dev_llm_rerank.py --sessions for the one-shot M exam replay.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path


def anon(sid: str) -> str:
    return "s" + hashlib.sha1(sid.encode()).hexdigest()[:12]


def iter_questions(path: Path, chunk: int = 1 << 22):
    dec = json.JSONDecoder()
    buf = ""
    with open(path, encoding="utf-8") as f:
        while True:  # skip to the array start
            c = f.read(1)
            if c == "[":
                break
        while True:
            buf = buf.lstrip().lstrip(",").lstrip()
            if buf.startswith("]"):
                return
            try:
                obj, end = dec.raw_decode(buf)
            except json.JSONDecodeError:
                more = f.read(chunk)
                if not more:
                    return
                buf += more
                continue
            yield obj
            buf = buf[end:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--perq-glob", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--top-n", type=int, default=5)
    args = ap.parse_args()

    want: dict[str, set[str]] = {}
    for f in sorted(glob.glob(args.perq_glob)):
        for r in json.load(open(f)):
            want[r["qid"]] = set(r["retrieved_top10_sids"][: args.top_n]) | set(r["answer_sids"])

    out: dict[str, dict[str, list[str]]] = {}
    for q in iter_questions(args.data):
        qid = q["question_id"]
        if qid not in want:
            continue
        keep = {}
        for sid, date, sess in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
            a = anon(sid)
            if a in want[qid]:
                keep[a] = [date, "\n".join(f'[{m["role"]}] {m["content"]}' for m in sess)]
        out[qid] = keep
    args.out.write_text(json.dumps(out))
    missing = sum(len(want[q] - set(out.get(q, {}))) for q in want)
    print(f"questions {len(out)}/{len(want)}, sessions {sum(len(v) for v in out.values())}, unresolved {missing}")


if __name__ == "__main__":
    main()
