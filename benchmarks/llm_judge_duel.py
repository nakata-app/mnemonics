"""Second-stage "final duel" over a finished llm_judge_rerank run.

The listwise judge sees only a few short windows per session; on dev most of its misses had the gold session
at its second pick. This stage re-asks the same question over the judge's first `--duel-k` picks with much
more of each session (CE-ranked windows up to `--duel-chars`, shown in original order) and moves the winner
to the front. Everything else keeps the judge order. No question_type label and no gold are visible.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from sentence_transformers import CrossEncoder

from dev_llm_rerank import locomo_sessions
from llm_judge_rerank import JsonlSessions, chat, windows

PROMPT = """You are the retrieval judge of a personal memory assistant.
{today}Question: {question}

Below are {n} past conversation sessions, each with its date. They are all related to the question; exactly
which one holds the information needed to answer it? Use the dates for questions about time, order, the latest
state, or how long ago something happened.

{blocks}

Reply with only the session number."""

PROMPT2 = """You are the retrieval judge of a personal memory assistant.
{today}Question: {question}

Below are {n} past conversation sessions, each with its date, all retrieved as candidates. Exactly one holds the
information needed to answer the question. Use the dates for questions about time, order, the latest state, or how
long ago something happened.

{blocks}

For each session, in at most one short sentence, say what it contributes to answering the question. Then on a new
line output exactly: ANSWER: <session number>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-calls", type=Path, required=True)
    ap.add_argument("--perq", type=Path, required=True)
    ap.add_argument("--locomo", type=Path)
    ap.add_argument("--sessions", type=Path)
    ap.add_argument("--sessions-jsonl", type=Path)
    ap.add_argument("--qdates", type=Path)
    ap.add_argument("--ce-cache", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--model", default="deepseek-v4-pro:cloud")
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--strict", action="store_true", help="abort on a failed call instead of keeping base order")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--duel-k", type=int, default=2)
    ap.add_argument("--duel-chars", type=int, default=8000)
    ap.add_argument("--window-chars", type=int, default=600)
    ap.add_argument("--reason-prompt", action="store_true",
                    help="use a prompt that asks for one sentence of reasoning then 'ANSWER: <n>'")
    ap.add_argument("--dump-prompts", type=Path, help="write {qid, prompt} JSONL instead of calling the LLM")
    ap.add_argument("--replay-raw", type=Path, help="JSONL {qid, raw} from a deterministic generation run")
    args = ap.parse_args()

    perq = {r["qid"]: r for r in json.load(open(args.perq))}
    judge = [json.loads(l) for l in open(args.judge_calls) if l.strip()]
    per_q = (JsonlSessions(args.sessions_jsonl) if args.sessions_jsonl
             else json.load(open(args.sessions)) if args.sessions else None)
    loco = locomo_sessions(args.locomo) if args.locomo else {}
    qdates = json.load(open(args.qdates)) if args.qdates else {}
    cache = json.load(open(args.ce_cache)) if args.ce_cache and args.ce_cache.exists() else {}
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")

    args.out.mkdir(parents=True, exist_ok=True)
    done = set()
    if (args.out / "calls.jsonl").exists():
        done = {json.loads(l)["qid"] for l in open(args.out / "calls.jsonl") if l.strip()}
    log = (args.out / "calls.jsonl").open("a")
    dump = args.dump_prompts.open("w") if args.dump_prompts else None
    replay = ({json.loads(l)["qid"]: json.loads(l)["raw"] for l in open(args.replay_raw) if l.strip()}
              if args.replay_raw else None)

    for o in judge:
        qid = o["qid"]
        if qid in done:
            continue
        r = perq[qid]
        sess = {k: tuple(v) for k, v in per_q[qid].items()} if per_q is not None else loco
        m = re.search(r"\[[\d,\s]*\]", o["raw"] or "")
        n_picks = len(dict.fromkeys(int(x) for x in re.findall(r"\d+", m.group(0)))) if m else 0
        duel = o["order"][: min(args.duel_k, n_picks)]
        order, raw, dt, usage = o["order"], None, 0.0, {}
        if len(duel) >= 2:
            blocks = []
            for i, s in enumerate(duel):
                ch = windows(sess[s][1], args.window_chars)
                sc = cache.get(qid, {}).get(f"{s}|{args.window_chars}")
                if sc is None or len(sc) != len(ch):
                    sc = ce.predict([(r["question"], c) for c in ch], batch_size=64, show_progress_bar=False)
                keep, used = [], 0
                for j in sorted(range(len(ch)), key=lambda j: -sc[j]):
                    if used + len(ch[j]) > args.duel_chars and keep:
                        continue
                    keep.append(j); used += len(ch[j])
                blocks.append(f"[{i}] date: {sess[s][0]}\n" + "\n...\n".join(ch[j] for j in sorted(keep)))
            today = f"Today's date: {qdates[qid]}\n" if qid in qdates else ""
            tpl = PROMPT2 if args.reason_prompt else PROMPT
            prompt = tpl.format(today=today, question=r["question"], n=len(duel),
                                blocks="\n\n".join(blocks))
            if dump is not None:
                dump.write(json.dumps({"qid": qid, "prompt": prompt}) + "\n")
                continue
            if replay is not None:
                raw, dt, usage = replay.get(qid), 0.0, {}
            else:
                raw, dt, usage = chat(args.provider, args.model, prompt,
                                      args.max_tokens, args.timeout, args.reasoning)
            if raw is None and args.strict:
                raise SystemExit(f"STRICT_ABORT qid={qid} {usage}")
            raw_s = raw or ""
            m_ans = re.search(r"(?i)answer:\s*(\d+)", raw_s)
            if m_ans:
                idx = int(m_ans.group(1))
            else:
                # A bare index ("0", "[1]") is an answer; prose is not. Grabbing the last number out of
                # prose ("... August 11 ...") silently moves a session for no reason, so keep the judge
                # order unless the whole output is just an index.
                bare = re.fullmatch(r"\s*[\[(]?\s*(\d{1,3})\s*[\])]?\s*[.!]?\s*", raw_s)
                idx = int(bare.group(1)) if bare else None
            if idx is not None and idx < len(duel):
                w = duel[idx]
                order = [w] + [s for s in o["order"] if s != w]
        log.write(json.dumps({"qid": qid, "qtype": o["qtype"], "base": o["base"], "judge": o["order"][:30],
                              "order": order[:30], "gold": o["gold"], "raw": raw, "latency_s": round(dt, 2),
                              "usage": usage}) + "\n")
        log.flush()

    if dump is not None:
        dump.close()
        print(f"prompts written -> {args.dump_prompts}")
        return

    rows = [json.loads(l) for l in open(args.out / "calls.jsonl") if l.strip()]
    ks = (1, 5, 10)
    H = {k: [0, 0, 0] for k in ("base", "judge", "order")}
    BT = defaultdict(lambda: {"n": 0, "judge": [0, 0, 0], "duel": [0, 0, 0]})
    for o in rows:
        g = set(o["gold"])
        BT[o["qtype"]]["n"] += 1
        for i, k in enumerate(ks):
            for key in H:
                H[key][i] += any(s in g for s in o[key][:k])
            BT[o["qtype"]]["judge"][i] += any(s in g for s in o["judge"][:k])
            BT[o["qtype"]]["duel"][i] += any(s in g for s in o["order"][:k])
    n = len(rows)
    summary = {"n": n, "model": f"replay:{args.replay_raw.name}" if replay is not None else args.model,
               "duel_k": args.duel_k, "duel_chars": args.duel_chars, "reason_prompt": args.reason_prompt,
               "duels": sum(o["raw"] is not None for o in rows),
               "base": [round(x / n, 4) for x in H["base"]], "judge": [round(x / n, 4) for x in H["judge"]],
               "duel": [round(x / n, 4) for x in H["order"]],
               "by_type": {t: {"n": v["n"], "judge": [round(x / v["n"], 3) for x in v["judge"]],
                               "duel": [round(x / v["n"], 3) for x in v["duel"]]} for t, v in sorted(BT.items())}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
