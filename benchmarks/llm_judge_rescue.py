"""Rescue stage: let deep candidates (ranks 11..K) into a small final judge round.

Flat listwise judging over 30 candidates was less precise than over 10 on dev, so deep candidates are shown in
groups and the LLM is asked which of them, if any, hold the needed information. Up to `--max-promote` of its
picks are inserted right after the top-10; the result is written as a calls.jsonl whose `order` is the new
base order for llm_judge_rerank.py (run it with --top-n 10 + max-promote). No question_type, no gold visible.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sentence_transformers import CrossEncoder

from dev_llm_rerank import locomo_sessions
from llm_judge_rerank import JsonlSessions, chat, windows

PROMPT = """You are the retrieval judge of a personal memory assistant.
{today}Question: {question}

Below are {n} past conversation sessions (most relevant excerpts only), each with its date. Most of them are
probably unrelated. Which of them, if any, contain information needed to answer the question?

{blocks}

Reply with only a JSON list of session numbers, most relevant first, or [] if none of them do."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perq", type=Path, required=True)
    ap.add_argument("--base-calls", type=Path, help="calls.jsonl whose `order` is the base order")
    ap.add_argument("--sids-key", default="cands")
    ap.add_argument("--locomo", type=Path)
    ap.add_argument("--sessions-jsonl", type=Path)
    ap.add_argument("--qdates", type=Path)
    ap.add_argument("--ce-cache", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--model", default="deepseek-v4-pro:cloud")
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--keep", type=int, default=10, help="top ranks kept as is")
    ap.add_argument("--depth", type=int, default=75, help="deepest rank considered")
    ap.add_argument("--group", type=int, default=22)
    ap.add_argument("--max-promote", type=int, default=3)
    ap.add_argument("--windows-per-session", type=int, default=3)
    ap.add_argument("--window-chars", type=int, default=600)
    args = ap.parse_args()

    perq = {r["qid"]: r for r in json.load(open(args.perq))}
    if args.base_calls:
        base = {}
        for l in open(args.base_calls):
            if l.strip():
                o = json.loads(l)
                base[o["qid"]] = o["order"]
    else:
        base = {q: r[args.sids_key] for q, r in perq.items()}
    per_q = JsonlSessions(args.sessions_jsonl) if args.sessions_jsonl else None
    loco = locomo_sessions(args.locomo) if args.locomo else {}
    qdates = json.load(open(args.qdates)) if args.qdates else {}
    cache = json.load(open(args.ce_cache)) if args.ce_cache and args.ce_cache.exists() else {}
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")

    args.out.mkdir(parents=True, exist_ok=True)
    done = set()
    if (args.out / "calls.jsonl").exists():
        done = {json.loads(l)["qid"] for l in open(args.out / "calls.jsonl") if l.strip()}
    log = (args.out / "calls.jsonl").open("a")

    for qid in [q for q in perq if q in base]:
        if qid in done:
            continue
        r = perq[qid]
        sess = per_q[qid] if per_q is not None else loco
        order = base[qid]
        top, deep = order[: args.keep], [s for s in order[args.keep: args.depth] if s in sess]
        today = f"Today's date: {qdates[qid]}\n" if qid in qdates else ""
        picks, raws, lat = [], [], 0.0
        for g0 in range(0, len(deep), args.group):
            grp = deep[g0: g0 + args.group]
            blocks = []
            for i, s in enumerate(grp):
                ch = windows(sess[s][1], args.window_chars)
                sc = cache.get(qid, {}).get(f"{s}|{args.window_chars}")
                if sc is None or len(sc) != len(ch):
                    sc = ce.predict([(r["question"], c) for c in ch], batch_size=64, show_progress_bar=False)
                keep = sorted(sorted(range(len(ch)), key=lambda j: -sc[j])[: args.windows_per_session])
                blocks.append(f"[{i}] date: {sess[s][0]}\n" + "\n...\n".join(ch[j] for j in keep))
            raw, dt, _ = chat(args.provider, args.model,
                              PROMPT.format(today=today, question=r["question"], n=len(grp), blocks="\n\n".join(blocks)),
                              args.max_tokens, args.timeout, args.reasoning)
            raws.append(raw); lat += dt
            m = re.search(r"\[[\d,\s]*\]", raw or "")
            if m:
                picks += [grp[int(x)] for x in dict.fromkeys(re.findall(r"\d+", m.group(0))) if int(x) < len(grp)]
        promoted = picks[: args.max_promote]
        new = top + promoted + [s for s in order[args.keep:] if s not in promoted]
        log.write(json.dumps({"qid": qid, "qtype": r["qtype"], "base": order[:75], "order": new[:75],
                              "promoted": promoted, "gold": sorted(r["answer_sids"]), "raw": raws,
                              "latency_s": round(lat, 2)}) + "\n")
        log.flush()

    rows = [json.loads(l) for l in open(args.out / "calls.jsonl") if l.strip()]
    n = len(rows)
    ks = (1, 5, 10, 13)
    hit = lambda o, key, k: any(s in set(o["gold"]) for s in o[key][:k])
    summary = {"n": n, "model": args.model, "keep": args.keep, "depth": args.depth, "max_promote": args.max_promote,
               "promoted_any": sum(bool(o["promoted"]) for o in rows),
               "gold_rescued": sum(any(s in set(o["gold"]) for s in o["promoted"]) and not hit(o, "base", args.keep)
                                   for o in rows),
               "base": {f"R@{k}": round(sum(hit(o, "base", k) for o in rows) / n, 4) for k in ks},
               "rescued": {f"R@{k}": round(sum(hit(o, "order", k) for o in rows) / n, 4) for k in ks}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
