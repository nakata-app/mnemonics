"""Final-step LLM rerank over a finished run's top-N sessions (dev data only, see ORIGINAL_SCORE_PLAN.md).

Takes the per-question output of a Colab retrieval run (perq.json with retrieved_top10_sids) and asks an
OpenAI-compatible LLM to pick, among the top-N sessions, the one that answers the question. Because it is
the last pipeline step, replaying it on the saved top-N gives the same answer a live run would.

Runs from the Mac: only light HTTP calls, no models are loaded locally.
Security: the API key is read from the keystore inside this process. It is never put in the environment,
never passed to a child process, and never printed or logged.

Usage:
  python3 benchmarks/dev_llm_rerank.py --perq <perq.json> --locomo benchmarks/data/locomo/locomo10.json \
      --out <dir> --sample 20 --top-n 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.request
from pathlib import Path

KEYSTORE = Path.home() / ".svelka/keys.json"
BASE_URL = "https://evren-llmapi.ssyz.org.tr/v1"

PROMPT = """You are a retrieval judge for a personal memory system.
Pick the single conversation session that contains the information needed to answer the question.
Pay attention to dates and to event order words such as first, after, before, latest, still.

Question: {question}

Sessions:
{blocks}

Reply with only the session number (0 to {last})."""


def load_key() -> str:
    return json.loads(KEYSTORE.read_text())["providers"]["evren"][0]


def locomo_sessions(path: Path) -> dict[str, tuple[str, str]]:
    out = {}
    for conv in json.load(open(path)):
        c = conv["conversation"]
        for k in c:
            m = re.fullmatch(r"session_(\d+)", k)
            if m and c[k]:
                text = "\n".join(f'{t["speaker"]}: {t["text"]}' for t in c[k])
                sid = f'{conv["sample_id"]}_s{m.group(1)}'
                val = (c[f"session_{m.group(1)}_date_time"], text)
                out[sid] = val
                # clean2 runs anonymize session ids as "s" + sha1(id)[:12]; index both forms
                out["s" + hashlib.sha1(sid.encode()).hexdigest()[:12]] = val
    return out


_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the and or of to in on at for with is are was were be been i you my me we our it this that what which who when where how did do does have has had".split())


def snippet(question: str, texts: list[str], budget: int) -> list[str]:
    """Query-relevant excerpt per session: score each turn by idf-weighted overlap with the question
    (idf over all turns of the candidates), keep the best turns up to `budget` chars, in original order."""
    import math
    q = {w for w in _WORD.findall(question.lower()) if w not in _STOP}
    turns = [[t for t in text.split("\n") if t.strip()] for text in texts]
    docs = [set(_WORD.findall(t.lower())) for ts in turns for t in ts]
    n = max(len(docs), 1)
    idf = {w: math.log(1 + n / (1 + sum(w in d for d in docs))) for w in q}
    out = []
    for ts in turns:
        scored = sorted(range(len(ts)), key=lambda i: -sum(idf[w] for w in q if w in set(_WORD.findall(ts[i].lower()))))
        keep, used = set(), 0
        for i in scored:
            if used + len(ts[i]) > budget and keep:
                continue
            keep.add(i); used += min(len(ts[i]), budget)
            if used >= budget:
                break
        out.append("\n".join((ts[i] if len(ts[i]) <= budget else ts[i][:budget]) for i in sorted(keep)))
    return out


def chat(key: str, model: str, prompt: str, max_tokens: int, timeout: float) -> tuple[str | None, float, dict]:
    # glm-5.3 is a reasoning model: it thinks in a separate field before writing content. A small
    # max_tokens is spent on reasoning and content comes back empty (finish_reason=length).
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        usage = dict(d.get("usage", {}), finish_reason=d["choices"][0].get("finish_reason"))
        return (d["choices"][0]["message"]["content"] or "").strip(), time.time() - t0, usage
    except Exception as e:  # report the error class only; never echo request headers
        return None, time.time() - t0, {"error": f"{type(e).__name__}: {getattr(e, 'code', '')}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perq", type=Path, required=True)
    ap.add_argument("--locomo", type=Path, help="LoCoMo json (dev)")
    ap.add_argument("--sessions", type=Path, help="per-question {qid: {sid: [date, text]}} from extract_m_candidates.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--max-chars", type=int, default=3000, help="per-session text cap")
    ap.add_argument("--snippets", action="store_true", help="show query-relevant turns instead of the session head")
    ap.add_argument("--max-tokens", type=int, default=4096, help="must cover the model's reasoning")
    ap.add_argument("--timeout", type=float, default=20.0, help="latency budget per question; on timeout keep base order")
    ap.add_argument("--sample", type=int, default=0, help="random sample size (0 = all)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = json.load(open(args.perq))
    if args.sample:
        rows = random.Random(args.seed).sample(rows, args.sample)
    per_q = json.load(open(args.sessions)) if args.sessions else None
    sess = locomo_sessions(args.locomo) if args.locomo else {}
    key = load_key()
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "calls.jsonl").open("a")

    base1 = new1 = fails = 0
    hard_err = 0
    lat, toks = [], 0
    for r in rows:
        if per_q is not None:
            sess = {k: tuple(v) for k, v in per_q[r["qid"]].items()}
        cands = [s for s in r["retrieved_top10_sids"][: args.top_n] if s in sess]
        gold = set(r["answer_sids"])
        base1 += int(bool(cands) and cands[0] in gold)
        bodies = (snippet(r["question"], [sess[s][1] for s in cands], args.max_chars) if args.snippets
                  else [sess[s][1][: args.max_chars] for s in cands])
        blocks = "\n\n".join(f"[{i}] date: {sess[s][0]}\n{b}" for i, (s, b) in enumerate(zip(cands, bodies)))
        raw, dt, usage = chat(key, args.model, PROMPT.format(question=r["question"], blocks=blocks, last=len(cands) - 1),
                              args.max_tokens, args.timeout)
        lat.append(dt)
        err = str(usage.get("error", ""))
        hard_err = hard_err + 1 if ("429" in err or "403" in err or "401" in err) else 0
        if hard_err >= 3:
            print("STOP: provider quota/auth errors 3x in a row:", err, flush=True)
            base1 -= int(bool(cands) and cands[0] in gold)  # drop the aborted question from both counts
            lat.pop()
            rows = rows[: rows.index(r)]
            break
        toks += int(usage.get("total_tokens", 0) or 0)
        m = re.fullmatch(r"\D*(\d+)\D*", raw or "")
        pick = int(m.group(1)) if m and int(m.group(1)) < len(cands) else 0
        if not m or int(m.group(1)) >= len(cands):  # empty, multi-number or out-of-range: keep base order, count it
            fails += 1
        new1 += int(bool(cands) and cands[pick] in gold)
        log.write(json.dumps({"qid": r["qid"], "qtype": r["qtype"], "cands": cands, "gold": sorted(gold),
                              "raw": raw, "pick": pick, "latency_s": round(dt, 2), "usage": usage}) + "\n")
        log.flush()

    n = len(rows)
    lat.sort()
    summary = {"n": n, "model": args.model, "top_n": args.top_n, "base_R@1": round(base1 / n, 4),
               "llm_R@1": round(new1 / n, 4), "unparsed_or_failed": fails,
               "latency_p50_s": round(lat[n // 2], 2), "latency_p90_s": round(lat[int(n * 0.9) - 1], 2),
               "total_tokens": toks}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
