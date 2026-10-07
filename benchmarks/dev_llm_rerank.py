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
                out[f'{conv["sample_id"]}_s{m.group(1)}'] = (c[f"session_{m.group(1)}_date_time"], text)
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
    ap.add_argument("--locomo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--max-chars", type=int, default=3000, help="per-session text cap")
    ap.add_argument("--max-tokens", type=int, default=4096, help="must cover the model's reasoning")
    ap.add_argument("--timeout", type=float, default=20.0, help="latency budget per question; on timeout keep base order")
    ap.add_argument("--sample", type=int, default=0, help="random sample size (0 = all)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = json.load(open(args.perq))
    if args.sample:
        rows = random.Random(args.seed).sample(rows, args.sample)
    sess = locomo_sessions(args.locomo)
    key = load_key()
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "calls.jsonl").open("a")

    base1 = new1 = fails = 0
    lat, toks = [], 0
    for r in rows:
        cands = [s for s in r["retrieved_top10_sids"][: args.top_n] if s in sess]
        gold = set(r["answer_sids"])
        base1 += int(bool(cands) and cands[0] in gold)
        blocks = "\n\n".join(f"[{i}] date: {sess[s][0]}\n{sess[s][1][: args.max_chars]}" for i, s in enumerate(cands))
        raw, dt, usage = chat(key, args.model, PROMPT.format(question=r["question"], blocks=blocks, last=len(cands) - 1),
                              args.max_tokens, args.timeout)
        lat.append(dt)
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
