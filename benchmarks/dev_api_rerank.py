"""Final-step rerank of a finished run's top-N sessions with a dedicated reranker API (evren /v1/rerank,
qwen3-reranker-8b). One forward pass per question, no LLM reasoning, so it stays far inside the 20 s budget.

Replays exactly on the saved top-N (it is the last pipeline step). Reports R@1/R@5/R@10 after reranking;
R@10 cannot change when top-n <= 10. Runs from the Mac (HTTP only). The key is read from the keystore in
process and never placed in the environment or printed.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.request
from pathlib import Path

from dev_llm_rerank import KEYSTORE, locomo_sessions, snippet

URL = "https://evren-llmapi.ssyz.org.tr/v1/rerank"


def rerank(key: str, model: str, query: str, docs: list[str], timeout: float):
    # 429 = provider rate limit: back off and retry (8 tries, 2-256 s) instead of silently keeping base order.
    for attempt in range(8):
        scores, dt, usage = _rerank_once(key, model, query, docs, timeout)
        if scores is not None or "429" not in str(usage.get("error", "")):
            return scores, dt, usage
        time.sleep(2 ** (attempt + 1))
    return scores, dt, usage


def _rerank_once(key: str, model: str, query: str, docs: list[str], timeout: float):
    body = json.dumps({"model": model, "query": query, "documents": docs}).encode()
    req = urllib.request.Request(URL, data=body, method="POST",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        scores = {x["index"]: x["relevance_score"] for x in d["results"]}
        return [scores.get(i, float("-inf")) for i in range(len(docs))], time.time() - t0, d.get("usage", {})
    except Exception as e:  # error class only, never request headers
        return None, time.time() - t0, {"error": f"{type(e).__name__}: {getattr(e, 'code', '')}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perq", type=Path, required=True)
    ap.add_argument("--locomo", type=Path)
    ap.add_argument("--sessions", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="qwen3-reranker-8b")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--snippets", action="store_true")
    ap.add_argument("--chunks", action="store_true", help="score each turn chunk; session score = best chunk")
    ap.add_argument("--chunk-chars", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sids-key", default="retrieved_top10_sids")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--strict", action="store_true", help="abort on the first failed call instead of keeping base order")
    args = ap.parse_args()

    rows = json.load(open(args.perq))
    if args.sample:
        rows = random.Random(args.seed).sample(rows, args.sample)
    rows = rows[args.part::args.parts]
    done = {json.loads(l)["qid"] for l in open(args.out / "calls.jsonl")} if (args.out / "calls.jsonl").exists() else set()
    rows = [r for r in rows if r["qid"] not in done]
    per_q = json.load(open(args.sessions)) if args.sessions else None
    sess = locomo_sessions(args.locomo) if args.locomo else {}
    key = json.loads(KEYSTORE.read_text())["providers"]["evren"][0]
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "calls.jsonl").open("a")

    hits = {"base": [0, 0, 0], "new": [0, 0, 0]}
    fails, lat = 0, []
    for r in rows:
        if per_q is not None:
            sess = {k: tuple(v) for k, v in per_q[r["qid"]].items()}
        top = r[args.sids_key]
        cands = [s for s in top[: args.top_n] if s in sess]
        texts = [f"date: {sess[s][0]}\n{sess[s][1]}" for s in cands]
        if args.snippets:
            texts = [f"date: {sess[s][0]}\n{b}" for s, b in
                     zip(cands, snippet(r["question"], [sess[s][1] for s in cands], args.max_chars))]
        else:
            texts = [t[: args.max_chars] for t in texts]
        if args.chunks and cands:
            owner, docs = [], []
            for ci, s in enumerate(cands):
                body = sess[s][1]
                parts = [x for x in re.split(r"\n(?=\[user\]|[A-Z][\w .'-]{0,40}: )", body) if x.strip()] or [body]
                cur = ""
                for part in parts + [None]:
                    if part is None or (cur and len(cur) + len(part) > args.chunk_chars):
                        owner.append(ci); docs.append(f"date: {sess[s][0]}\n" + cur[: args.chunk_chars])
                        cur = ""
                    if part is not None:
                        cur = (cur + "\n" + part) if cur else part
            best, dt, usage, scores = [float("-inf")] * len(cands), 0.0, {}, []
            for b0 in range(0, len(docs), args.batch):
                sc, d1, usage = rerank(key, args.model, r["question"], docs[b0:b0 + args.batch], args.timeout)
                dt += d1
                if sc is None:
                    scores = None
                    break
                for o, v in zip(owner[b0:b0 + args.batch], sc):
                    best[o] = max(best[o], v)
            if scores is not None:
                scores = best
        else:
            scores, dt, usage = rerank(key, args.model, r["question"], texts, args.timeout) if cands else (None, 0.0, {})
        lat.append(dt)
        order = cands if scores is None else [c for _, c in sorted(zip(scores, cands), key=lambda x: -x[0])]
        if scores is None:
            fails += 1
            if args.strict:
                raise SystemExit(f"STRICT_ABORT qid={r['qid']} {usage}")
        new = order + [s for s in top if s not in order]
        gold = set(r["answer_sids"])
        for i, k in enumerate((1, 5, 10)):
            hits["base"][i] += any(s in gold for s in top[:k])
            hits["new"][i] += any(s in gold for s in new[:k])
        log.write(json.dumps({"qid": r["qid"], "qtype": r["qtype"], "cands": cands, "order": order,
                              "gold": sorted(gold), "latency_s": round(dt, 2), "usage": usage}) + "\n")
        log.flush()

    n = len(rows)
    lat.sort()
    summary = {"n": n, "model": args.model, "top_n": args.top_n, "max_chars": args.max_chars, "snippets": args.snippets, "chunks": args.chunks,
               "base": [round(x / n, 4) for x in hits["base"]], "reranked": [round(x / n, 4) for x in hits["new"]],
               "failed": fails, "latency_p50_s": round(lat[n // 2], 2), "latency_p90_s": round(lat[int(n * 0.9) - 1], 2)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
