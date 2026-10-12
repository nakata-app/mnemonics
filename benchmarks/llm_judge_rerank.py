"""LLM listwise judge over the top-N sessions of a finished rerank (dev first, then one logged M exam).

Base order: the `order` field of a previous dev_api_rerank calls.jsonl (qwen3-reranker-8b chunk-max), so the
judge only reorders sessions that stage already ranked. For each of the top-N candidates, the best turn chunks
(local cross-encoder ms-marco-MiniLM-L-12-v2 vs the question) are shown with the session date. The LLM returns
the session numbers that hold the needed information, most relevant first; those go to the front, the rest keep
base order. No question_type label and no gold are visible to the judge.

The API key is read inside this process from the environment and is never printed or logged.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from sentence_transformers import CrossEncoder

from dev_llm_rerank import locomo_sessions

PROMPT = """You are the retrieval judge of a personal memory assistant.
{today}Question: {question}

Below are {n} past conversation sessions (most relevant excerpts only), each with its date.
Decide which sessions contain the information needed to answer the question. Use the dates for questions
about time, order, the latest state, or how long ago something happened.

{blocks}

Reply with only a JSON list of session numbers, most relevant first, at most 5 numbers, e.g. [3, 0, 7]."""

PROVIDERS = {
    "deepseek": ("https://api.deepseek.com/v1/chat/completions", "DEEPSEEK_API_KEY"),
    "evren": ("https://evren-llmapi.ssyz.org.tr/v1/chat/completions", "EVREN_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY"),
    "ollama": ("http://localhost:11434/v1/chat/completions", None),
}


def chat(provider: str, model: str, prompt: str, max_tokens: int, timeout: float, reasoning: str = ""):
    url, env = PROVIDERS[provider]
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0,
                       **({"reasoning_effort": reasoning} if reasoning else {})}).encode()
    headers = {"Content-Type": "application/json"}
    if env:
        headers["Authorization"] = f"Bearer {os.environ[env]}"
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    t0 = time.time()
    err = ""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            return (d["choices"][0]["message"]["content"] or "").strip(), time.time() - t0, d.get("usage", {})
        except Exception as e:  # error class only
            code = getattr(e, "code", None)
            err = f"{type(e).__name__}: {code or ''}"
            if code == 429 or (isinstance(code, int) and code >= 500):
                time.sleep(2 ** (attempt + 1))
                continue
            return None, time.time() - t0, {"error": err}
    return None, time.time() - t0, {"error": err}


class JsonlSessions:
    """Per-question session texts from extract_m_candidates_jsonl.py, read one question at a time by offset."""

    def __init__(self, path: Path):
        self.f = open(path, "rb")
        self.off = {}
        pos = 0
        for line in self.f:
            m = re.match(rb'\{"qid": "([^"]+)"', line)
            if m:
                self.off[m.group(1).decode()] = pos
            pos += len(line)

    def __getitem__(self, qid: str) -> dict:
        self.f.seek(self.off[qid])
        return json.loads(self.f.readline())["sessions"]


def windows(text: str, width: int) -> list[str]:
    """Split a session into messages, and long messages into overlapping windows that keep the speaker tag,
    so evidence deep inside a long reply can still be selected."""
    msgs = [x for x in re.split(r"\n(?=\[user\] |\[assistant\] |[A-Z][\w .'-]{0,40}: )", text) if x.strip()] or [text]
    out = []
    for m in msgs:
        if len(m) <= width:
            out.append(m)
            continue
        tag = m.split("] ", 1)[0] + "]" if m.startswith("[") else m.split(": ", 1)[0] + ":"
        step = width * 3 // 4
        for i in range(0, len(m), step):
            out.append(m[i:i + width] if i == 0 else f"{tag} ...{m[i:i + width]}")
            if i + width >= len(m):
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perq", type=Path, required=True)
    ap.add_argument("--base-calls", type=Path, help="calls.jsonl whose `order` is the base order")
    ap.add_argument("--sids-key", default="retrieved_all_sids", help="perq base order when --base-calls is absent")
    ap.add_argument("--locomo", type=Path)
    ap.add_argument("--sessions", type=Path)
    ap.add_argument("--sessions-jsonl", type=Path, help="per-question JSONL from extract_m_candidates_jsonl.py")
    ap.add_argument("--qdates", type=Path, help="optional {qid: question_date}")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--provider", default="deepseek")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--windows-per-session", type=int, default=3)
    ap.add_argument("--window-chars", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--reasoning", default="none", help="reasoning_effort; '' to omit")
    ap.add_argument("--strict", action="store_true", help="abort on a failed call instead of keeping base order")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ce-cache", type=Path, help="json cache of window CE scores, keyed qid -> sid -> scores")
    ap.add_argument("--ce-only", action="store_true", help="only fill --ce-cache, no LLM calls")
    ap.add_argument("--front-k", type=int, default=3, help="judge picks moved to the front (dev: >3 hurts R@5)")
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
    qids = [q for q in perq if q in base]
    if args.limit:
        qids = qids[: args.limit]
    per_q = (JsonlSessions(args.sessions_jsonl) if args.sessions_jsonl
             else json.load(open(args.sessions)) if args.sessions else None)
    loco = locomo_sessions(args.locomo) if args.locomo else {}
    qdates = json.load(open(args.qdates)) if args.qdates else {}
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")
    cache = json.load(open(args.ce_cache)) if args.ce_cache and args.ce_cache.exists() else {}
    cache_dirty = False
    processed = 0

    args.out.mkdir(parents=True, exist_ok=True)
    done = set()
    if (args.out / "calls.jsonl").exists():
        done = {json.loads(l)["qid"] for l in open(args.out / "calls.jsonl") if l.strip()}
    log = (args.out / "calls.jsonl").open("a")

    for qid in qids:
        if qid in done:
            continue
        r = perq[qid]
        sess = {k: tuple(v) for k, v in per_q[qid].items()} if per_q is not None else loco
        cands = [s for s in base[qid][: args.top_n] if s in sess]
        blocks = []
        for i, s in enumerate(cands):
            ch = windows(sess[s][1], args.window_chars)
            key = f"{s}|{args.window_chars}"
            sc = cache.get(qid, {}).get(key)
            if sc is None or len(sc) != len(ch):
                sc = [float(x) for x in ce.predict([(r["question"], c) for c in ch], batch_size=64,
                                                     show_progress_bar=False)] if ch else []
                cache.setdefault(qid, {})[key] = sc
                cache_dirty = True
            keep = sorted(sorted(range(len(ch)), key=lambda j: -sc[j])[: args.windows_per_session])
            blocks.append(f"[{i}] date: {sess[s][0]}\n" + "\n...\n".join(ch[j] for j in keep))
        if args.ce_only:
            processed += 1
            if cache_dirty and processed % 20 == 0:
                args.ce_cache.write_text(json.dumps(cache))
                cache_dirty = False
            continue
        today = f"Today's date: {qdates[qid]}\n" if qid in qdates else ""
        raw, dt, usage = chat(args.provider, args.model,
                              PROMPT.format(today=today, question=r["question"], n=len(cands),
                                            blocks="\n\n".join(blocks)),
                              args.max_tokens, args.timeout, args.reasoning)
        if raw is None and args.strict:
            raise SystemExit(f"STRICT_ABORT qid={qid} {usage}")
        picks = []
        if raw:
            m = re.search(r"\[[\d,\s]*\]", raw)
            if m:
                picks = [int(x) for x in re.findall(r"\d+", m.group(0)) if int(x) < len(cands)]
        picks = list(dict.fromkeys(picks))[: args.front_k]
        order = [cands[i] for i in picks] + [s for s in base[qid] if s not in {cands[i] for i in picks}]
        log.write(json.dumps({"qid": qid, "qtype": r["qtype"], "base": base[qid][:30], "order": order[:30],
                              "gold": sorted(r["answer_sids"]), "raw": raw, "latency_s": round(dt, 2),
                              "usage": usage}) + "\n")
        log.flush()
        processed += 1
        if cache_dirty and args.ce_cache and processed % 20 == 0:
            args.ce_cache.write_text(json.dumps(cache))
            cache_dirty = False
    if cache_dirty and args.ce_cache:
        args.ce_cache.write_text(json.dumps(cache))
    if args.ce_only:
        print(f"ce cache filled for {processed} questions -> {args.ce_cache}")
        return

    rows = [json.loads(l) for l in open(args.out / "calls.jsonl") if l.strip()]
    ks = (1, 5, 10)
    H = {"base": [0, 0, 0], "judge": [0, 0, 0]}
    BT = defaultdict(lambda: {"n": 0, "base": [0, 0, 0], "judge": [0, 0, 0]})
    fails = 0
    for o in rows:
        g = set(o["gold"])
        fails += not o["raw"]
        BT[o["qtype"]]["n"] += 1
        for i, k in enumerate(ks):
            b = any(s in g for s in o["base"][:k]); j = any(s in g for s in o["order"][:k])
            H["base"][i] += b; H["judge"][i] += j
            BT[o["qtype"]]["base"][i] += b; BT[o["qtype"]]["judge"][i] += j
    n = len(rows)
    lat = sorted(o["latency_s"] for o in rows)
    summary = {"n": n, "provider": args.provider, "model": args.model, "top_n": args.top_n,
               "reasoning": args.reasoning, "failed": fails,
               "base": [round(x / n, 4) for x in H["base"]], "judge": [round(x / n, 4) for x in H["judge"]],
               "latency_p50_s": lat[n // 2], "latency_p90_s": lat[int(n * 0.9) - 1] if n >= 10 else lat[-1],
               "by_type": {t: {"n": v["n"], "base": [round(x / v["n"], 3) for x in v["base"]],
                               "judge": [round(x / v["n"], 3) for x in v["judge"]]} for t, v in sorted(BT.items())}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
