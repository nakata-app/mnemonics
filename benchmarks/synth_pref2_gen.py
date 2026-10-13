"""Synthetic dev set for LongMemEval's single-session-preference TYPE, built from the paper's public definition
(arXiv 2410.10813): "tests whether the model can utilize the user information to generate a personalized
response"; evidence is conveyed indirectly by the user; the rubric accepts an answer that "recalls and utilizes
the user's personal information correctly". No M/S question is read.

Per item an LLM (evren deepseek-v4-flash) writes, as JSON: a personal fact stated in passing inside an evidence
session about something else, a later request for personalized help in that domain that does not restate the
fact, and three same-domain distractor sessions with no relevant personal information. Each question's haystack
= its evidence + its distractors + sessions of other items, dated. Output in LongMemEval field layout.
Generation is cached per item (raw JSON), so the dataset is fixed once built and reruns do not resample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

DOMAINS = [
    "photography", "home cooking", "travel planning", "strength training", "playing guitar", "vegetable gardening",
    "video games", "home office setup", "dog care", "coffee brewing", "reading fiction", "hiking", "road cycling",
    "skincare", "car maintenance", "baking bread", "watching films", "clothing and style", "learning Spanish",
    "camping", "yoga", "board games", "smartphone choice", "watercolor painting", "knitting", "smart home devices",
    "running races", "freshwater aquarium", "budget planning", "podcast listening", "houseplants", "cat care",
    "sleep habits", "meal prep", "piano practice", "birdwatching", "rock climbing", "tea", "sewing", "3D printing",
]

PROMPT = """You are generating one test item for a long-term memory benchmark of a personal chat assistant.
Domain: {domain}. Variant seed: {seed}.

Write JSON with these keys:
- "fact": one concrete personal detail about the user relevant to {domain} (something they own, a constraint,
  a past experience, or a taste), e.g. a specific model, an allergy, a skill level, a dislike.
- "evidence_session": a realistic chat (8-12 messages, alternating "user"/"assistant") where the user is talking
  about something else and mentions the fact only in passing, indirectly, without the words "prefer", "favorite"
  or "like".
- "question": a request the user makes weeks later for personalized help, advice or a recommendation in
  {domain}. It must NOT restate the fact, but a good answer must use it.
- "distractor_sessions": a list of 3 chats (6-10 messages each) about {domain} in which the user asks generic
  questions and reveals no personal detail relevant to the question.
Each chat is a list of {{"role": ..., "content": ...}}. Reply with JSON only."""


def anon(s: str) -> str:
    return "s" + hashlib.sha1(s.encode()).hexdigest()[:12]


def call(prompt: str) -> str:
    body = json.dumps({"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 6000, "temperature": 0.7,
                       "response_format": {"type": "json_object"}}).encode()
    req = urllib.request.Request("https://evren-llmapi.ssyz.org.tr/v1/chat/completions", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {os.environ['EVREN_API_KEY']}",
                                          "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except Exception as e:  # error class only
            if attempt == 4:
                raise SystemExit(f"generation failed: {type(e).__name__} {getattr(e, 'code', '')}")
            time.sleep(2 ** (attempt + 2))
    return ""


def valid(item: dict) -> bool:
    chats = [item.get("evidence_session")] + list(item.get("distractor_sessions") or [])
    ok_chat = lambda c: isinstance(c, list) and len(c) >= 4 and all(
        isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content") for m in c)
    return (isinstance(item.get("fact"), str) and isinstance(item.get("question"), str)
            and len(chats) == 4 and all(ok_chat(c) for c in chats)
            and not re.search(r"\b(prefer|favou?rite)\b", " ".join(m["content"] for m in item["evidence_session"]
                                                                   if m["role"] == "user"), re.I))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True, help="dir of raw generated items (one json per item)")
    ap.add_argument("--per-domain", type=int, default=2)
    ap.add_argument("--haystack", type=int, default=40)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    items = []
    for d in DOMAINS:
        for v in range(args.per_domain):
            f = args.cache / f"{re.sub(r'\\W+', '_', d)}_{v}.json"
            item = json.loads(f.read_text()) if f.exists() else None
            tries = 0
            while item is None or not valid(item):
                tries += 1
                if tries > 3:
                    item = None
                    break
                try:
                    item = json.loads(call(PROMPT.format(domain=d, seed=v)))
                except json.JSONDecodeError:
                    item = None
            if item is None:
                print("skip", d, v)
                continue
            f.write_text(json.dumps(item))
            items.append((d, v, item))
    print("items", len(items))

    rng = random.Random(args.seed)
    pool = [(i, j, s) for i, (_, _, it) in enumerate(items) for j, s in enumerate([it["evidence_session"]] + it["distractor_sessions"])]
    out = []
    for i, (d, v, it) in enumerate(items):
        own = [(i, 0, it["evidence_session"])] + [(i, j + 1, s) for j, s in enumerate(it["distractor_sessions"])]
        others = [p for p in pool if p[0] != i]
        rng.shuffle(others)
        hay = own + others[: args.haystack - len(own)]
        rng.shuffle(hay)
        day = date(2024, 1, 1) + timedelta(days=rng.randint(0, 30))
        sids, dates, sessions = [], [], []
        for (oi, oj, s) in hay:
            sids.append(anon(f"pref2_{oi}_{oj}"))
            dates.append(day.isoformat())
            sessions.append(s)
            day += timedelta(days=rng.randint(1, 9))
        out.append({"question_id": f"synthpref2_{i}", "question_type": "single-session-preference",
                    "question": it["question"], "answer": it["fact"], "question_date": day.isoformat(),
                    "answer_session_ids": [anon(f"pref2_{i}_0")], "haystack_session_ids": sids,
                    "haystack_dates": dates, "haystack_sessions": sessions})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"wrote {len(out)} questions x {args.haystack} sessions -> {args.out}")


if __name__ == "__main__":
    main()
