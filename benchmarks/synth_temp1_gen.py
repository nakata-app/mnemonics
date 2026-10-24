"""Synthetic dev set for LongMemEval's temporal-reasoning TYPE, relative-anchor subclass ("temp1").

Built from the paper's public ability definition (arXiv 2410.10813, temporal reasoning: "awareness of temporal
aspects of user information, including explicit time references in the user information and the timestamp
metadata"). No M/S question is read. The set exercises the subclass where the QUESTION anchors the target
event relative to the question date ("two weeks ago", "last Saturday") and the EVENT is mentioned in a session
that may itself be dated after the event ("yesterday I ...", "last Friday I ..."). The four sessions of an item
are the same recurring kind of event in one domain; only their dates and one unique detail differ, so the
item is a topical near-tie that time disambiguates.

Two subcommands, so the LLM step runs as a deterministic Kaggle kernel (Qwen3-14B, open weights) instead of a
paid cloud call:
  prompts   -> ~/.local/share/mnemonics-exam15/gen_prompts.jsonl  ({qid, prompt}; 40 domains x 3 seeds = 120)
  assemble  -> reads ~/.local/share/mnemonics-exam15/kaggle-gen/raw.jsonl, validates each item mechanically,
               keeps the first VALID 2 per domain (validity only, never a score), fills the time tokens with a
               seeded dating scheme and writes the dataset in LongMemEval field layout.

Dating scheme (seed 15, fixed in this file): question date Q in 2024-03-01..2024-06-30. Items cycle over four
anchor kinds: "N days ago" (3..6, exact), "N weeks ago" (1..4, event = Q-7N +-1 day), "N months ago" (1..3,
event = Q-30N +-2 days), "last <weekday>" (exact). Evidence session date S0: on odd items S0 = E ("today");
on even items S0 = E + d, d in {1 (yesterday), 2..6 (last <weekday of E>), 7 (a week ago), 14 (two weeks
ago)}, always S0 < Q. Distractor events E_k are at least window+3 days from E (window = the router's
tolerance for that anchor kind) and before Q; on even items ONE distractor's SESSION date is placed inside the
window around E so that session-timestamp routing alone points at the wrong session (the M failure mode the
2026-10-07 arithmetic documented, 10/20). Items 64..79 use the item's duration question ("How many weeks ago
did I <evidence event>?"), which names the event: a within-set control on which a target-time lever must be
inert. Haystack = own 4 sessions + sessions of OTHER-domain items up to 40, shuffled, each session keeps the
date assigned in its own item.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import date, timedelta
from pathlib import Path

D = Path.home() / ".local/share/mnemonics-exam15"
SEED = 15
PER_DOMAIN_PROMPTS = 3
PER_DOMAIN_KEEP = 2
HAYSTACK = 40
N_CONTROL = 16  # last items use the duration question

DOMAINS = [
    "dentist and medical appointments", "car repairs and servicing", "restaurant visits", "hiking trips",
    "concerts and live shows", "job interviews", "home repairs", "flights and train trips", "haircuts",
    "visits to relatives", "gym personal records", "online orders and deliveries", "library and bookshop visits",
    "cinema outings", "bike maintenance", "pet vet visits", "cooking a new recipe", "museum and gallery visits",
    "coffee shop discoveries", "sports matches watched", "parcel returns", "phone and laptop repairs",
    "parking tickets and fines", "doctor prescriptions", "garden planting", "furniture purchases",
    "language lessons", "driving tests and lessons", "charity and volunteering shifts", "weddings and parties",
    "swimming sessions", "bank and insurance calls", "children's school events", "camping weekends",
    "dinner parties hosted", "fishing trips", "tailor and shoe repair", "podcast or lecture attended",
    "board game nights", "football training sessions",
]

PROMPT = """You are generating one test item for a long-term memory benchmark of a personal chat assistant.
Domain: {domain}. Variant seed: {seed}.

The user has FOUR separate chats with the assistant. In each chat the user mentions ONE event of the same
recurring kind in this domain (e.g. four different dentist visits), each with a DIFFERENT unique detail
(a name, place, brand, number or outcome) that is the memorable answer for that event. The chats are
otherwise about something ordinary in the domain (asking for advice, planning, small talk).

Write JSON with these keys:
- "event_kind": short noun phrase for the recurring event (e.g. "dentist visit").
- "sessions": a list of exactly 4 objects {{"detail": ..., "chat": [...]}}:
    - "detail": the unique memorable detail of that event, 2-8 words, verbatim as it appears in the chat.
    - "chat": 6-8 messages alternating "user"/"assistant", starting with "user". Exactly ONE user message
      mentions the event and WHEN it happened using the literal token {{REL}} in place of the time phrase,
      e.g. "{{REL}} I finally had my cleaning at Dr. Okafor's clinic". The token {{REL}} must appear exactly
      once per chat and only in a user message. Do not write any other dates or time phrases.
- "question": what the user asks weeks later to recall the detail of the event that happened at a time given
  by the literal token {{WHEN}} (exactly once), e.g. "What was the name of the dentist I saw {{WHEN}}?".
  The question must NOT contain any of the four details.
- "duration_question": a question of the form "How many weeks ago did I <the event of sessions[0], named by
  its detail>?" that contains sessions[0].detail verbatim and no {{WHEN}} token.
Each chat message is {{"role": ..., "content": ...}}. Reply with JSON only."""

ANCHOR_KINDS = ("days", "weeks", "months", "weekday")
NUM_WORDS = {1: "a", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def anon(s: str) -> str:
    return "s" + hashlib.sha1(s.encode()).hexdigest()[:12]


def prompts() -> None:
    D.mkdir(parents=True, exist_ok=True)
    out = D / "gen_prompts.jsonl"
    with open(out, "w") as f:
        for d in DOMAINS:
            for v in range(PER_DOMAIN_PROMPTS):
                qid = f"{re.sub(r'\\W+', '_', d)}__{v}"
                f.write(json.dumps({"qid": qid, "prompt": PROMPT.format(domain=d, seed=v)}) + "\n")
    print("wrote", out, len(DOMAINS) * PER_DOMAIN_PROMPTS)


def parse_json(raw: str):
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None


def chat_ok(chat) -> bool:
    if not isinstance(chat, list) or len(chat) < 4:
        return False
    for k, m in enumerate(chat):
        if not isinstance(m, dict) or m.get("role") != ("user" if k % 2 == 0 else "assistant"):
            return False
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            return False
    user_text = " ".join(m["content"] for m in chat if m["role"] == "user")
    asst_text = " ".join(m["content"] for m in chat if m["role"] == "assistant")
    return user_text.count("{REL}") == 1 and "{REL}" not in asst_text


def valid(item) -> str | None:
    """Mechanical validity (returns a reason string when invalid). Never reads any score."""
    if not isinstance(item, dict):
        return "not-object"
    ss = item.get("sessions")
    if not isinstance(ss, list) or len(ss) != 4:
        return "sessions!=4"
    details = []
    for s in ss:
        if not isinstance(s, dict) or not isinstance(s.get("detail"), str) or not s["detail"].strip():
            return "detail"
        if not chat_ok(s.get("chat")):
            return "chat"
        details.append(s["detail"].strip())
    low = [d.lower() for d in details]
    if len(set(low)) != 4:
        return "details-not-distinct"
    for k, s in enumerate(ss):
        text = " ".join(m["content"] for m in s["chat"]).lower()
        if low[k] not in text:
            return "detail-not-in-own-chat"
        if any(low[j] in text for j in range(4) if j != k):
            return "detail-leaks-into-other-chat"
    q = item.get("question")
    if not isinstance(q, str) or q.count("{WHEN}") != 1 or any(d in q.lower() for d in low):
        return "question"
    dq = item.get("duration_question")
    if not isinstance(dq, str) or "{WHEN}" in dq or low[0] not in dq.lower() or not dq.lower().startswith("how many"):
        return "duration_question"
    if any(d in dq.lower() for d in low[1:]):
        return "duration_question-leaks"
    return None


def rel_phrase(delta: int, event: date) -> str:
    if delta == 0:
        return "Today"
    if delta == 1:
        return "Yesterday"
    if 2 <= delta <= 6:
        return f"Last {WEEKDAYS[event.weekday()]}"
    if delta == 7:
        return "A week ago"
    if delta == 14:
        return "Two weeks ago"
    raise ValueError(delta)


def anchor(kind: str, rng: random.Random, q: date):
    """Return (phrase, event_date, window_days) for the question anchor; window = router tolerance."""
    if kind == "days":
        n = rng.randint(3, 6)
        return f"{NUM_WORDS[n]} days ago", q - timedelta(days=n), 1
    if kind == "weeks":
        n = rng.randint(1, 4)
        return (f"{NUM_WORDS[n]} week{'s' if n > 1 else ''} ago", q - timedelta(days=7 * n + rng.choice((-1, 0, 1))), 1)
    if kind == "months":
        n = rng.randint(1, 3)
        return (f"{NUM_WORDS[n]} month{'s' if n > 1 else ''} ago", q - timedelta(days=30 * n + rng.choice((-2, -1, 0, 1, 2))),
                max(2, n))
    back = rng.randint(1, 7)
    e = q - timedelta(days=back)
    return f"last {WEEKDAYS[e.weekday()]}", e, 0


def fill(chat, phrase: str):
    out = []
    for m in chat:
        c = m["content"]
        k = c.find("{REL}")
        if k >= 0:
            head = c[:k].rstrip()
            p = phrase if (not head or head[-1] in ".!?") else phrase[0].lower() + phrase[1:]
            c = c[:k] + p + c[k + 5:]
        out.append({"role": m["role"], "content": c})
    return out


def assemble() -> None:
    raw_path = D / "kaggle-gen/raw.jsonl"
    rows = [json.loads(l) for l in open(raw_path) if l.strip()]
    by_domain: dict[str, list] = {}
    reasons: dict[str, int] = {}
    for r in rows:
        dom, _v = r["qid"].rsplit("__", 1)
        item = parse_json(r["raw"])
        why = valid(item)
        if why:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        by_domain.setdefault(dom, [])
        if len(by_domain[dom]) < PER_DOMAIN_KEEP:
            by_domain[dom].append(item)
    items = [(dom, it) for dom in sorted(by_domain) for it in by_domain[dom]]
    print("valid items kept", len(items), "| rejected reasons", json.dumps(reasons, sort_keys=True))

    rng = random.Random(SEED)
    built = []
    for i, (dom, it) in enumerate(items):
        q = date(2024, 3, 1) + timedelta(days=rng.randint(0, 121))
        kind = ANCHOR_KINDS[i % 4]
        when, e, win = anchor(kind, rng, q)
        later = i % 2 == 0  # even items: evidence session dated after the event
        d0 = rng.choice((1, 2, 3, 4, 5, 6, 7, 14)) if later else 0
        s0 = e + timedelta(days=d0)
        if s0 >= q:
            d0, s0 = 0, e
        sessions = [{"date": s0, "event": e, "chat": fill(it["sessions"][0]["chat"], rel_phrase(d0, e))}]
        used = {e}
        trap = rng.randrange(1, 4) if later else None
        for k in range(1, 4):
            for _ in range(200):
                ek = q - timedelta(days=rng.randint(1, 120))
                if ek < q and all(abs((ek - u).days) >= win + 3 for u in used):
                    break
            else:
                raise SystemExit("could not place distractor event")
            used.add(ek)
            if k == trap:
                sk = e + timedelta(days=rng.randint(-win, win))
                dk = (sk - ek).days
                if dk not in (0, 1, 2, 3, 4, 5, 6, 7, 14) or sk >= q:
                    dk = rng.choice((1, 2, 7))
                    sk = ek + timedelta(days=dk)
            else:
                dk = rng.choice((0, 0, 1, 2, 3, 4, 5, 6, 7, 14))
                sk = ek + timedelta(days=dk)
            if sk >= q:
                dk, sk = 0, ek
            sessions.append({"date": sk, "event": ek, "chat": fill(it["sessions"][k]["chat"], rel_phrase(dk, ek))})
        control = i >= len(items) - N_CONTROL
        question = it["duration_question"] if control else it["question"].replace("{WHEN}", when)
        built.append({"i": i, "domain": dom, "q": q, "when": when, "kind": kind, "later": later, "trap": trap,
                      "control": control, "question": question, "answer": it["sessions"][0]["detail"],
                      "event_kind": it.get("event_kind", ""), "sessions": sessions})

    pool = [(b["i"], j, b["domain"], s) for b in built for j, s in enumerate(b["sessions"])]
    out, meta = [], []
    for b in built:
        own = [(b["i"], j, b["domain"], s) for j, s in enumerate(b["sessions"])]
        others = [p for p in pool if p[0] != b["i"] and p[2] != b["domain"]]
        rng.shuffle(others)
        hay = own + others[: HAYSTACK - len(own)]
        rng.shuffle(hay)
        sids = [anon(f"temp1_{oi}_{oj}") for (oi, oj, _, _) in hay]
        out.append({"question_id": f"synthtemp1_{b['i']}", "question_type": "temporal-reasoning",
                    "question": b["question"], "answer": b["answer"], "question_date": b["q"].isoformat(),
                    "answer_session_ids": [anon(f"temp1_{b['i']}_0")], "haystack_session_ids": sids,
                    "haystack_dates": [s["date"].isoformat() for (_, _, _, s) in hay],
                    "haystack_sessions": [s["chat"] for (_, _, _, s) in hay]})
        meta.append({"qid": f"synthtemp1_{b['i']}", "domain": b["domain"], "kind": b["kind"], "later": b["later"],
                     "trap": b["trap"], "control": b["control"], "when": b["when"], "event_kind": b["event_kind"],
                     "event_date": b["sessions"][0]["event"].isoformat(), "own_dates": [s["date"].isoformat() for s in b["sessions"]],
                     "own_events": [s["event"].isoformat() for s in b["sessions"]]})
    (D / "temp1.json").write_text(json.dumps(out))
    (D / "temp1_meta.json").write_text(json.dumps(meta, indent=1))
    (D / "temp1_qdates.json").write_text(json.dumps({r["question_id"]: r["question_date"] for r in out}))
    kinds = {k: sum(m["kind"] == k and not m["control"] for m in meta) for k in ANCHOR_KINDS}
    print(f"wrote {len(out)} questions x {HAYSTACK} sessions -> {D / 'temp1.json'} | anchor items {kinds} | "
          f"control {sum(m['control'] for m in meta)} | later-dated evidence {sum(m['later'] for m in meta)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prompts", "assemble"])
    a = ap.parse_args()
    prompts() if a.cmd == "prompts" else assemble()
