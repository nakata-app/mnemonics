"""Generate a synthetic first-person preference dev set in LongMemEval format.

Purpose: develop the preference recall lane off the M exam. M's single-session-preference type is the
worst recall hole (R@10 0.733) because the question and the stated preference share little vocabulary:
the user says "I can't stand crowded bars, I'd take a quiet corner cafe any day" and the question asks
"what kind of place do I prefer for a night out". Dense retrieval must bridge that semantic gap, and
decoy sessions that echo the question's surface words outrank the gold.

This generator reproduces that structure deterministically, no LLM:
- each item has a GOLD statement (colloquial, first person, vocabulary disjoint from the question),
- a QUESTION (abstract phrasing),
- DECOYS (sentences that reuse the question's surface words but mean something else / another domain).
The gold statement is buried mid-session in filler chit-chat; decoys and filler fill the haystack, all
dated across a timeline. Output matches the fields longmemeval_eval / extract_m_candidates consume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date, timedelta
from pathlib import Path

# Each item: (domain, gold [first person, vocabulary disjoint from the question], question [abstract],
#   answer, decoys [echo the QUESTION's topic/surface words but state NO preference of the user -- news,
#   other people, logistics. They sit near the question in embedding space, so with enough of them the
#   differently-worded gold drops out of the top-k: exactly how M's preference gold is lost]).
ITEMS = [
    ("venue",
     "honestly I can't stand packed noisy bars, I'd always take a quiet little corner cafe instead",
     "What kind of place do I prefer for a night out?",
     "a quiet cafe, not crowded bars",
     ["we were debating where to go for a night out this weekend, so many places",
      "that new place everyone wants for a night out is fully booked for weeks",
      "my coworkers are planning a big night out at some bar downtown",
      "there's an article ranking the best places in town for a night out"]),
    ("coffee",
     "i gave up dark roast years ago, anything that bitter wrecks my stomach, light and fruity is my thing",
     "How do I like my coffee?",
     "light, fruity roast; not dark/bitter",
     ["the barista asked how I like my coffee and I was still half asleep",
      "the office finally got a machine so now everyone argues how they like their coffee",
      "there's a quiz going around about how you like your coffee says about you",
      "my sister keeps asking how I like my coffee so she can buy beans"]),
    ("travel",
     "i never book those rigid guided tours, give me a backpack and no fixed plan and I'm happy",
     "What's my preferred travel style?",
     "spontaneous backpacking, no fixed itinerary",
     ["the travel agent asked about my preferred travel style for the quote",
      "there's a survey about preferred travel style doing the rounds at work",
      "my parents have a very different preferred travel style from each other",
      "the magazine had a feature on how your preferred travel style changes with age"]),
    ("music",
     "studio pop does nothing for me, it's raw live recordings with crowd noise that I put on repeat",
     "What music do I enjoy most?",
     "raw live recordings, not polished studio pop",
     ["the app keeps asking what music I enjoy most so it can recommend",
      "we argued for an hour about what music everyone enjoys most on road trips",
      "there's a year-end list of the music people enjoyed most this year",
      "my roommate's taste in the music he enjoys most is nothing like mine"]),
    ("work",
     "open offices kill me, I do my best thinking alone at home with the door shut and no meetings",
     "How do I prefer to work?",
     "solo, remote, few meetings; not open offices",
     ["HR sent a survey asking how we prefer to work after the move",
      "the whole team has a different opinion on how they prefer to work",
      "there's a big debate online about how people prefer to work these days",
      "my manager asked how I prefer to work but then scheduled five meetings"]),
    ("food",
     "i steer clear of heavy creamy dishes, a bright bowl of grilled veg and lemon is what I crave",
     "What kind of food do I go for?",
     "light grilled vegetables, citrus; avoids creamy/heavy",
     ["the waiter asked what kind of food I go for so he could suggest specials",
      "there's a trend piece on what kind of food people go for when stressed",
      "my friends can never agree on what kind of food to go for on Fridays",
      "the delivery app keeps guessing what kind of food I go for and gets it wrong"]),
    ("exercise",
     "i can't deal with crowded gyms and machines, a long solo trail run is the only workout that sticks for me",
     "What's my go-to way to stay active?",
     "solo trail running; dislikes crowded gyms",
     ["the trainer asked what my go-to way to stay active is before the plan",
      "there's a poll about everyone's go-to way to stay active in winter",
      "my doctor wants to know my go-to way to stay active for the checkup",
      "the app keeps nagging about a go-to way to stay active each day"]),
    ("reading",
     "thick fantasy series exhaust me, I'd rather a short sharp essay I can finish in one sitting",
     "What do I like to read?",
     "short essays; not long fantasy series",
     ["the librarian asked what I like to read to set up recommendations",
      "there's a book club survey about what everyone likes to read this season",
      "my nephew asked what I like to read so he could pick a gift",
      "the article was about how what we like to read shapes our thinking"]),
    ("shopping",
     "fast-fashion hauls aren't for me, I buy one well-made thing and wear it for a decade",
     "What's my approach to buying clothes?",
     "few durable quality items; avoids fast fashion",
     ["the stylist asked about my approach to buying clothes before the session",
      "there's a documentary on how our approach to buying clothes hurts the planet",
      "my sister and I have the opposite approach to buying clothes entirely",
      "the store survey asked shoppers about their approach to buying clothes"]),
    ("communication",
     "phone calls stress me out, send me a written note and I'll answer thoughtfully whenever",
     "How do I prefer people reach me?",
     "written messages; dislikes phone calls",
     ["the new tool asked how I prefer people reach me for notifications",
      "there's a generational split in how people prefer others to reach them",
      "my team lead asked how I prefer people reach me outside work hours",
      "the form had a field for how you prefer people to reach you"]),
    ("weekend",
     "big social weekends drain me, a slow morning with a book and nowhere to be is my idea of perfect",
     "What's my ideal weekend?",
     "quiet, solitary, slow; not big social plans",
     ["everyone at lunch was describing their ideal weekend and they all differed",
      "there's a quiz claiming your ideal weekend reveals your personality",
      "my partner's ideal weekend and mine are almost the exact opposite",
      "the travel ad kept asking what my ideal weekend looks like"]),
]

FILLER = [
    "anyway, how was your week?",
    "did you see the game last night?",
    "I finally fixed that leaky tap at home.",
    "work's been busy but nothing crazy.",
    "the weather's finally turning, about time.",
    "remind me to call the dentist tomorrow.",
    "we're out of milk again somehow.",
    "traffic was a nightmare this morning.",
    "I started that show you mentioned, it's decent.",
    "my phone battery dies by noon these days.",
]


def anon(sid: str) -> str:
    return "s" + hashlib.sha1(sid.encode()).hexdigest()[:12]


def session_turns(rng: random.Random, core_user_line: str) -> list[dict]:
    """A short chat session with the core user line buried mid-way in filler."""
    n_before = rng.randint(1, 3)
    n_after = rng.randint(1, 3)
    turns: list[dict] = []
    for _ in range(n_before):
        turns.append({"role": "user", "content": rng.choice(FILLER)})
        turns.append({"role": "assistant", "content": "got it, thanks for sharing."})
    turns.append({"role": "user", "content": core_user_line})
    turns.append({"role": "assistant", "content": "noted."})
    for _ in range(n_after):
        turns.append({"role": "user", "content": rng.choice(FILLER)})
        turns.append({"role": "assistant", "content": "makes sense."})
    return turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("benchmarks/data/synth/pref_dev.json"))
    ap.add_argument("--haystack", type=int, default=30, help="sessions per question (1 gold + distractors)")
    ap.add_argument("--repeats", type=int, default=5, help="question variants per item (different haystacks)")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    start = date(2024, 1, 1)

    # a pool of distractor user-lines: every item's hard decoys + its gold phrased for OTHER questions
    decoy_lines: list[str] = []
    for _, _, _, _, hard in ITEMS:
        decoy_lines.extend(hard)
    gold_lines = [g for _, g, _, _, _ in ITEMS]

    out = []
    for domain, gold, question, answer, hard in ITEMS:
        for rep in range(args.repeats):
            qid = f"synthpref_{domain}_{rep}"
            # always include THIS item's hard decoys (they echo the question and out-rank the buried
            # gold), then fill the rest with other items' lines; never this item's own gold.
            others = [l for l in decoy_lines + gold_lines if l != gold and l not in hard]
            rng.shuffle(others)
            distractor_lines = hard + others[: max(0, args.haystack - 1 - len(hard))]

            lines = distractor_lines + [gold]
            rng.shuffle(lines)
            gold_pos = lines.index(gold)

            sids, dates, sessions = [], [], []
            d = start + timedelta(days=rng.randint(0, 20))
            for si, line in enumerate(lines):
                raw = f"{qid}_sess{si}"
                sids.append(anon(raw))
                dates.append(d.isoformat())
                sessions.append(session_turns(rng, line))
                d += timedelta(days=rng.randint(2, 12))

            out.append({
                "question_id": qid,
                "question_type": "single-session-preference",
                "question": question,
                "answer": answer,
                "answer_session_ids": [sids[gold_pos]],
                "haystack_session_ids": sids,
                "haystack_dates": dates,
                "haystack_sessions": sessions,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"wrote {len(out)} questions, {args.haystack} sessions each, -> {args.out}")


if __name__ == "__main__":
    main()
