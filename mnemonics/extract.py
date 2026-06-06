"""Optional LLM fact-extraction layer — verified facts with provenance.

Core mnemonics stays LLM-free; nothing in ingest/retrieve imports or calls
this module. Callers (benchmark bridges, opt-in users) invoke it explicitly
and feed the returned fact records into the regular ``ingest()`` path as
additional texts/meta. Every fact carries the source turn ids it was
distilled from, so a fact is always one hop away from its evidence —
"verified memory" extends to the semantic layer instead of being diluted
by it.

Design constraints:
  - No heavy imports (torch/sentence-transformers); safe to import anywhere.
  - LLM client is OpenAI-compatible and injected (tests use a fake).
  - Hard call budget (``max_calls``) so a runaway corpus cannot run up cost.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

_FACT_PROMPT = """You distill a chat session into atomic memory facts.

Each turn below is prefixed with its ID. Output a JSON array where each item is
{{"fact": "<one self-contained sentence>", "sources": ["<turn id>", ...]}}.

Rules:
1. A fact must be fully self-contained: include WHO (speaker names), WHAT, and
   WHEN. Resolve relative time ("yesterday", "last week") against the session
   date given below; write absolute dates when possible.
2. Only state what the turns actually say. No speculation, no advice given by
   the assistant, no chit-chat.
3. Prefer facts about people: events, preferences, plans, possessions,
   relationships, dates.
4. "sources" lists the turn ID(s) the fact comes from (usually one).
5. 0 to {max_facts} facts. Empty array if nothing memorable.

Session date: {session_date}
Session:
{turns_block}

JSON array only:"""


def _default_client():
    """OpenAI-compatible client from env (DeepSeek by default)."""
    from openai import OpenAI  # lazy: only when extraction is actually used
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return OpenAI(api_key=key)
    raise RuntimeError("fact extraction needs DEEPSEEK_API_KEY or OPENAI_API_KEY")


def _parse_json_array(raw: str) -> list[Any]:
    """Tolerant JSON-array parse: strips code fences / prose around the array."""
    if not raw:
        return []
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.MULTILINE).strip()
    s, e = txt.find("["), txt.rfind("]")
    if s < 0 or e <= s:
        return []
    try:
        arr = json.loads(txt[s:e + 1])
    except json.JSONDecodeError:
        return []
    return arr if isinstance(arr, list) else []


class FactExtractor:
    """Distills sessions into provenance-carrying atomic facts via an LLM.

    Args:
        client: OpenAI-compatible client; defaults to DeepSeek from env.
        model: chat model name.
        max_calls: hard budget — further sessions raise RuntimeError so a
            runaway corpus cannot silently run up an API bill.
        max_facts: per-session cap fed into the prompt and enforced on output.
    """

    def __init__(self, client=None, model: str = "deepseek-chat",
                 max_calls: int | None = None, max_facts: int = 25,
                 temperature: float = 0.0, retries: int = 3):
        self._client = client
        self.model = model
        self.max_calls = max_calls
        self.max_facts = max_facts
        self.temperature = temperature
        self.retries = retries
        self.stats = {"calls": 0, "facts": 0, "empty_sessions": 0,
                      "parse_failures": 0, "dropped_bad_sources": 0}

    @property
    def client(self):
        if self._client is None:
            self._client = _default_client()
        return self._client

    def extract_session(self, turns: list[dict],
                        session_date: str = "") -> list[dict]:
        """turns: [{"id": str, "speaker": str, "text": str}, ...]
        Returns: [{"text": fact, "source_ids": [...], "kind": "fact"}, ...]
        Facts whose every source id is unknown are dropped (hallucinated
        provenance); unknown ids inside a valid fact are filtered out.
        """
        valid_ids = {t.get("id") for t in turns if t.get("id")}
        if not valid_ids:
            return []
        if self.max_calls is not None and self.stats["calls"] >= self.max_calls:
            raise RuntimeError(f"fact extraction call budget exhausted "
                               f"({self.max_calls})")
        block = "\n".join(
            f"[{t.get('id')}] {t.get('speaker', '?')}: {t.get('text', '')}"
            for t in turns if t.get("text"))
        prompt = _FACT_PROMPT.format(max_facts=self.max_facts,
                                     session_date=session_date or "unknown",
                                     turns_block=block)
        raw = ""
        for attempt in range(self.retries + 1):
            try:
                self.stats["calls"] += 1
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1800,
                    temperature=self.temperature)
                raw = resp.choices[0].message.content or ""
                break
            except Exception:
                if attempt >= self.retries:
                    raise
                time.sleep(4 * (attempt + 1))

        arr = _parse_json_array(raw)
        if not arr:
            if raw.strip():
                self.stats["parse_failures"] += 1
            else:
                self.stats["empty_sessions"] += 1
            return []

        facts: list[dict] = []
        for item in arr[: self.max_facts]:
            if not isinstance(item, dict):
                continue
            text = (item.get("fact") or "").strip()
            srcs = item.get("sources") or []
            if not text or not isinstance(srcs, list):
                continue
            good = [s for s in srcs if s in valid_ids]
            if not good:
                self.stats["dropped_bad_sources"] += 1
                continue
            facts.append({"text": text, "source_ids": good, "kind": "fact"})
        self.stats["facts"] += len(facts)
        if not facts:
            self.stats["empty_sessions"] += 1
        return facts

    def extract_many(self, sessions: list[tuple[list[dict], str]]) -> list[list[dict]]:
        """[(turns, session_date), ...] -> per-session fact lists."""
        return [self.extract_session(t, d) for t, d in sessions]
