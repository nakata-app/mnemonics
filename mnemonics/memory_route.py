"""Label-free routing primitives for scale-aware memory retrieval.

The router does not inspect benchmark labels or stored answers.  It classifies
only the user query so higher-level retrieval can choose a cheap, specialized
candidate lane (time, preference, personal state, chronology) before an
expensive semantic reranker is invoked.
"""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

_NUM_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "couple": 2,
    "few": 3,
}
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
_RELATIVE_TARGET_RE = re.compile(
    r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|couple|few)"
    r"(?:\s+of)?\s+(day|week|month|year)s?\s+ago\b",
    re.IGNORECASE,
)
_DURATION_RE = re.compile(
    r"\bhow\s+(?:many\s+(?:days?|weeks?|months?|years?)\s+ago|long)\b",
    re.IGNORECASE,
)
_DURATION_EVENT_PREFIXES = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"^how many (?:days?|weeks?|months?|years?) ago did i\s+",
    r"^how many (?:days?|weeks?|months?|years?) (?:have|had) passed since i\s+",
    r"^how many (?:days?|weeks?|months?|years?) since i\s+",
    r"^how long ago did i\s+",
))
_LAST_WEEKDAY_RE = re.compile(
    r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_CHRONOLOGY_RE = re.compile(
    r"\b(order|sequence|earliest|latest|most recent|first|second|third|before|after)\b",
    re.IGNORECASE,
)
_PREFERENCE_RE = re.compile(
    r"\b(recommend(?:ation|ations|ed|ing)?|suggest(?:ion|ions|ed|ing)?|advice|tips?|"
    r"what should i|should i|do you think|good idea|would i like|would i prefer|"
    r"what should i (?:serve|cook|make))\b",
    re.IGNORECASE,
)
_PERSONAL_STATE_RE = re.compile(
    r"\b(my|i|me)\b.*\b(previous|current|occupation|job|role|favorite|gift|practice|"
    r"using|working|living|own|have|had|did|was|were|am)\b|"
    r"\bwhat (?:was|is|did) my\b|\bhow much time do i\b|"
    r"\b(?:did|do|have|had|was|am) i\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MemoryRoute:
    """Query-only routing decision.

    ``lanes`` is ordered by preferred cheap retrieval lane. ``target_time`` is
    populated only when the query itself gives enough information to compute a
    concrete time anchor. ``window_days`` is a retrieval tolerance, not a gold
    answer window.
    """

    lanes: tuple[str, ...]
    target_time: datetime | None = None
    window_days: int | None = None


def _parse_num(raw: str) -> int:
    raw = raw.lower()
    return int(raw) if raw.isdigit() else _NUM_WORDS[raw]


def _last_weekday(reference: datetime, weekday_name: str) -> datetime:
    target_weekday = list(calendar.day_name).index(weekday_name.capitalize())
    days_back = (reference.weekday() - target_weekday) % 7
    if days_back == 0:
        days_back = 7
    return reference - timedelta(days=days_back)


def duration_event_query(query: str) -> str:
    """Strip count wording so retrieval can identify the event before computing time.

    Example: ``How many weeks ago did I attend the sale?`` becomes
    ``I attend the sale``. Queries whose duration grammar is not a simple
    removable prefix are returned unchanged. No benchmark labels are used.
    """
    text = " ".join((query or "").split()).strip().rstrip("?")
    for pattern in _DURATION_EVENT_PREFIXES:
        rewritten = pattern.sub("I ", text, count=1)
        if rewritten != text:
            return rewritten
    return text


def route_memory_query(query: str, *, reference_time: datetime | None = None) -> MemoryRoute:
    """Route a query into productionizable memory lanes using query text only."""
    text = " ".join((query or "").split())
    if not text:
        return MemoryRoute(("semantic",))

    lanes: list[str] = []
    target: datetime | None = None
    window: int | None = None

    rel = _RELATIVE_TARGET_RE.search(text)
    if rel:
        lanes.append("temporal-target")
        n = _parse_num(rel.group(1))
        unit = rel.group(2).lower()
        if reference_time is not None:
            target = reference_time - timedelta(days=n * _UNIT_DAYS[unit])
            # Month/year wording is approximate; day/week wording is usually exact.
            window = 1 if unit in {"day", "week"} else max(2, n)

    weekday = _LAST_WEEKDAY_RE.search(text)
    if weekday:
        if "temporal-target" not in lanes:
            lanes.append("temporal-target")
        if reference_time is not None:
            target = _last_weekday(reference_time, weekday.group(1))
            window = 0

    if _DURATION_RE.search(text):
        lanes.append("temporal-duration")
    if _CHRONOLOGY_RE.search(text):
        lanes.append("chronology")
    if _PREFERENCE_RE.search(text):
        lanes.append("preference")
    if _PERSONAL_STATE_RE.search(text):
        lanes.append("personal-state")

    lanes.append("semantic")
    # Stable dedupe while preserving lane priority.
    lanes = list(dict.fromkeys(lanes))
    return MemoryRoute(tuple(lanes), target, window)
