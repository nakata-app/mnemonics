"""Ingest text into the store: chunk → embed → save."""
from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

from mnemonics.store import Store

_encoder: Any = None
_encoder_name: str = "all-MiniLM-L6-v2"


# Preference / memory / concern phrase patterns. When ingest's
# augment_preferences=True, each input text is scanned with these regexes;
# matches become a synthetic "User has mentioned: ..." chunk that the
# retriever can hit when the question's wording is paraphrastic and the raw
# session text alone fails to match.
_PREF_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"i(?:'ve been| have been) having (?:trouble|issues?|problems?) with ([^,\.!?]{5,80})",
        r"i(?:'ve been| have been) feeling ([^,\.!?]{5,60})",
        r"i(?:'ve been| have been) (?:struggling|dealing) with ([^,\.!?]{5,80})",
        r"i(?:'ve been| have been) (?:worried|concerned) about ([^,\.!?]{5,80})",
        r"i(?:'m| am) (?:worried|concerned) about ([^,\.!?]{5,80})",
        r"i prefer ([^,\.!?]{5,60})",
        r"i usually ([^,\.!?]{5,60})",
        r"i(?:'ve been| have been) (?:trying|attempting) to ([^,\.!?]{5,80})",
        r"i(?:'ve been| have been) (?:considering|thinking about) ([^,\.!?]{5,80})",
        r"lately[,\s]+(?:i've been|i have been|i'm|i am) ([^,\.!?]{5,80})",
        r"recently[,\s]+(?:i've been|i have been|i'm|i am) ([^,\.!?]{5,80})",
        r"i(?:'ve been| have been) (?:working on|focused on|interested in) ([^,\.!?]{5,80})",
        r"i want to ([^,\.!?]{5,60})",
        r"i(?:'m| am) looking (?:to|for) ([^,\.!?]{5,60})",
        r"i(?:'m| am) thinking (?:about|of) ([^,\.!?]{5,60})",
        r"i(?:'ve been| have been) (?:noticing|experiencing) ([^,\.!?]{5,80})",
        r"i (?:still )?remember (?:the |my )?([^,\.!?]{5,80})",
        r"i used to ([^,\.!?]{5,60})",
        r"when i was (?:in high school|in college|young|a kid|growing up)[,\s]+([^,\.!?]{5,80})",
        r"growing up[,\s]+([^,\.!?]{5,80})",
    )
]

# Preserve any leading "TAG=value|" pseudo-namespace prefix the caller used on
# the input (e.g. "SID=abc|...") so synthetic preference docs still carry the
# same identifier and downstream code that parses out a session id sees a
# match. No-op when no prefix is present.
_INPUT_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z_0-9]*=[^|\s]+\|)")


def extract_preferences(text: str) -> list[str]:
    """Return distinct preference / memory / concern mentions for synth indexing.

    Caps at 12 to keep the synth doc tight. Empty list when nothing fires.
    """
    seen: set[str] = set()
    out: list[str] = []
    for pat in _PREF_PATTERNS:
        for m in pat.finditer(text):
            if not m.groups():
                continue
            clean = m.group(1).strip().rstrip(".,;!? ")
            if 5 <= len(clean) <= 80 and clean.lower() not in seen:
                seen.add(clean.lower())
                out.append(clean)
                if len(out) >= 12:
                    return out
    return out


def _build_preference_doc(text: str, prefs: list[str]) -> str:
    m = _INPUT_PREFIX_RE.match(text)
    prefix = m.group(1) if m else ""
    return f"{prefix}User has mentioned: " + "; ".join(prefs)


def _resolve_model(model: str) -> str:
    # MNEMONICS_ADAPTMEM_PATH swaps the default base encoder with a fine-tuned
    # adaptmem checkpoint at runtime. Same 384-dim contract, drop-in.
    adaptmem_path = os.environ.get("MNEMONICS_ADAPTMEM_PATH")
    if adaptmem_path and model == "all-MiniLM-L6-v2":
        return adaptmem_path
    return model


def _get_encoder(model: str = _encoder_name) -> Any:
    global _encoder, _encoder_name
    resolved = _resolve_model(model)
    if _encoder is None or resolved != _encoder_name:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(resolved)
        _encoder_name = resolved
    return _encoder


def _chunk(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


def ingest(
    texts: list[str],
    store: Store,
    ns: str = "default",
    meta: list[dict] | None = None,
    summaries: list[str | None] | None = None,
    model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 200,
    chunk_overlap: int = 40,
    augment_preferences: bool = False,
) -> int:
    """Chunk, embed and store texts. Returns total chunks stored.

    `summaries`, when provided, holds one optional summary per input text.
    Every chunk derived from a given input inherits that text's summary, so
    BM25 over the FTS mirror can find a chunk either via its raw words or
    via the higher-level gist. Vector embeddings stay over the raw chunk —
    the summary is a parallel keyword surface, not a replacement.

    When `augment_preferences=True`, each input text is scanned for
    preference/memory/concern phrasings ("I prefer X", "I remember Y",
    "growing up Z"). Matches collapse into one synthetic
    "User has mentioned: ..." chunk that is indexed alongside the raw
    chunks. The synth chunk carries meta["kind"] = "preference" so
    callers can tell augmented rows apart from raw ones.
    """
    # Guard: a single string is iterable in Python, would be ingested char-by-char.
    if isinstance(texts, str):
        texts = [texts]
    if summaries is not None and len(summaries) != len(texts):
        raise ValueError("summaries length must match texts length")
    enc = _get_encoder(model)
    all_chunks: list[str] = []
    all_meta: list[dict] = []
    all_summaries: list[str | None] = []

    for i, text in enumerate(texts):
        chunks = _chunk(text, chunk_size, chunk_overlap)
        m = (meta[i] if meta else {}) | {"source_idx": i}
        summary = summaries[i] if summaries is not None else None
        all_chunks.extend(chunks)
        all_meta.extend([m] * len(chunks))
        all_summaries.extend([summary] * len(chunks))
        if augment_preferences:
            prefs = extract_preferences(text)
            if prefs:
                all_chunks.append(_build_preference_doc(text, prefs))
                all_meta.append(m | {"kind": "preference"})
                all_summaries.append(summary)

    if not all_chunks:
        return 0

    vecs = enc.encode(all_chunks, batch_size=64, show_progress_bar=False,
                      normalize_embeddings=True, convert_to_numpy=True)
    store.add(all_chunks, vecs, ns=ns, meta=all_meta, summaries=all_summaries)
    return len(all_chunks)
