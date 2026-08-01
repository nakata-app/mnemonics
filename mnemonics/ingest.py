"""Ingest text into the store: chunk → embed → save."""
from __future__ import annotations

import os
import re
from typing import Any

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
        # Generic past-tense action + object: "I bought a new sofa", "I repainted my walls a lighter gray"
        r"i (?:just |recently |finally |already )?(?:bought|purchased|got|received|earned|completed|finished|made|painted|repainted|baked|cooked|tried|visited|moved to|started|joined|left|quit|sold|wrote|installed|fixed|lost|found|met|adopted) ([^,\.!?]{5,80})",
        # Receive-from-person: "my sister gave me a stand mixer", "Dad sent me a card"
        r"(?:my |our )?(?:mom|mother|dad|father|sister|brother|wife|husband|partner|son|daughter|friend|aunt|uncle|cousin|grandma|grandpa|boss|colleague|coworker|neighbor|teacher|[A-Z][a-z]+) (?:gave|sent|brought|gifted|got) me ([^,\.!?]{5,80})",
        # Occasion gifts: "for my birthday I got a new bike"
        r"for (?:my |our )?(?:birthday|christmas|anniversary|wedding|graduation|housewarming)[,\s]+([^,\.!?]{5,80})",
        # Attendance: "I attended a production of The Glass Menagerie", "I went to see X"
        r"i (?:just |recently )?attended ([^,\.!?]{5,80})",
        r"i (?:just |recently )?went to (?:see |watch |attend )?([^,\.!?]{5,80})",
        # My favorite X (preserves "favorite" context): "my favorite Japanese short-grain rice"
        r"(?:my|our) (favorite [^,\.!?]{5,50})",
        # Event venue: "concert at the Xfinity Center", "wedding at the Grand Ballroom"
        r"(?:concert|show|wedding|ceremony|event|performance|game|match|play) (?:was )?at (?:the )?([^,\.!?\n]{5,60})",
        # "it was at the X": follow-up location reference
        r"it was at (?:the )?([A-Z][^,\.!?\n]{3,55})",
        # Upgrade facts: "I upgraded to 500 Mbps"
        r"i upgraded (?:\w+ )?to ([^,\.!?]{5,60})",
        # Platform usage: "on Spotify lately", "on Netflix recently"
        r"on ([A-Z][a-zA-Z0-9+]{2,20}) (?:lately|recently|a lot|all the time)",
        # Study / academic location: "study abroad program at the University of Melbourne"
        r"study abroad (?:program )?at (?:the )?([^,\.!?\n]{5,80})",
        # Positive experiences: "I love X", "I enjoy X", "I always go for X"
        r"i (?:love|enjoy|adore) ([^,\.!?]{5,60})",
        r"i always (?:go for|choose|pick|prefer|get) ([^,\.!?]{5,60})",
    )
]

# Preserve any leading "TAG=value|" pseudo-namespace prefix the caller used on
# the input (e.g. "SID=abc|...") so synthetic preference docs still carry the
# same identifier and downstream code that parses out a session id sees a
# match. No-op when no prefix is present.
_INPUT_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z_0-9]*=[^|\s]+\|)")


# Numeric and declarative fact patterns. When ingest's
# augment_assistant_facts=True, the assistant-role turns in a session are
# scanned for these; the surrounding ~60-char snippet becomes part of a
# synthetic "Key facts: ..." chunk. Questions like "what % were women?" or
# "how much did I save on the train?" embed near the numeric snippet even
# when the raw chunk gets buried in a long session.
_FACT_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        # Percentages: "20%", "10-20%", "20 percent"
        r"\d+(?:[-–]\d+)?(?:\.\d+)?\s*(?:%|percent\b)",
        # Dollar amounts: "$120", "$1,234.56", "$80-$100"
        r"\$\s?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?[-–]\s?\$?\d+(?:,\d{3})*(?:\.\d+)?)?",
        # Quantities with units (durations, distances, weights, calories)
        r"\b\d+(?:,\d{3})*(?:[-–]\d+(?:,\d{3})*)?(?:\.\d+)?\s*(?:hours?|hrs?|minutes?|mins?|seconds?|days?|weeks?|months?|years?|miles?|km|kilometers?|kg|kilograms?|lbs?|pounds?|grams?|mg|liters?|gallons?|gal|calories?|kcal|°[CF]|degrees?)\b",
        # Second-person assistant-confirmation: "you saved $50", "you spent 2 hours"
        # Restricted to verbs that imply a stated past/computed value (not "you need to").
        r"\byou\s+(?:saved|spent|paid|earned|owe|received|got|made|gained|lost|traveled|covered|burned)\s+[^.!?\n]{2,60}",
        # "Your X is/was Y" / "The X is Y" — narrow metric-keyword set
        r"\b(?:your|the)\s+(?:total|result|answer|sum|average|count|distance|cost|price|fee|amount|duration|score|rate|ratio|share|salary|income|budget)\s+(?:of\s+[^.!?\n]{1,30}\s+)?(?:is|was|comes? to|equals?|amounts? to)\s+[^.!?\n]{2,40}",
    )
]


# Role marker as emitted by callers that flatten conversations into a single
# string ("[user] ...\n[assistant] ...\n"). We split on this when scanning
# for assistant-only facts; absent any markers we fall back to scanning the
# whole text.
_ROLE_LINE_RE = re.compile(r"^\[(user|assistant|system|tool)\]\s?", re.MULTILINE)


def extract_preferences(text: str) -> list[str]:
    """Return distinct preference / memory / concern mentions for synth indexing.

    Caps at 12 to keep the synth doc tight. Empty list when nothing fires.
    """
    seen: set[str] = set()
    out: list[str] = []
    for pat in _PREF_PATTERNS:
        for m in pat.finditer(text):
            if not m.groups():  # pragma: no cover — all _PREF_PATTERNS have capture groups
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


def _assistant_segments(text: str) -> list[str]:
    """Return assistant-role content segments from a role-tagged transcript.

    Falls back to ``[text]`` when no role markers are present so legacy
    callers (raw notes, single-author docs) still get scanned.
    """
    if not _ROLE_LINE_RE.search(text):
        return [text]
    parts: list[str] = []
    cur_role: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = _ROLE_LINE_RE.match(line)
        if m:
            if cur_role == "assistant" and buf:
                parts.append("\n".join(buf).strip())
            cur_role = m.group(1)
            buf = [line[m.end():]]
        else:
            buf.append(line)
    if cur_role == "assistant" and buf:
        parts.append("\n".join(buf).strip())
    return [p for p in parts if p]


def extract_facts(text: str) -> list[str]:
    """Return distinct numeric/declarative fact snippets from assistant turns.

    Match spans that overlap or sit within 5 chars of each other are merged
    before snipping, so "70°F to 80°F" or "$80 (was $100)" yield one snippet
    rather than two near-duplicates. Each snippet is bounded to ~80 chars of
    surrounding context so the embedding keeps the numeric token next to
    its referent (currency name, what was being measured). Caps at 12 to
    keep the synth chunk tight.
    """
    seen: set[str] = set()
    out: list[str] = []
    for seg in _assistant_segments(text):
        spans: list[tuple[int, int]] = []
        for pat in _FACT_PATTERNS:
            for m in pat.finditer(seg):
                spans.append((m.start(), m.end()))
        if not spans:
            continue
        spans.sort()
        # Merge overlapping or near-overlapping (gap <= 5 chars) spans.
        merged: list[tuple[int, int]] = []
        for s, e in spans:
            if merged and s <= merged[-1][1] + 5:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        for s, e in merged:
            lo = max(0, s - 25)
            hi = min(len(seg), e + 25)
            snip = seg[lo:hi].strip()
            snip = re.sub(r"\s+", " ", snip).rstrip(".,;!? ")
            if 8 <= len(snip) <= 140 and snip.lower() not in seen:
                seen.add(snip.lower())
                out.append(snip)
                if len(out) >= 12:
                    return out
    return out


def _build_fact_doc(text: str, facts: list[str]) -> str:
    m = _INPUT_PREFIX_RE.match(text)
    prefix = m.group(1) if m else ""
    return f"{prefix}Key facts: " + "; ".join(facts)


def _resolve_model(model: str) -> str:
    # MNEMONICS_ENCODER_MODEL fully overrides the base encoder; takes precedence
    # over adaptmem. Use it to A/B stronger sentence-transformers (e.g.
    # BAAI/bge-base-en-v1.5) without code changes. Different dim than 384 is
    # OK — the store holds whatever the encoder emits.
    env_model = os.environ.get("MNEMONICS_ENCODER_MODEL")
    if env_model and model == "all-MiniLM-L6-v2":
        return env_model
    # MNEMONICS_ADAPTMEM_PATH swaps the default base encoder with a fine-tuned
    # adaptmem checkpoint at runtime. Dim may differ from 384 (e.g. BGE-large
    # uses 1024); set MNEMONICS_DIM accordingly or let ingest auto-detect.
    adaptmem_path = os.environ.get("MNEMONICS_ADAPTMEM_PATH")
    if adaptmem_path and model == "all-MiniLM-L6-v2":
        return adaptmem_path
    return model


class _OnnxEmbeddingEncoder:
    """encode()-compatible ONNX encoder for the fine-tuned AdaptMem checkpoint.

    Uses the exported graph (BERT + mean pooling + L2 normalize; see
    scripts/export_adaptmem_onnx.py) with the Rust tokenizer. Vectors are
    bit-identical to the torch path (cosine 1.0, maxdiff ~2e-7) at ~110MB RSS
    instead of ~590MB (torch + SentenceTransformer measured before this swap).
    """

    def __init__(self, model_dir: str):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._session = ort.InferenceSession(
            model_dir.rstrip("/") + ".onnx", providers=["CPUExecutionProvider"]
        )
        self._tok = Tokenizer.from_file(model_dir.rstrip("/") + "/tokenizer.json")
        self._tok.enable_truncation(max_length=512)
        self._tok.enable_padding(
            pad_id=self._tok.token_to_id("[PAD]"), pad_token="[PAD]"
        )

    def encode(
        self,
        texts,
        batch_size: int = 256,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        **kwargs,
    ):
        import numpy as np

        texts = list(texts)
        if not texts:
            return np.zeros((0, 384), dtype="float32")
        outs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            encs = self._tok.encode_batch(chunk)
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            outs.append(
                self._session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
            )
        return np.concatenate(outs, axis=0).astype("float32")


class _FastEmbedEncoder:
    """encode()-compatible ONNX encoder for stock models (fastembed, torch-free)."""

    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._emb = TextEmbedding(model_name=model_name)

    def encode(
        self,
        texts,
        batch_size: int = 256,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        **kwargs,
    ):
        import numpy as np

        vecs = list(self._emb.embed(list(texts), batch_size=batch_size))
        return np.asarray(vecs, dtype="float32")


def _build_encoder(resolved: str) -> Any:
    # Fine-tuned AdaptMem checkpoint: prefer the exported ONNX graph (torch-free,
    # ~110MB RSS). Fall back to torch if the export is missing.
    if os.path.isdir(resolved):
        onnx_path = resolved.rstrip("/") + ".onnx"
        if os.path.exists(onnx_path):
            return _OnnxEmbeddingEncoder(resolved)
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(resolved)
    # Stock model (MNEMONICS_ENCODER_MODEL or default): fastembed, torch-free.
    return _FastEmbedEncoder(resolved)


def _get_encoder(model: str = _encoder_name) -> Any:
    global _encoder, _encoder_name
    resolved = _resolve_model(model)
    if _encoder is None or resolved != _encoder_name:
        _encoder = _build_encoder(resolved)
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
    augment_assistant_facts: bool = False,
    tier: int = 1,
    return_ids: bool = False,
) -> int | list[int]:
    """Chunk, embed and store texts. Returns total chunks stored.

    When ``return_ids=True``, returns the list of stored row ids instead of the
    count. Needed by callers (e.g. reconcile_ingest) that must reference the
    freshly written rows to record a supersede link.

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
        if augment_assistant_facts:
            facts = extract_facts(text)
            if facts:
                all_chunks.append(_build_fact_doc(text, facts))
                all_meta.append(m | {"kind": "fact"})
                all_summaries.append(summary)

    if not all_chunks:
        return [] if return_ids else 0

    vecs = enc.encode(all_chunks, batch_size=64, show_progress_bar=False,
                      normalize_embeddings=True, convert_to_numpy=True)
    ids = store.add(all_chunks, vecs, ns=ns, meta=all_meta, summaries=all_summaries, tier=tier)
    # Version stamp: bu vektorleri hangi encoder gomdu, kalici isaretle. Encoder
    # ileride degisip re-embed atlanirsa retrieve drift'i fark eder.
    try:
        from mnemonics import embed_manifest as _em
        _em.write(store.root, _em.encoder_fingerprint(_resolve_model(model), store.dim))
    except Exception:
        pass
    return ids if return_ids else len(all_chunks)


def reembed_all(store, model: str = _encoder_name, batch_size: int = 256) -> list[dict]:
    """Re-encode the whole store with the (possibly swapped) encoder and rebuild
    every namespace index. Run this after dropping a new AdaptMem checkpoint in
    via MNEMONICS_ADAPTMEM_PATH: stored vectors are stale in the new model's
    space until recomputed. Re-stamps the encoder fingerprint so retrieve() stops
    flagging drift. Returns per-namespace ``{ns, n}``.
    """
    enc = _get_encoder(model)
    resolved = _resolve_model(model)

    def _encode(texts):
        return enc.encode(texts, batch_size=batch_size, show_progress_bar=False,
                          normalize_embeddings=True, convert_to_numpy=True)

    results = []
    for ns in store.list_namespaces():
        n = store.reembed_ns(ns, _encode, batch_size=batch_size)
        if n:
            results.append({"ns": ns, "n": n})
    try:
        from mnemonics import embed_manifest as _em
        _em.write(store.root, _em.encoder_fingerprint(resolved, store.dim))
    except Exception:
        pass
    return results
