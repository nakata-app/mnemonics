"""Ingest text into the store: chunk → embed → save."""
from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

from mnemonics.store import Store

_encoder: Any = None
_encoder_name: str = "all-MiniLM-L6-v2"


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
) -> int:
    """Chunk, embed and store texts. Returns total chunks stored.

    `summaries`, when provided, holds one optional summary per input text.
    Every chunk derived from a given input inherits that text's summary, so
    BM25 over the FTS mirror can find a chunk either via its raw words or
    via the higher-level gist. Vector embeddings stay over the raw chunk —
    the summary is a parallel keyword surface, not a replacement.
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

    if not all_chunks:
        return 0

    vecs = enc.encode(all_chunks, batch_size=64, show_progress_bar=False,
                      normalize_embeddings=True, convert_to_numpy=True)
    store.add(all_chunks, vecs, ns=ns, meta=all_meta, summaries=all_summaries)
    return len(all_chunks)
