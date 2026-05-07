"""Ingest text into the store: chunk → embed → save."""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from mnemonics.store import Store

_encoder: Any = None
_encoder_name: str = "all-MiniLM-L6-v2"


def _get_encoder(model: str = _encoder_name) -> Any:
    global _encoder, _encoder_name
    if _encoder is None or model != _encoder_name:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(model)
        _encoder_name = model
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
    model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 200,
    chunk_overlap: int = 40,
) -> int:
    """Chunk, embed and store texts. Returns total chunks stored."""
    # Guard: a single string is iterable in Python, would be ingested char-by-char.
    if isinstance(texts, str):
        texts = [texts]
    enc = _get_encoder(model)
    all_chunks: list[str] = []
    all_meta: list[dict] = []

    for i, text in enumerate(texts):
        chunks = _chunk(text, chunk_size, chunk_overlap)
        m = (meta[i] if meta else {}) | {"source_idx": i}
        all_chunks.extend(chunks)
        all_meta.extend([m] * len(chunks))

    if not all_chunks:
        return 0

    vecs = enc.encode(all_chunks, batch_size=64, show_progress_bar=False,
                      normalize_embeddings=True, convert_to_numpy=True)
    store.add(all_chunks, vecs, ns=ns, meta=all_meta)
    return len(all_chunks)
