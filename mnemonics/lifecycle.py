"""Deterministic memory lifecycle operations.

Canonical slots are caller-defined fact identities. They provide a safe way to
represent changing facts without asking an embedding similarity threshold to
decide whether two statements contradict each other.
"""
from __future__ import annotations

from typing import Any

from mnemonics.ingest import _get_encoder, _resolve_model_for_store
from mnemonics.store import Store


def canonical_ingest(
    text: str,
    store: Store,
    *,
    canonical_key: str,
    ns: str = "default",
    summary: str | None = None,
    meta: dict[str, Any] | None = None,
    tier: int = 1,
    model: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Upsert a canonical fact and archive any prior active value atomically."""
    if not text.strip():
        raise ValueError("text must not be empty")
    resolved = _resolve_model_for_store(model, store)
    encoder = _get_encoder(resolved)
    vector = encoder.encode(
        [text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    result = store.add_canonical(
        text,
        vector,
        ns=ns,
        canonical_key=canonical_key,
        summary=summary,
        meta=meta,
        tier=tier,
    )
    return {
        **result,
        "canonical_key": canonical_key.strip(),
        "ns": ns,
        "encoder": resolved,
    }