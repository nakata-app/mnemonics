"""Retrieve memories — embed query, search the index, return ranked results."""
from __future__ import annotations

from typing import Any

from mnemonics.store import Store
from mnemonics.ingest import _get_encoder


def retrieve(
    query: str,
    store: Store,
    ns: str = "default",
    top_k: int = 5,
    model: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Search the store for query. Returns: {results}."""
    enc = _get_encoder(model)
    qvec = enc.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    results = store.search(qvec, ns=ns, top_k=top_k)
    return {"results": results}
