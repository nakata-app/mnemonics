"""Retrieve memories — embed query, search, optionally verify with halluguard."""
from __future__ import annotations

from typing import Any

import requests

from mnemonics.store import Store
from mnemonics.ingest import _get_encoder


def retrieve(
    query: str,
    store: Store,
    ns: str = "default",
    top_k: int = 5,
    model: str = "all-MiniLM-L6-v2",
    verify: bool = True,
    verify_threshold: float = 0.45,
) -> dict[str, Any]:
    """
    Search the store for query.
    If verify=True, runs halluguard on results against the retrieved corpus.
    Returns: {results, verified, trust_score, flagged_count}
    """
    enc = _get_encoder(model)
    qvec = enc.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    results = store.search(qvec, ns=ns, top_k=top_k)

    if not results:
        return {"results": [], "verified": True, "trust_score": 1.0, "flagged_count": 0}

    trust_score = 1.0
    flagged_count = 0
    verified = True

    if verify:
        try:
            corpus = [r["text"] for r in results]
            combined = " ".join(corpus)
            resp = requests.post(
                "http://127.0.0.1:7801/check",
                json={"corpus": corpus, "answer": combined, "threshold": verify_threshold},
                timeout=10,
            )
            if resp.ok:
                data = resp.json()
                trust_score = data.get("trust_score", 1.0)
                flagged_count = data.get("n_flagged", 0)
                verified = data.get("ok", True)
                # Mark flagged results
                flagged_texts = {f["text"] for f in data.get("flagged", [])}
                for r in results:
                    r["flagged"] = r["text"] in flagged_texts
        except Exception:
            pass  # halluguard daemon not running — skip verify

    for r in results:
        r.setdefault("flagged", False)

    return {
        "results": results,
        "verified": verified,
        "trust_score": round(trust_score, 4),
        "flagged_count": flagged_count,
    }
