"""Encoder version stamping for the vector store.

The HNSW indices (`index_<ns>.bin`) store vectors but carry no record of *which*
encoder produced them. `ingest._resolve_model()` silently swaps the encoder based
on `MNEMONICS_ENCODER_MODEL` / `MNEMONICS_ADAPTMEM_PATH`, so a freshly trained
checkpoint dropped in via env var leaves every stored vector in the *old* model's
coordinate space while queries are embedded in the *new* one. If the dim matches
(both 384) nothing even errors, retrieval quality just silently rots.

This module records an encoder fingerprint alongside the store and verifies the
live encoder against it. It is the gate that makes a re-embed / checkpoint swap
safe: promote a new encoder only after `write()` re-stamps the manifest, which by
contract happens together with a full re-embed.

Pure I/O + hashing, no sentence-transformers dependency. Safe to import anywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

MANIFEST_NAME = "embed_manifest.json"


def encoder_fingerprint(resolved_model: str, dim: int) -> dict[str, Any]:
    """Stable identity for the encoder that gomdu the current indices.

    Local checkpoint  -> hash of weights file (path + size + mtime).
    Hub model id      -> the id itself.

    size+mtime is cheap and changes on any retrain; we deliberately avoid hashing
    86MB of weights on every store open. The full content hash only matters at
    swap time, where `write()` is called explicitly.
    """
    p = Path(resolved_model).expanduser()
    if p.exists():
        weights = None
        for name in ("model.safetensors", "pytorch_model.bin"):
            cand = p / name if p.is_dir() else p
            if cand.exists() and cand.is_file():
                weights = cand
                break
        if weights is not None:
            st = weights.stat()
            sig = f"{weights.name}:{st.st_size}:{int(st.st_mtime)}"
            fp = hashlib.blake2b(sig.encode(), digest_size=16).hexdigest()
        else:
            fp = hashlib.blake2b(str(p).encode(), digest_size=16).hexdigest()
        kind = "local"
    else:
        fp = resolved_model
        kind = "hub"
    return {"encoder": resolved_model, "dim": int(dim), "fingerprint": fp, "kind": kind}


def _manifest_path(root: str | Path) -> Path:
    return Path(root).expanduser() / MANIFEST_NAME


def read(root: str | Path) -> dict[str, Any] | None:
    """Return the stamped manifest, or None if the store was never stamped."""
    path = _manifest_path(root)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write(root: str | Path, fingerprint: dict[str, Any]) -> None:
    """Atomically stamp the store with `fingerprint` (tmp file + rename)."""
    path = _manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(fingerprint, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def verify(root: str | Path, live: dict[str, Any]) -> tuple[str, str]:
    """Compare the live encoder fingerprint against the stamped one.

    Returns (status, reason):
      ("ok",       "")                      identities match, retrieval is sound
      ("missing",  ...)                      never stamped (legacy store) -> backfill
      ("mismatch", "dim 384 != 1024")        DIFFERENT dim, indices are unusable
      ("mismatch", "fingerprint ...")        SAME dim, different weights -> silent rot
    """
    stamped = read(root)
    if stamped is None:
        return ("missing", "store has no embed manifest")
    if int(stamped.get("dim", -1)) != int(live.get("dim", -2)):
        return ("mismatch", f"dim {stamped.get('dim')} != {live.get('dim')}")
    if stamped.get("fingerprint") != live.get("fingerprint"):
        return (
            "mismatch",
            f"encoder fingerprint changed: stamped {stamped.get('encoder')!r} "
            f"({stamped.get('fingerprint')}) vs live {live.get('encoder')!r} "
            f"({live.get('fingerprint')}) -- re-embed required",
        )
    return ("ok", "")
