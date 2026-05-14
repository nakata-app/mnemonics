"""Sync between two mnemonics stores via portable archive files.

The on-disk archive format is the same .tar.gz backup() produces, plus a
small `manifest.json` describing each row in a transport-friendly way:

    {
      "version": 1,
      "exported_at": "<iso8601>",
      "rows": [
        {"hash": "<sha256(text)>", "ns": "...", "text": "...",
         "meta": {...}, "tier": 1, "created": "..."},
        ...
      ]
    }

Import re-embeds the texts on the target so the local encoder choice
(adaptmem vs base) governs the resulting vectors. Conflict policy is
explicit at the call site — there's no silent merge.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Strategy = Literal["skip-existing", "force-new-id", "overwrite"]

_MANIFEST_NAME = "manifest.json"
_MANIFEST_VERSION = 1


def _row_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def export_store(
    store_path: str | Path = "~/.mnemonics",
    out: str | Path | None = None,
) -> Path:
    """Write a transport archive: every memory row plus its content hash.

    Index files are NOT included — the target re-embeds with its own
    encoder. This keeps the archive small and side-steps the cross-machine
    encoder-mismatch problem entirely.
    """
    src = Path(store_path).expanduser()
    if not (src / "memories.db").is_file():
        raise FileNotFoundError(f"no memories.db at {src}")

    out_path = (
        Path(out).expanduser() if out
        else Path.home() / ".mnemonics-sync" / (
            datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".sync.tar.gz"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(str(src / "memories.db"))
    rows = db.execute(
        "SELECT id, ns, text, meta, created, tier FROM memories ORDER BY id"
    ).fetchall()
    db.close()

    manifest = {
        "version": _MANIFEST_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "rows": [
            {
                "hash": _row_hash(r[2]),
                "ns": r[1],
                "text": r[2],
                "meta": json.loads(r[3]) if r[3] else {},
                "created": r[4],
                "tier": r[5],
            }
            for r in rows
        ],
    }

    with tarfile.open(out_path, "w:gz") as tf:
        data = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name=_MANIFEST_NAME)
        info.size = len(data)
        import io
        tf.addfile(info, io.BytesIO(data))
    return out_path


def _read_manifest(archive: Path) -> dict[str, Any]:
    with tarfile.open(archive, "r:gz") as tf:
        member = tf.getmember(_MANIFEST_NAME)
        f = tf.extractfile(member)
        if f is None:
            raise ValueError(f"manifest unreadable in {archive}")
        return json.loads(f.read().decode("utf-8"))


def import_store(
    archive: str | Path,
    store_path: str | Path = "~/.mnemonics",
    strategy: Strategy = "skip-existing",
    only_ns: str | None = None,
) -> dict[str, int]:
    """Merge a sync archive into the target store.

    Returns a summary dict: {"imported": N, "skipped": M, "overwritten": K}.
    """
    arc = Path(archive).expanduser()
    if not arc.is_file():
        raise FileNotFoundError(f"archive not found: {arc}")
    if strategy not in ("skip-existing", "force-new-id", "overwrite"):
        raise ValueError(f"unknown strategy: {strategy}")

    manifest = _read_manifest(arc)
    if manifest.get("version") != _MANIFEST_VERSION:
        raise ValueError(
            f"unsupported manifest version: {manifest.get('version')}"
        )

    # Import lazily to avoid loading sentence-transformers when callers only
    # want to inspect the archive (export-only paths).
    from mnemonics.ingest import _get_encoder
    from mnemonics.store import Store

    store = Store(path=store_path)
    encoder = _get_encoder()

    # Pre-index existing text hashes per namespace for skip-existing.
    existing_hashes: dict[str, dict[str, int]] = {}
    if strategy in ("skip-existing", "overwrite"):
        with store._lock:
            cur = store._db.execute("SELECT id, ns, text FROM memories")
            for rid, ns, text in cur.fetchall():
                existing_hashes.setdefault(ns, {})[_row_hash(text)] = rid

    summary = {"imported": 0, "skipped": 0, "overwritten": 0}
    # Group by ns for batched encoding and one save_index() per ns.
    by_ns: dict[str, list[dict[str, Any]]] = {}
    for row in manifest["rows"]:
        if only_ns is not None and row["ns"] != only_ns:
            continue
        by_ns.setdefault(row["ns"], []).append(row)

    for ns, rows in by_ns.items():
        to_insert: list[dict[str, Any]] = []
        for row in rows:
            h = row["hash"]
            existing_id = existing_hashes.get(ns, {}).get(h)
            if existing_id is not None:
                if strategy == "skip-existing":
                    summary["skipped"] += 1
                    continue
                if strategy == "overwrite":
                    # Overwrite path: drop the old row first; the new vector
                    # will replace it below via store.add() with a new id.
                    with store._lock:
                        store._db.execute(
                            "DELETE FROM memories WHERE id=?", (existing_id,)
                        )
                        store._db.commit()
                    summary["overwritten"] += 1
                # force-new-id falls through and always inserts.
            to_insert.append(row)

        if not to_insert:
            continue

        texts = [r["text"] for r in to_insert]
        metas = [r["meta"] for r in to_insert]
        vecs = encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        store.add(texts=texts, vectors=vecs, ns=ns, meta=metas)
        summary["imported"] += len(texts)

    return summary
