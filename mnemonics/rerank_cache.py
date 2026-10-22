"""Persistent cache for reranker (query, document) scores.

Rationale: a reranker call scores every document against the query and the
scores are then thrown away.  Two consequences:

* the same document is rescored for every new query, and
* a re-run of an evaluation pays the whole cost again.

Storing the scores turns them into reusable experience (see arXiv 2608.22767,
which amortises reranker cost by treating LLM relevance scores as a matrix to
be completed).  The cache is keyed by model name and content hashes, so it is
safe across processes and never goes stale when a document changes.

Environment
-----------
``MNEMONICS_RERANK_CACHE``
    Path to the SQLite file.  Defaults to ``~/.mnemonics/rerank_scores.db``.
    Set it to ``off`` (or ``0`` / ``disabled``) to bypass the cache entirely.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from typing import Sequence

DEFAULT_PATH = Path.home() / ".mnemonics" / "rerank_scores.db"
_DISABLED = {"", "0", "off", "none", "disabled"}


def _path() -> Path | None:
    raw = os.environ.get("MNEMONICS_RERANK_CACHE")
    if raw is None:
        return DEFAULT_PATH
    if raw.strip().lower() in _DISABLED:
        return None
    return Path(raw).expanduser()


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS scores ("
        " model TEXT NOT NULL, qhash TEXT NOT NULL, dhash TEXT NOT NULL,"
        " score REAL NOT NULL, PRIMARY KEY (model, qhash, dhash))"
    )
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


def lookup(model: str, query: str, documents: Sequence[str]) -> dict[int, float]:
    """Return ``{position: score}`` for the documents already scored."""
    path = _path()
    if path is None or not len(documents):
        return {}
    qhash = _digest(query)
    hashes = [_digest(d) for d in documents]
    try:
        conn = _connect(path)
        try:
            marks = ",".join("?" * len(hashes))
            rows = conn.execute(
                f"SELECT dhash, score FROM scores"
                f" WHERE model = ? AND qhash = ? AND dhash IN ({marks})",
                [model, qhash, *hashes],
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return {}
    by_hash = dict(rows)
    return {i: by_hash[h] for i, h in enumerate(hashes) if h in by_hash}


def store(model: str, query: str, documents: Sequence[str], scores: Sequence[float]) -> None:
    """Persist the scores that were just computed."""
    path = _path()
    if path is None or not len(documents):
        return
    qhash = _digest(query)
    rows = [(_digest(d), float(s)) for d, s in zip(documents, scores)]
    try:
        conn = _connect(path)
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO scores (model, qhash, dhash, score)"
                " VALUES (?, ?, ?, ?)",
                [(model, qhash, h, s) for h, s in rows],
            )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error:
        pass


def stats() -> dict[str, int]:
    """Row count and distinct query count, for verification and reporting."""
    path = _path()
    if path is None or not path.exists():
        return {"rows": 0, "queries": 0}
    try:
        conn = _connect(path)
        try:
            rows = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
            queries = conn.execute(
                "SELECT COUNT(DISTINCT qhash) FROM scores"
            ).fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error:
        return {"rows": 0, "queries": 0}
    return {"rows": int(rows), "queries": int(queries)}
