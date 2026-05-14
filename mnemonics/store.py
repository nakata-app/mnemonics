"""Persistent storage: SQLite (metadata) + hnswlib (vectors)."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import hnswlib
import numpy as np


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ns            TEXT NOT NULL DEFAULT 'default',
    text          TEXT NOT NULL,
    meta          TEXT NOT NULL DEFAULT '{}',
    created       TEXT NOT NULL DEFAULT (datetime('now')),
    tier          INTEGER NOT NULL DEFAULT 1,
    last_accessed TEXT,
    access_count  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ns ON memories(ns);

-- FTS5 contentless-mirror over `memories`, keyed on `id` (used for BM25 lookup
-- in hybrid retrieval). Triggers keep it in sync; older DBs are backfilled
-- by Store._migrate_fts().
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text,
    content='memories',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
"""

# Tier semantics:
#   0 = pinned   (no decay, retained forever, manual)
#   1 = default  (slow decay)
#   2 = ambient  (fast decay)

DIM = 384  # all-MiniLM-L6-v2 default dim


class Store:
    """Thread-safe memory store backed by SQLite + hnswlib."""

    def __init__(self, path: str | Path = "~/.mnemonics", dim: int = DIM):
        self.root = Path(path).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.root / "memories.db"), check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._migrate()
        self._migrate_fts()
        self._db.commit()
        self._index: dict[str, hnswlib.Index] = {}

    def _migrate(self) -> None:
        """Idempotent column additions for older DBs created before the schema bump."""
        cols = {row[1] for row in self._db.execute("PRAGMA table_info(memories)").fetchall()}
        if "tier" not in cols:
            self._db.execute("ALTER TABLE memories ADD COLUMN tier INTEGER NOT NULL DEFAULT 1")
        if "last_accessed" not in cols:
            self._db.execute("ALTER TABLE memories ADD COLUMN last_accessed TEXT")
        if "access_count" not in cols:
            self._db.execute("ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0")
        # idx_tier must come after migration so the column exists.
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_tier ON memories(tier)")

    def _migrate_fts(self) -> None:
        """Backfill the FTS5 mirror for DBs that existed before hybrid search.

        Triggers keep new rows in sync, but pre-existing rows aren't indexed
        until we explicitly rebuild. Cheap when already populated.
        """
        fts_count = self._db.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
        mem_count = self._db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if fts_count < mem_count:
            self._db.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")

    def _index_for(self, ns: str) -> hnswlib.Index:
        if ns not in self._index:
            idx_path = self.root / f"index_{ns}.bin"
            idx = hnswlib.Index(space="cosine", dim=self.dim)
            if idx_path.exists():
                idx.load_index(str(idx_path))
                idx.set_ef(64)
            else:
                idx.init_index(max_elements=100_000, ef_construction=200, M=16)
                idx.set_ef(64)
            self._index[ns] = idx
        return self._index[ns]

    def add(self, texts: list[str], vectors: np.ndarray, ns: str = "default", meta: list[dict] | None = None) -> list[int]:
        if meta is None:
            meta = [{} for _ in texts]
        with self._lock:
            ids = []
            for text, m in zip(texts, meta):
                cur = self._db.execute(
                    "INSERT INTO memories (ns, text, meta) VALUES (?, ?, ?)",
                    (ns, text, json.dumps(m)),
                )
                ids.append(cur.lastrowid)
            self._db.commit()
            idx = self._index_for(ns)
            idx.add_items(vectors, ids)
            idx.save_index(str(self.root / f"index_{ns}.bin"))
        return ids

    def search(self, vector: np.ndarray, ns: str = "default", top_k: int = 5) -> list[dict[str, Any]]:
        with self._lock:
            idx = self._index_for(ns)
            n = min(top_k, idx.get_current_count())
            if n == 0:
                return []
            labels, distances = idx.knn_query(vector, k=n)
            row_ids = [int(x) for x in labels[0]]
            placeholders = ",".join("?" * len(row_ids))
            rows = self._db.execute(
                f"SELECT id, text, meta, created, tier, last_accessed, access_count "
                f"FROM memories WHERE id IN ({placeholders})",
                row_ids,
            ).fetchall()
            by_id = {r[0]: r for r in rows}
            results = []
            for rid, dist in zip(labels[0], distances[0]):
                row = by_id.get(int(rid))
                if row is None:
                    continue
                results.append({
                    "id": row[0],
                    "text": row[1],
                    "meta": json.loads(row[2]),
                    "created": row[3],
                    "tier": row[4],
                    "last_accessed": row[5],
                    "access_count": row[6],
                    "score": float(1 - dist),
                })
            # Touch retrieved rows: bump access_count, update last_accessed.
            if results:
                touched_ids = [r["id"] for r in results]
                touch_placeholders = ",".join("?" * len(touched_ids))
                self._db.execute(
                    f"UPDATE memories SET last_accessed = datetime('now'), "
                    f"access_count = access_count + 1 WHERE id IN ({touch_placeholders})",
                    touched_ids,
                )
                self._db.commit()
        return results

    # FTS5's MATCH grammar treats bare punctuation as syntax errors. We only
    # need word-level recall, so flatten the query to alphanumerics + space and
    # OR the surviving tokens together (subset match, not phrase).
    @staticmethod
    def _fts_sanitize(query: str) -> str:
        import re as _re
        tokens = _re.findall(r"\w+", query, flags=_re.UNICODE)
        # Quote each token so FTS5 treats it as a literal term (handles digits,
        # underscores, mixed-case identifiers like "PR1490" or "v0_2_1").
        return " OR ".join(f'"{t}"' for t in tokens if t)

    def search_bm25(self, query: str, ns: str = "default", top_k: int = 20) -> list[dict[str, Any]]:
        """BM25 keyword search via SQLite FTS5. Returns rows ordered best-first.

        score is negated so that higher = better, matching the vector path.
        Empty / punctuation-only queries return [].
        """
        match = self._fts_sanitize(query)
        if not match:
            return []
        with self._lock:
            try:
                rows = self._db.execute(
                    "SELECT m.id, m.text, m.meta, m.created, m.tier, m.last_accessed, "
                    "m.access_count, bm25(memories_fts) AS bm25_raw "
                    "FROM memories_fts "
                    "JOIN memories m ON m.id = memories_fts.rowid "
                    "WHERE memories_fts MATCH ? AND m.ns = ? "
                    "ORDER BY bm25_raw LIMIT ?",
                    (match, ns, top_k),
                ).fetchall()
            except sqlite3.OperationalError:
                # Malformed MATCH expression (rare; sanitizer should prevent it).
                return []
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "text": row[1],
                "meta": json.loads(row[2]),
                "created": row[3],
                "tier": row[4],
                "last_accessed": row[5],
                "access_count": row[6],
                "score": -float(row[7]),  # lower bm25 = better → negate to align with vector
            })
        return results

    def set_tier(self, memory_id: int, tier: int) -> bool:
        if tier not in (0, 1, 2):
            raise ValueError("tier must be 0 (pinned), 1 (default), or 2 (ambient)")
        with self._lock:
            cur = self._db.execute("UPDATE memories SET tier=? WHERE id=?", (tier, memory_id))
            self._db.commit()
        return cur.rowcount > 0

    def pin(self, memory_id: int) -> bool:
        return self.set_tier(memory_id, 0)

    def list_namespaces(self) -> list[str]:
        rows = self._db.execute("SELECT DISTINCT ns FROM memories ORDER BY ns").fetchall()
        return [r[0] for r in rows]

    def count(self, ns: str = "default") -> int:
        row = self._db.execute("SELECT COUNT(*) FROM memories WHERE ns=?", (ns,)).fetchone()
        return row[0] if row else 0

    def delete(self, memory_id: int) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            self._db.commit()
        return cur.rowcount > 0

    def gc_candidates(self, ns: str | None = None, age_days: int = 30) -> list[dict[str, Any]]:
        """Rows safe to garbage-collect: tier 2 (ambient) older than `age_days`, never accessed."""
        sql = (
            "SELECT id, ns, substr(text, 1, 80) AS preview, "
            "CAST(julianday('now') - julianday(created) AS INTEGER) AS age_days "
            "FROM memories "
            "WHERE tier = 2 AND access_count = 0 "
            "AND julianday('now') - julianday(created) > ?"
        )
        params: list[Any] = [age_days]
        if ns is not None:
            sql += " AND ns = ?"
            params.append(ns)
        sql += " ORDER BY age_days DESC"
        rows = self._db.execute(sql, params).fetchall()
        return [{"id": r[0], "ns": r[1], "preview": r[2], "age_days": r[3]} for r in rows]

    def gc(self, ns: str | None = None, age_days: int = 30) -> int:
        """Delete the candidates returned by gc_candidates. Returns deleted count."""
        candidates = self.gc_candidates(ns=ns, age_days=age_days)
        if not candidates:
            return 0
        ids = [c["id"] for c in candidates]
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            self._db.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})",
                ids,
            )
            self._db.commit()
        return len(ids)
