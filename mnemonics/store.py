"""Persistent storage: SQLite (metadata) + hnswlib (vectors)."""
from __future__ import annotations

import contextlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import hnswlib
import numpy as np

from mnemonics import crypto

# Swap in sqlcipher3 only when the caller has opted into encryption. Keeping
# the stdlib path as default means existing plain-text DBs and the 155-test
# suite are untouched; opt-in is an explicit env-var flip.
if crypto.encryption_requested():  # pragma: no cover
    try:
        import sqlcipher3 as sqlite3  # type: ignore[no-redef]
    except ImportError as _e:
        raise RuntimeError(
            "MNEMONICS_ENCRYPT=1 is set but sqlcipher3 is not installed. "
            "Install with `pip install mnemonics[encrypt]` or set the env "
            "var back to 0 to fall back to plain SQLite."
        ) from _e
    _ENCRYPTED = True
else:
    import sqlite3
    _ENCRYPTED = False


def _apply_key(conn: Any) -> None:
    """Run ``PRAGMA key`` on a fresh sqlcipher3 connection.

    Uses raw-hex syntax (``PRAGMA key = "x'...'"``) which bypasses key
    derivation entirely — our keys are already cryptographically random
    256-bit values, so KDF would just add startup cost without raising
    the security bar. No-op when encryption is disabled.
    """
    if not _ENCRYPTED:
        return
    key_hex = crypto.require_key()
    # Defensive: only allow hex, otherwise the raw-key pragma would fail
    # opaquely. Surface the misconfiguration explicitly instead.
    if len(key_hex) != 64 or any(c not in "0123456789abcdefABCDEF" for c in key_hex):
        raise RuntimeError(
            "MNEMONICS_DB_KEY must be a 64-character hex string (256-bit). "
            f"Got length {len(key_hex)}."
        )
    conn.execute(f"PRAGMA key = \"x'{key_hex}'\"")


try:
    import fcntl  # POSIX advisory file locking — used for cross-process safety
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover — Windows path, not supported
    _HAS_FCNTL = False


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ns            TEXT NOT NULL DEFAULT 'default',
    text          TEXT NOT NULL,
    summary       TEXT,
    meta          TEXT NOT NULL DEFAULT '{}',
    created       TEXT NOT NULL DEFAULT (datetime('now')),
    tier          INTEGER NOT NULL DEFAULT 1,
    last_accessed TEXT,
    access_count  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ns ON memories(ns);

-- FTS5 contentless-mirror over `memories`, keyed on `id`. Indexes both raw
-- `text` and optional `summary` so BM25 can match either layer in hybrid
-- retrieval. Older DBs that still carry a single-column mirror are dropped
-- and rebuilt by Store._migrate_fts() so the schema stays self-healing.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text,
    summary,
    content='memories',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text, summary) VALUES (new.id, new.text, new.summary);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text, summary) VALUES('delete', old.id, old.text, old.summary);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text, summary) VALUES('delete', old.id, old.text, old.summary);
    INSERT INTO memories_fts(rowid, text, summary) VALUES (new.id, new.text, new.summary);
END;
"""

# Tier semantics:
#   0 = pinned   (no decay, retained forever, manual)
#   1 = default  (slow decay)
#   2 = ambient  (fast decay)

DIM = int(os.environ.get("MNEMONICS_DIM", "384"))


class Store:
    """Thread-safe memory store backed by SQLite + hnswlib."""

    def __init__(self, path: str | Path = "~/.mnemonics", dim: int = DIM):
        self.root = Path(path).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.root / "memories.db"), check_same_thread=False)
        _apply_key(self._db)
        # busy_timeout MUST come first: switching journal_mode takes a brief
        # RESERVED lock, and two peer processes opening the DB at the same
        # instant otherwise race and one gets `database is locked`.
        self._db.execute("PRAGMA busy_timeout=5000")
        # WAL is a persistent DB-file property — only one peer needs to flip
        # it, and the switch itself takes an EXCLUSIVE lock that bypasses
        # busy_timeout. Skip the PRAGMA if we're already in WAL to avoid
        # racing with concurrent openers on a fresh DB.
        current_mode = self._db.execute("PRAGMA journal_mode").fetchone()[0]
        if str(current_mode).lower() != "wal":
            try:
                self._db.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError:
                # A peer just won the race and is mid-switch. Their WAL takes
                # effect for us too; if not, the next Store() call will retry.
                pass
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)
        self._migrate()
        self._migrate_fts()
        self._db.commit()
        self._index: dict[str, hnswlib.Index] = {}
        # Mtime watermark per namespace; we use it to detect peer writes
        # since our last load and force a reload before we add anything.
        self._index_mtime: dict[str, float] = {}

    def _migrate(self) -> None:
        """Idempotent column additions for older DBs created before the schema bump."""
        cols = {row[1] for row in self._db.execute("PRAGMA table_info(memories)").fetchall()}
        if "tier" not in cols:
            self._db.execute("ALTER TABLE memories ADD COLUMN tier INTEGER NOT NULL DEFAULT 1")
        if "last_accessed" not in cols:
            self._db.execute("ALTER TABLE memories ADD COLUMN last_accessed TEXT")
        if "access_count" not in cols:
            self._db.execute("ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0")
        if "summary" not in cols:
            # NULL-default keeps the column reversible for older rows; ingest
            # writes summary only when the caller hands one in.
            self._db.execute("ALTER TABLE memories ADD COLUMN summary TEXT")
        # idx_tier must come after migration so the column exists.
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_tier ON memories(tier)")

    def _migrate_fts(self) -> None:
        """Heal the FTS5 mirror across schema generations.

        Two cases to cover:
          1. Pre-hybrid DBs: FTS table did not exist, triggers just created it,
             but the existing `memories` rows are not indexed yet.
          2. Pre-summary DBs: FTS table exists but only has the `text` column,
             so the new triggers that reference `summary` would fail. Drop
             and rebuild with the two-column schema; triggers re-fire on
             rebuild and pre-existing rows get re-indexed in one pass.

        We do not lean on FTS5's own ``'rebuild'`` command for case (2):
        empirically the first invocation after a drop/recreate writes the
        row payloads but skips index update under some SQLite builds, leaving
        rows present but unmatched. A direct INSERT into the contentless
        mirror is deterministic across versions.
        """
        fts_cols = [
            row[1]
            for row in self._db.execute("PRAGMA table_info(memories_fts)").fetchall()
        ]
        schema_was_replaced = False
        if "summary" not in fts_cols:
            # Drop the old single-column mirror and the triggers that wrote to
            # it; the schema script will recreate both with the new shape.
            self._db.execute("DROP TRIGGER IF EXISTS memories_ai")
            self._db.execute("DROP TRIGGER IF EXISTS memories_ad")
            self._db.execute("DROP TRIGGER IF EXISTS memories_au")
            self._db.execute("DROP TABLE IF EXISTS memories_fts")
            self._db.executescript(_SCHEMA)
            schema_was_replaced = True
        fts_count = self._db.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
        mem_count = self._db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if schema_was_replaced and mem_count > 0:
            # Explicit re-mirror: walk `memories` and insert text+summary into
            # the freshly created FTS table. Avoids the rebuild-skip bug.
            self._db.execute(
                "INSERT INTO memories_fts(rowid, text, summary) "
                "SELECT id, text, summary FROM memories"
            )
        elif fts_count < mem_count:  # pragma: no cover — FTS5 content-table COUNT always mirrors memories
            self._db.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")

    @contextlib.contextmanager
    def _ns_file_lock(self, ns: str, exclusive: bool = True):
        """POSIX file lock on `index_<ns>.lock`. Cross-process safe; per-ns scoped.

        Reads share (LOCK_SH), writes are exclusive (LOCK_EX). On systems
        without fcntl (Windows), this is a no-op and falls back to the
        in-process threading.Lock alone.
        """
        if not _HAS_FCNTL:
            yield
            return
        lock_path = self.root / f"index_{ns}.lock"
        # Touch + open r+ so concurrent readers don't truncate each other.
        lock_path.touch(exist_ok=True)
        fh = open(lock_path, "r+")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()

    def _reload_if_stale(self, ns: str) -> None:
        """Force an on-disk reload of the namespace index when a peer has
        written since we last loaded it. No-op if there is no disk file yet
        or our cached copy is already current.

        Callers MUST already hold the ns file lock when invoking this.
        """
        idx_path = self.root / f"index_{ns}.bin"
        if not idx_path.exists():
            return
        disk_mtime = idx_path.stat().st_mtime
        cached_mtime = self._index_mtime.get(ns, -1.0)
        if ns in self._index and disk_mtime <= cached_mtime:
            return
        idx = hnswlib.Index(space="cosine", dim=self.dim)
        idx.load_index(str(idx_path))
        idx.set_ef(64)
        self._index[ns] = idx
        self._index_mtime[ns] = disk_mtime

    def _index_for(self, ns: str) -> hnswlib.Index:
        if ns not in self._index:
            idx_path = self.root / f"index_{ns}.bin"
            idx = hnswlib.Index(space="cosine", dim=self.dim)
            if idx_path.exists():
                idx.load_index(str(idx_path))
                self._index_mtime[ns] = idx_path.stat().st_mtime
                idx.set_ef(64)
            else:
                idx.init_index(max_elements=100_000, ef_construction=200, M=16)
                idx.set_ef(64)
            self._index[ns] = idx
        return self._index[ns]

    def add(
        self,
        texts: list[str],
        vectors: np.ndarray,
        ns: str = "default",
        meta: list[dict] | None = None,
        summaries: list[str | None] | None = None,
    ) -> list[int]:
        if meta is None:
            meta = [{} for _ in texts]
        if summaries is None:
            summaries = [None for _ in texts]
        if len(summaries) != len(texts):
            raise ValueError("summaries length must match texts length")
        # Order matters: take the file lock OUTSIDE the threading.Lock so a
        # blocked peer doesn't pin this process's GIL-bound queues. Inside the
        # file lock we refresh from disk to absorb any peer writes that
        # happened since our last load — this is the fix for the corrupt
        # 14.6 GB index we just rebuilt.
        with self._ns_file_lock(ns, exclusive=True), self._lock:
            self._reload_if_stale(ns)
            ids = []
            for text, m, summary in zip(texts, meta, summaries):
                cur = self._db.execute(
                    "INSERT INTO memories (ns, text, summary, meta) VALUES (?, ?, ?, ?)",
                    (ns, text, summary, json.dumps(m)),
                )
                ids.append(cur.lastrowid)
            self._db.commit()
            idx = self._index_for(ns)
            needed = idx.get_current_count() + len(ids)
            if needed > idx.get_max_elements():
                idx.resize_index(max(needed * 2, idx.get_max_elements() * 2))
            idx.add_items(vectors, ids)
            idx_path = self.root / f"index_{ns}.bin"
            idx.save_index(str(idx_path))
            self._index_mtime[ns] = idx_path.stat().st_mtime
        return ids

    def search(self, vector: np.ndarray, ns: str = "default", top_k: int = 5) -> list[dict[str, Any]]:
        # Shared lock — multiple peers may search the same ns concurrently;
        # only a writer needs to block them. Reload-if-stale picks up freshly
        # written rows from a peer between two of our search calls.
        with self._ns_file_lock(ns, exclusive=False), self._lock:
            self._reload_if_stale(ns)
            idx = self._index_for(ns)
            n = min(top_k, idx.get_current_count())
            if n == 0:
                return []
            try:
                labels, distances = idx.knn_query(vector, k=n)
            except RuntimeError:
                # All elements in this index are mark_deleted; nothing to return.
                return []
            row_ids = [int(x) for x in labels[0]]
            placeholders = ",".join("?" * len(row_ids))
            rows = self._db.execute(
                f"SELECT id, text, summary, meta, created, tier, last_accessed, access_count "
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
                    "summary": row[2],
                    "meta": json.loads(row[3]),
                    "created": row[4],
                    "tier": row[5],
                    "last_accessed": row[6],
                    "access_count": row[7],
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

    def search_bm25(
        self,
        query: str,
        ns: str = "default",
        top_k: int = 20,
        min_tier: int | None = None,
        max_tier: int | None = None,
    ) -> list[dict[str, Any]]:
        """BM25 keyword search via SQLite FTS5. Returns rows ordered best-first.

        score is negated so that higher = better, matching the vector path.
        Empty / punctuation-only queries return [].
        min_tier / max_tier filter by tier (0=pinned, 1=default, 2=ambient).
        """
        match = self._fts_sanitize(query)
        if not match:
            return []
        tier_clause = ""
        tier_params: list[int] = []
        if min_tier is not None:
            tier_clause += " AND m.tier >= ?"
            tier_params.append(min_tier)
        if max_tier is not None:
            tier_clause += " AND m.tier <= ?"
            tier_params.append(max_tier)
        with self._lock:
            try:
                # MATCH without a column qualifier searches every indexed
                # column in the FTS table, so BM25 already covers both `text`
                # and `summary` once the schema bump completes.
                rows = self._db.execute(
                    "SELECT m.id, m.text, m.summary, m.meta, m.created, m.tier, "
                    "m.last_accessed, m.access_count, bm25(memories_fts) AS bm25_raw "
                    "FROM memories_fts "
                    "JOIN memories m ON m.id = memories_fts.rowid "
                    f"WHERE memories_fts MATCH ? AND m.ns = ?{tier_clause} "
                    "ORDER BY bm25_raw LIMIT ?",
                    (match, ns, *tier_params, top_k),
                ).fetchall()
            except sqlite3.OperationalError:
                # Malformed MATCH expression (rare; sanitizer should prevent it).
                return []
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "text": row[1],
                "summary": row[2],
                "meta": json.loads(row[3]),
                "created": row[4],
                "tier": row[5],
                "last_accessed": row[6],
                "access_count": row[7],
                "score": -float(row[8]),  # lower bm25 = better → negate to align with vector
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

    def count(self, ns: str | None = "default") -> int:
        if ns is None:
            row = self._db.execute("SELECT COUNT(*) FROM memories").fetchone()
        else:
            row = self._db.execute("SELECT COUNT(*) FROM memories WHERE ns=?", (ns,)).fetchone()
        return row[0] if row else 0

    def update_summary(self, memory_id: int, summary: str | None) -> bool:
        with self._lock:
            cur = self._db.execute(
                "UPDATE memories SET summary=? WHERE id=?", (summary, memory_id)
            )
            self._db.commit()
        return cur.rowcount > 0

    def list_memories(
        self, ns: str = "default", limit: int = 20, offset: int = 0,
        tier: int | None = None,
    ) -> list[dict]:
        if tier is not None:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created, last_accessed, access_count "
                "FROM memories WHERE ns=? AND tier=? ORDER BY id DESC LIMIT ? OFFSET ?",
                (ns, tier, limit, offset),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created, last_accessed, access_count "
                "FROM memories WHERE ns=? ORDER BY id DESC LIMIT ? OFFSET ?",
                (ns, limit, offset),
            ).fetchall()
        return [
            {
                "id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
                "tier": r[4], "created": r[5], "last_accessed": r[6],
                "access_count": r[7],
            }
            for r in rows
        ]

    def get(self, memory_id: int) -> dict | None:
        row = self._db.execute(
            "SELECT id, ns, text, summary, tier, created, last_accessed, access_count "
            "FROM memories WHERE id=?",
            (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "ns": row[1], "text": row[2], "summary": row[3],
            "tier": row[4], "created": row[5], "last_accessed": row[6],
            "access_count": row[7],
        }

    def delete(self, memory_id: int) -> bool:
        with self._lock:
            row = self._db.execute("SELECT ns FROM memories WHERE id=?", (memory_id,)).fetchone()
            if row is None:
                return False
            ns = row[0]
            cur = self._db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            self._db.commit()
        if cur.rowcount > 0:
            with self._ns_file_lock(ns, exclusive=True), self._lock:
                try:
                    idx = self._index_for(ns)
                    idx.mark_deleted(memory_id)
                    idx_path = self.root / f"index_{ns}.bin"
                    idx.save_index(str(idx_path))
                    self._index_mtime[ns] = idx_path.stat().st_mtime
                except Exception:
                    pass  # label absent from index (e.g. index rebuilt without this entry)
        return cur.rowcount > 0

    def rebuild_ns_index(self, ns: str) -> tuple[int, int]:
        """Rebuild the hnswlib index for *ns* from the SQL source of truth.

        Reads vectors for current SQL row IDs from the existing on-disk index
        (no re-encoding needed), builds a clean index without orphan labels,
        and saves it. Returns (old_element_count, new_element_count).

        Safe on empty namespaces: creates a fresh empty index and removes any
        orphan .bin file (so doctor reports it as a missing index rather than
        a mismatch).
        """
        with self._lock:
            ids = [
                r[0]
                for r in self._db.execute(
                    "SELECT id FROM memories WHERE ns=? ORDER BY id", (ns,)
                ).fetchall()
            ]

        idx_path = self.root / f"index_{ns}.bin"
        # Guard against case-insensitive filesystems (macOS APFS): if another
        # namespace maps to the same on-disk path, the rebuild must be
        # performed via ingest re-encoding, not a blind path overwrite.
        resolved = idx_path.resolve() if idx_path.exists() else idx_path
        for other_ns in self.list_namespaces():
            if other_ns == ns:
                continue
            other_path = (self.root / f"index_{other_ns}.bin")
            if other_path.exists() and other_path.resolve() == resolved:
                raise RuntimeError(
                    f"rebuild_ns_index('{ns}'): index path collides with ns='{other_ns}' "
                    f"on a case-insensitive filesystem. Merge the namespaces first."
                )
        with self._ns_file_lock(ns, exclusive=True), self._lock:
            old_count = 0
            if idx_path.exists():
                old_idx = hnswlib.Index(space="cosine", dim=self.dim)
                old_idx.load_index(str(idx_path))
                old_count = old_idx.get_current_count()
            else:
                old_idx = None

            if not ids:
                # Namespace has no SQL rows — do NOT create an empty .bin file
                # (that would produce a new orphan). Leave any existing .bin for
                # health_check / repair to clean up.
                return old_count, 0

            new_idx = hnswlib.Index(space="cosine", dim=self.dim)
            new_idx.init_index(
                max_elements=max(100_000, len(ids) * 2),
                ef_construction=200,
                M=16,
            )
            new_idx.set_ef(64)

            if old_idx is not None:
                # Retrieve stored vectors in batches; skip IDs absent from index.
                surviving, vecs = [], []
                for mid in ids:
                    try:
                        vec = old_idx.get_items([mid])
                        surviving.append(mid)
                        vecs.append(vec[0])
                    except Exception:
                        pass
                if surviving:
                    new_idx.add_items(np.array(vecs, dtype="float32"), surviving)

            new_idx.save_index(str(idx_path))
            self._index[ns] = new_idx
            self._index_mtime[ns] = idx_path.stat().st_mtime

        return old_count, len(ids)

    def forget_candidates(
        self,
        ns: str,
        before: str | None = None,
        tier: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return rows that would be deleted by forget(). Does NOT modify the store."""
        sql = (
            "SELECT id, ns, tier, created, substr(text, 1, 80) AS preview "
            "FROM memories WHERE ns=?"
        )
        params: list[Any] = [ns]
        if before is not None:
            sql += " AND created < ?"
            params.append(before)
        if tier is not None:
            sql += " AND tier = ?"
            params.append(tier)
        else:
            sql += " AND tier != 0"  # protect pinned rows by default
        sql += " ORDER BY created"
        rows = self._db.execute(sql, params).fetchall()
        return [
            {"id": r[0], "ns": r[1], "tier": r[2], "created": r[3], "preview": r[4]}
            for r in rows
        ]

    def forget(
        self,
        ns: str,
        before: str | None = None,
        tier: int | None = None,
    ) -> int:
        """Bulk-delete memories in *ns*, optionally filtered by date/tier.

        Pinned rows (tier=0) are skipped unless tier=0 is explicitly passed.
        Also removes the deleted IDs from the hnswlib vector index.
        Returns the number of rows deleted.
        """
        candidates = self.forget_candidates(ns=ns, before=before, tier=tier)
        if not candidates:
            return 0
        ids = [c["id"] for c in candidates]
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            self._db.execute(f"DELETE FROM memories WHERE id IN ({placeholders})", ids)
            self._db.commit()
        with self._ns_file_lock(ns, exclusive=True), self._lock:
            try:
                idx = self._index_for(ns)
                for mid in ids:
                    try:
                        idx.mark_deleted(mid)
                    except Exception:
                        pass
                idx_path = self.root / f"index_{ns}.bin"
                idx.save_index(str(idx_path))
                self._index_mtime[ns] = idx_path.stat().st_mtime
            except Exception:
                pass
        return len(ids)

    def health_check(self) -> dict[str, Any]:
        """Return a health report: DB integrity, WAL size, per-ns index counts, orphan indexes."""
        report: dict[str, Any] = {}

        # SQLite integrity
        row = self._db.execute("PRAGMA integrity_check").fetchone()
        report["db_integrity"] = row[0] if row else "unknown"

        # WAL file size
        wal_path = self.root / "memories.db-wal"
        report["wal_size"] = wal_path.stat().st_size if wal_path.exists() else 0

        # Per-namespace counts
        ns_sql: dict[str, int] = {}
        for r in self._db.execute("SELECT ns, count(*) FROM memories GROUP BY ns").fetchall():
            ns_sql[r[0]] = r[1]

        ns_reports = []
        for ns, sql_count in sorted(ns_sql.items()):
            idx_path = self.root / f"index_{ns}.bin"
            idx_count: int | None = None
            max_elements: int | None = None
            if idx_path.exists():
                try:
                    idx = hnswlib.Index(space="cosine", dim=self.dim)
                    idx.load_index(str(idx_path))
                    idx_count = idx.get_current_count()
                    max_elements = idx.get_max_elements()
                except Exception:
                    idx_count = None
            soft_deleted = max(0, (idx_count or 0) - sql_count) if idx_count is not None else 0
            missing_vectors = max(0, sql_count - (idx_count or 0)) if idx_count is not None else 0
            usage_pct = round(100 * (idx_count or 0) / max_elements, 1) if max_elements else None
            ns_reports.append({
                "ns": ns,
                "sql_count": sql_count,
                "idx_count": idx_count,
                "max_elements": max_elements,
                "usage_pct": usage_pct,
                "soft_deleted": soft_deleted,
                "missing_vectors": missing_vectors,
                "idx_missing": idx_count is None,
                "capacity_warning": usage_pct is not None and usage_pct >= 85.0,
            })
        report["namespaces"] = ns_reports

        # Orphan indexes (index file exists but 0 SQL rows)
        orphans = []
        for idx_path in sorted(self.root.glob("index_*.bin")):
            ns = idx_path.name[len("index_"):-len(".bin")]
            if ns_sql.get(ns, 0) == 0:
                orphans.append({"ns": ns, "size": idx_path.stat().st_size, "path": str(idx_path)})
        report["orphan_indexes"] = orphans

        return report

    def repair(self) -> dict[str, Any]:
        """Auto-repair issues found by health_check().

        Fixes:
          - Orphan vectors (idx > sql): calls rebuild_ns_index() per namespace.
          - Orphan .bin files (no SQL rows): deletes the file.

        Cannot fix:
          - Missing vectors (sql > idx): requires re-encoding; reported only.

        Returns a summary dict with keys: orphan_vectors_fixed,
        orphan_indexes_removed, missing_vectors_reported.
        """
        report = self.health_check()
        result: dict[str, Any] = {
            "orphan_vectors_fixed": [],
            "orphan_indexes_removed": [],
            "missing_vectors_reported": [],
        }

        for ns_info in report["namespaces"]:
            if ns_info.get("soft_deleted", 0) > 0:
                try:
                    old_n, new_n = self.rebuild_ns_index(ns_info["ns"])
                    result["orphan_vectors_fixed"].append({
                        "ns": ns_info["ns"],
                        "removed": old_n - new_n,
                    })
                except RuntimeError as e:
                    result["orphan_vectors_fixed"].append({
                        "ns": ns_info["ns"],
                        "error": str(e),
                    })
            if ns_info.get("missing_vectors", 0) > 0:
                result["missing_vectors_reported"].append({
                    "ns": ns_info["ns"],
                    "missing": ns_info["missing_vectors"],
                })

        for orphan in report["orphan_indexes"]:
            try:
                Path(orphan["path"]).unlink(missing_ok=True)
                result["orphan_indexes_removed"].append(orphan["path"])
            except Exception as e:
                result["orphan_indexes_removed"].append({"path": orphan["path"], "error": str(e)})

        return result

    def gc_candidates(self, ns: str | None = None, age_days: int = 30, tier: int = 2) -> list[dict[str, Any]]:
        """Rows eligible for garbage collection.

        tier=2 (default): ambient rows that were never accessed, older than `age_days`.
        tier=1: default rows older than `age_days` (access_count not required to be 0).
        Pinned rows (tier=0) are never eligible.
        """
        if tier not in (1, 2):
            raise ValueError("gc tier must be 1 or 2 (pinned tier=0 is never gc-eligible)")
        sql = (
            "SELECT id, ns, substr(text, 1, 80) AS preview, "
            "CAST(julianday('now') - julianday(created) AS INTEGER) AS age_days "
            "FROM memories "
            f"WHERE tier = {tier} "
        )
        if tier == 2:
            sql += "AND access_count = 0 "
        sql += "AND julianday('now') - julianday(created) > ?"
        params: list[Any] = [age_days]
        if ns is not None:
            sql += " AND ns = ?"
            params.append(ns)
        sql += " ORDER BY age_days DESC"
        rows = self._db.execute(sql, params).fetchall()
        return [{"id": r[0], "ns": r[1], "preview": r[2], "age_days": r[3]} for r in rows]

    def gc(self, ns: str | None = None, age_days: int = 30, tier: int = 2) -> int:
        """Delete the candidates returned by gc_candidates. Returns deleted count."""
        candidates = self.gc_candidates(ns=ns, age_days=age_days, tier=tier)
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
        ns_to_ids: dict[str, list[int]] = {}
        for c in candidates:
            ns_to_ids.setdefault(c["ns"], []).append(c["id"])
        for cand_ns, cand_ids in ns_to_ids.items():
            with self._ns_file_lock(cand_ns, exclusive=True), self._lock:
                try:
                    idx = self._index_for(cand_ns)
                    for cid in cand_ids:
                        try:
                            idx.mark_deleted(cid)
                        except Exception:
                            pass
                    idx_path = self.root / f"index_{cand_ns}.bin"
                    idx.save_index(str(idx_path))
                    self._index_mtime[cand_ns] = idx_path.stat().st_mtime
                except Exception:
                    pass
        return len(ids)
