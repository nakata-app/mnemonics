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
        tier: int = 1,
    ) -> list[int]:
        if tier not in (0, 1, 2):
            raise ValueError(f"tier must be 0, 1, or 2; got {tier!r}")
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
                    "INSERT INTO memories (ns, text, summary, meta, tier) VALUES (?, ?, ?, ?, ?)",
                    (ns, text, summary, json.dumps(m), tier),
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

    def search(
        self,
        vector: np.ndarray,
        ns: str = "default",
        top_k: int = 5,
        min_tier: int | None = None,
        max_tier: int | None = None,
    ) -> list[dict[str, Any]]:
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
            tier_clause = ""
            tier_params: list[int] = []
            if min_tier is not None:
                tier_clause += " AND tier >= ?"
                tier_params.append(min_tier)
            if max_tier is not None:
                tier_clause += " AND tier <= ?"
                tier_params.append(max_tier)
            rows = self._db.execute(
                f"SELECT id, text, summary, meta, created, tier, last_accessed, access_count "
                f"FROM memories WHERE id IN ({placeholders}){tier_clause}",
                (*row_ids, *tier_params),
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

    def hybrid_search(
        self,
        vector: np.ndarray,
        query: str,
        ns: str = "default",
        top_k: int = 5,
        rrf_k: int = 60,
        min_tier: int | None = None,
        max_tier: int | None = None,
    ) -> list[dict[str, Any]]:
        """Combine vector search and BM25 with Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (rrf_k + rank)) across result lists, where rank
        is 1-based. Top *top_k* results by combined RRF score are returned.
        Each result dict carries the original vector *score*, plus
        *rrf_score*, *vector_rank*, and *bm25_rank* for transparency.
        """
        fetch_n = max(top_k * 4, 20)
        vec_results = self.search(vector, ns=ns, top_k=fetch_n,
                                  min_tier=min_tier, max_tier=max_tier)
        bm25_results = self.search_bm25(query, ns=ns, top_k=fetch_n,
                                        min_tier=min_tier, max_tier=max_tier)

        rrf: dict[int, float] = {}
        by_id: dict[int, dict] = {}

        for rank, item in enumerate(vec_results, start=1):
            iid = item["id"]
            rrf[iid] = rrf.get(iid, 0.0) + 1.0 / (rrf_k + rank)
            by_id[iid] = item
            by_id[iid]["vector_rank"] = rank
            by_id[iid].setdefault("bm25_rank", None)

        for rank, item in enumerate(bm25_results, start=1):
            iid = item["id"]
            rrf[iid] = rrf.get(iid, 0.0) + 1.0 / (rrf_k + rank)
            if iid not in by_id:
                by_id[iid] = item
                by_id[iid].setdefault("vector_rank", None)
            by_id[iid]["bm25_rank"] = rank

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for iid, rrf_score in ranked:
            item = dict(by_id[iid])
            item["rrf_score"] = round(rrf_score, 6)
            results.append(item)
        return results

    def similar_to(
        self,
        memory_id: int,
        top_k: int = 5,
        min_tier: int | None = None,
        max_tier: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top_k memories most similar to *memory_id*.

        Loads the stored vector for *memory_id* from its namespace's hnswlib
        index, then runs a vector search excluding *memory_id* itself.
        Returns [] if the memory does not exist or has no vector.
        """
        row = self._db.execute(
            "SELECT ns FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if row is None:
            return []
        ns = row[0]
        with self._ns_file_lock(ns, exclusive=False), self._lock:
            self._reload_if_stale(ns)
            idx = self._index_for(ns)
            try:
                vec = idx.get_items([memory_id])[0]
            except Exception:
                return []
            vec_arr = np.array(vec, dtype="float32")
            fetch_n = min(top_k + 1, idx.get_current_count())
            if fetch_n == 0:
                return []
            try:
                labels, distances = idx.knn_query(vec_arr, k=fetch_n)
            except RuntimeError:
                return []
            row_ids = [int(x) for x in labels[0] if int(x) != memory_id][:top_k]
            if not row_ids:
                return []
            tier_clause = ""
            tier_params: list[int] = []
            if min_tier is not None:
                tier_clause += " AND tier >= ?"
                tier_params.append(min_tier)
            if max_tier is not None:
                tier_clause += " AND tier <= ?"
                tier_params.append(max_tier)
            placeholders = ",".join("?" * len(row_ids))
            rows = self._db.execute(
                f"SELECT id, text, summary, meta, created, tier, last_accessed, access_count "
                f"FROM memories WHERE id IN ({placeholders}){tier_clause}",
                (*row_ids, *tier_params),
            ).fetchall()
        by_id = {r[0]: r for r in rows}
        dist_by_id = {int(l): float(d) for l, d in zip(labels[0], distances[0])}
        results = []
        for rid in row_ids:
            row = by_id.get(rid)
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
                "score": float(1 - dist_by_id.get(rid, 1.0)),
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

    def touch_many(self, memory_ids: list[int]) -> int:
        """Update last_accessed = now() and increment access_count for the given IDs.

        Returns the number of rows actually updated (non-existent IDs are skipped).
        Useful for marking a batch of memories as accessed after retrieval via
        get_many() or any other method that doesn't auto-touch.
        """
        if not memory_ids:
            return 0
        placeholders = ",".join("?" * len(memory_ids))
        with self._lock:
            cur = self._db.execute(
                f"UPDATE memories SET last_accessed = datetime('now'), "
                f"access_count = access_count + 1 WHERE id IN ({placeholders})",
                memory_ids,
            )
            self._db.commit()
        return cur.rowcount

    def recent_accessed(
        self,
        ns: str | None = "default",
        limit: int = 20,
        tier: int | None = None,
    ) -> list[dict]:
        """Return memories ordered by last_accessed descending (most recently used first).

        Memories that have never been accessed (last_accessed IS NULL) are included
        last. Useful for surfacing recently retrieved context in AI sessions.
        """
        where: list[str] = []
        params: list = []
        if ns is not None:
            where.append("ns=?")
            params.append(ns)
        if tier is not None:
            where.append("tier=?")
            params.append(tier)
        where_clause = ("WHERE " + " AND ".join(where)) if where else ""
        params.append(limit)
        rows = self._db.execute(
            f"SELECT id, ns, text, summary, tier, created, last_accessed, access_count "
            f"FROM memories {where_clause} "
            f"ORDER BY last_accessed DESC NULLS LAST, id DESC LIMIT ?",
            params,
        ).fetchall()
        return [
            {
                "id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
                "tier": r[4], "created": r[5], "last_accessed": r[6],
                "access_count": r[7],
            }
            for r in rows
        ]

    def top_accessed(
        self,
        ns: str | None = "default",
        limit: int = 20,
        tier: int | None = None,
    ) -> list[dict]:
        """Return memories ordered by access_count descending (most frequently used first).

        Memories never accessed (access_count=0) appear last. Pairs with
        recent_accessed: this surfaces long-term valuable memories; that one
        surfaces active session context.
        """
        where: list[str] = []
        params: list = []
        if ns is not None:
            where.append("ns=?")
            params.append(ns)
        if tier is not None:
            where.append("tier=?")
            params.append(tier)
        where_clause = ("WHERE " + " AND ".join(where)) if where else ""
        params.append(limit)
        rows = self._db.execute(
            f"SELECT id, ns, text, summary, tier, created, last_accessed, access_count "
            f"FROM memories {where_clause} "
            f"ORDER BY access_count DESC, id DESC LIMIT ?",
            params,
        ).fetchall()
        return [
            {
                "id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
                "tier": r[4], "created": r[5], "last_accessed": r[6],
                "access_count": r[7],
            }
            for r in rows
        ]

    def set_tier_many(self, memory_ids: list[int], tier: int) -> int:
        """Set tier for multiple memories in a single transaction.

        Returns the number of rows actually updated (missing IDs are skipped).
        """
        if tier not in (0, 1, 2):
            raise ValueError("tier must be 0 (pinned), 1 (default), or 2 (ambient)")
        if not memory_ids:
            return 0
        placeholders = ",".join("?" * len(memory_ids))
        with self._lock:
            cur = self._db.execute(
                f"UPDATE memories SET tier=? WHERE id IN ({placeholders})",
                [tier, *memory_ids],
            )
            self._db.commit()
        return cur.rowcount

    def list_namespaces(self) -> list[str]:
        rows = self._db.execute("SELECT DISTINCT ns FROM memories ORDER BY ns").fetchall()
        return [r[0] for r in rows]

    def count(self, ns: str | None = "default") -> int:
        if ns is None:
            row = self._db.execute("SELECT COUNT(*) FROM memories").fetchone()
        else:
            row = self._db.execute("SELECT COUNT(*) FROM memories WHERE ns=?", (ns,)).fetchone()
        return row[0] if row else 0

    def update_text(self, memory_id: int, new_text: str, new_vec: "np.ndarray") -> bool:
        """Replace the text and embedding vector for an existing memory.

        The old embedding is marked deleted in the hnswlib index and the new
        vector is added under the same label (``memory_id``).  The SQLite row
        is updated in the same transaction.  Returns ``True`` on success,
        ``False`` if *memory_id* does not exist.
        """
        import numpy as _np_ut
        row = self._db.execute(
            "SELECT ns FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if row is None:
            return False
        ns = row[0]
        vec = _np_ut.asarray(new_vec, dtype="float32").reshape(1, -1)
        with self._lock:
            self._db.execute(
                "UPDATE memories SET text=? WHERE id=?", (new_text, memory_id)
            )
            self._db.commit()
            idx = self._index_for(ns)
            try:
                idx.mark_deleted(memory_id)
            except Exception:
                pass
            idx.add_items(vec, ids=[memory_id])
        return True

    def clone(self, memory_id: int, target_ns: str) -> int | None:
        """Clone a single memory to *target_ns*.

        Creates a new row in *target_ns* with the same ``text``, ``summary``,
        ``meta``, and ``tier`` as *memory_id*, resets ``access_count`` and
        ``last_accessed`` to their initial values, and copies the stored
        embedding vector into the target namespace index.

        Returns the new memory ID on success, or ``None`` if *memory_id* is
        not found or its vector cannot be read.
        """
        row = self._db.execute(
            "SELECT ns, text, summary, meta, tier FROM memories WHERE id=?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        src_ns, text, summary, meta_json, tier = row
        import json as _json_clone
        meta = _json_clone.loads(meta_json) if meta_json else {}
        try:
            src_idx = self._index_for(src_ns)
            vec_list = src_idx.get_items([memory_id])[0]
        except Exception:
            return None
        import numpy as _np_clone
        vec = _np_clone.array(vec_list, dtype="float32").reshape(1, -1)
        new_ids = self.add(
            [text], vec, ns=target_ns,
            meta=[meta], summaries=[summary], tier=tier,
        )
        return new_ids[0] if new_ids else None

    def move_to_ns(self, memory_ids: list[int], target_ns: str) -> int:
        """Move a list of memories to *target_ns* by updating their ``ns`` column.

        Vector index files are NOT updated here — the source namespace index
        may now contain orphan labels (the moved IDs) and the destination
        namespace index will be missing them. Run :meth:`rebuild_ns_index`
        on both source and destination namespaces after moving to keep the
        indexes consistent. Returns the count of rows actually moved
        (missing IDs are silently skipped).
        """
        if not memory_ids:
            return 0
        placeholders = ",".join("?" * len(memory_ids))
        with self._lock:
            cur = self._db.execute(
                f"UPDATE memories SET ns=? WHERE id IN ({placeholders})",
                [target_ns, *memory_ids],
            )
            self._db.commit()
        return cur.rowcount

    def pinned_memories(
        self,
        ns: str | None = "default",
        limit: int = 100,
    ) -> list[dict]:
        """Return all pinned (tier=0) memories in *ns*, ordered by ``created DESC``.

        Convenience wrapper around the tier=0 filter.  *ns=None* spans all
        namespaces.  Each result dict has ``id``, ``ns``, ``text``,
        ``summary``, ``tier``, ``created``, and ``meta`` (as parsed dict).
        """
        import json as _j_pm

        if ns is not None:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created, meta "
                "FROM memories WHERE ns=? AND tier=0 ORDER BY created DESC LIMIT ?",
                (ns, limit),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created, meta "
                "FROM memories WHERE tier=0 ORDER BY created DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [
            {
                "id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
                "tier": r[4], "created": r[5],
                "meta": _j_pm.loads(r[6]) if r[6] else {},
            }
            for r in rows
        ]

    def update_meta_key(self, memory_id: int, key: str, value: object) -> bool:
        """Set or update a single key in a memory's meta dict.

        Reads the current meta, merges in ``{key: value}``, and writes it
        back.  If *value* is ``None`` the key is removed (if present).
        Returns ``True`` on success, ``False`` if *memory_id* does not exist.
        """
        import json as _j_umk

        row = self._db.execute(
            "SELECT meta FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if row is None:
            return False

        meta = _j_umk.loads(row[0]) if row[0] else {}
        if value is None:
            meta.pop(key, None)
        else:
            meta[key] = value
        with self._lock:
            self._db.execute(
                "UPDATE memories SET meta=? WHERE id=?",
                (_j_umk.dumps(meta, ensure_ascii=False), memory_id),
            )
            self._db.commit()
        return True

    def toggle_tier(self, memory_id: int) -> int | None:
        """Cycle a memory's tier: pinned(0) → ambient(2) → default(1) → pinned(0).

        Returns the new tier value, or ``None`` if *memory_id* does not exist.
        This round-trip cycle lets a UI button step through all three tiers
        with a single call.
        """
        row = self._db.execute(
            "SELECT tier FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        current = row[0]
        _cycle = {0: 2, 2: 1, 1: 0}
        new_tier = _cycle.get(current, 1)
        with self._lock:
            self._db.execute(
                "UPDATE memories SET tier=? WHERE id=?", (new_tier, memory_id)
            )
            self._db.commit()
        return new_tier

    def merge_texts(
        self,
        ids: list[int],
        separator: str = "\n\n",
        ns: str = "default",
        delete_originals: bool = False,
    ) -> int | None:
        """Concatenate the text of multiple memories into a new memory in *ns*.

        Returns the ``id`` of the newly created memory, or ``None`` if no
        memories were found for the given *ids*.  The new memory uses zero
        vectors (call ``reindex_all(ns)`` afterwards).  When
        *delete_originals* is ``True``, the source memories are deleted after
        the merge.
        """
        import json as _j_mt
        import numpy as _np_mt

        rows = self._db.execute(
            f"SELECT id, text, summary, meta FROM memories "
            f"WHERE id IN ({','.join('?' * len(ids))})",
            ids,
        ).fetchall() if ids else []

        if not rows:
            return None

        merged_text = separator.join(r[1] for r in rows)
        first_meta = _j_mt.loads(rows[0][3]) if rows[0][3] else {}
        zero_vec = _np_mt.zeros((1, self.dim), dtype=_np_mt.float32)
        new_ids = self.add([merged_text], zero_vec, ns=ns, meta=[first_meta])
        new_id = new_ids[0] if new_ids else None

        if delete_originals and new_id is not None:
            src_ids = [r[0] for r in rows]
            placeholders = ",".join("?" * len(src_ids))
            with self._lock:
                self._db.execute(
                    f"DELETE FROM memories WHERE id IN ({placeholders})", src_ids
                )
                self._db.commit()

        return new_id

    def truncate_text(self, memory_id: int, max_chars: int) -> bool:
        """Truncate a memory's text to at most *max_chars* characters.

        No-ops if the text is already shorter. Returns ``True`` on success,
        ``False`` if *memory_id* does not exist.  Does NOT re-embed; call the
        ``update`` method if you need the vector to stay in sync.
        """
        row = self._db.execute(
            "SELECT text FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if row is None:
            return False
        text = row[0]
        if len(text) <= max_chars:
            return True
        with self._lock:
            self._db.execute(
                "UPDATE memories SET text=? WHERE id=?",
                (text[:max_chars], memory_id),
            )
            self._db.commit()
        return True

    def search_by_access_count(
        self,
        min_count: int = 0,
        max_count: int | None = None,
        ns: str | None = "default",
        limit: int = 20,
    ) -> list[dict]:
        """Return memories whose ``access_count`` falls in [*min_count*, *max_count*].

        *max_count=None* means no upper bound.  *ns=None* spans all
        namespaces.  Ordered by access_count DESC.  Useful for finding
        heavily-used memories (min_count=10) or completely untouched ones
        (min_count=0, max_count=0).
        """
        where: list[str] = ["access_count >= ?"]
        params: list = [min_count]
        if max_count is not None:
            where.append("access_count <= ?")
            params.append(max_count)
        if ns is not None:
            where.append("ns = ?")
            params.append(ns)
        params.append(limit)
        rows = self._db.execute(
            "SELECT id, ns, text, summary, tier, created, access_count "
            "FROM memories WHERE " + " AND ".join(where) +
            " ORDER BY access_count DESC LIMIT ?",
            params,
        ).fetchall()
        return [
            {"id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
             "tier": r[4], "created": r[5], "access_count": r[6]}
            for r in rows
        ]

    def bulk_summarize(
        self,
        updates: dict[int, str | None],
    ) -> int:
        """Update summaries for multiple memories in a single operation.

        *updates* maps memory ID → summary string (or ``None`` to clear).
        Silently skips IDs that do not exist.  Returns the count of rows
        actually updated.

        Functionally equivalent to calling ``update_summary`` in a loop but
        issued as a single transaction, which is faster for large batches.
        """
        if not updates:
            return 0
        count = 0
        with self._lock:
            for mid, summary in updates.items():
                cur = self._db.execute(
                    "UPDATE memories SET summary=? WHERE id=?", (summary, int(mid))
                )
                count += cur.rowcount
            self._db.commit()
        return count

    def cross_ns_search(
        self,
        query: str,
        namespaces: list[str],
        limit: int = 10,
    ) -> list[dict]:
        """Case-insensitive LIKE search across a specific set of namespaces.

        Unlike ``search_text(ns=None)`` which spans ALL namespaces, this method
        restricts the search to the given *namespaces* list.  Returns up to
        *limit* results ordered by created DESC.  Returns empty list when
        *namespaces* is empty.
        """
        if not namespaces:
            return []
        ph = ",".join("?" * len(namespaces))
        rows = self._db.execute(
            f"SELECT id, ns, text, summary, tier, created FROM memories "
            f"WHERE ns IN ({ph}) AND text LIKE ? COLLATE NOCASE "
            f"ORDER BY created DESC LIMIT ?",
            (*namespaces, f"%{query}%", limit),
        ).fetchall()
        return [
            {"id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
             "tier": r[4], "created": r[5]}
            for r in rows
        ]

    def memory_timeline(
        self,
        ns: str | None = "default",
        limit: int = 50,
        tier: int | None = None,
    ) -> list[dict]:
        """Return memories in chronological order (oldest first) with a running index.

        Useful for reviewing the narrative arc of a namespace.  Each dict
        includes ``{"seq": int, "id": int, "ns": str, "text": str,
        "summary": str|None, "tier": int, "created": str}``.

        *ns=None* spans all namespaces.  *tier* optionally filters by tier.
        """
        conds = ["1=1"]
        params: list = []
        if ns is not None:
            conds.append("ns=?")
            params.append(ns)
        if tier is not None:
            conds.append("tier=?")
            params.append(tier)
        params.append(limit)
        rows = self._db.execute(
            f"SELECT id, ns, text, summary, tier, created FROM memories "
            f"WHERE {' AND '.join(conds)} ORDER BY created ASC LIMIT ?",
            params,
        ).fetchall()
        return [
            {"seq": i + 1, "id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
             "tier": r[4], "created": r[5]}
            for i, r in enumerate(rows)
        ]

    def keyword_extract(
        self,
        memory_id: int,
        top_n: int = 10,
    ) -> list[dict] | None:
        """Extract top-N keywords from a single memory's text using TF-style scoring.

        Uses term-frequency weighted by inverse log-frequency across the entire
        store.  Returns a list of ``{"word": str, "score": float}`` dicts sorted
        descending, or ``None`` if *memory_id* does not exist.

        This is a lightweight in-process approximation — no external NLP library
        required.  Stop-words are filtered.
        """
        import math
        import re

        row = self._db.execute(
            "SELECT text FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if row is None:
            return None

        _STOP = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "it", "i", "you", "he", "she", "we", "they", "this", "that", "as",
            "not", "no", "so", "if", "do", "did", "has", "have", "had", "will",
            "can", "may", "my", "your", "its", "our", "their", "which", "who",
        }

        def _tokens(text: str) -> list[str]:
            return [w for w in re.findall(r"[a-zA-Z]{3,}", text.lower()) if w not in _STOP]

        doc_tokens = _tokens(row[0])
        if not doc_tokens:
            return []

        tf: dict[str, int] = {}
        for w in doc_tokens:
            tf[w] = tf.get(w, 0) + 1

        total_docs = self._db.execute("SELECT COUNT(*) FROM memories").fetchone()[0] or 1
        vocab = list(tf)
        df_rows = self._db.execute(
            f"SELECT lower(text) FROM memories"
        ).fetchall()
        df: dict[str, int] = {}
        for (text_row,) in df_rows:
            seen = set(_tokens(text_row))
            for w in seen:
                if w in tf:
                    df[w] = df.get(w, 0) + 1

        results = []
        for w, freq in tf.items():
            idf = math.log((total_docs + 1) / (df.get(w, 0) + 1)) + 1
            results.append({"word": w, "score": round(freq * idf, 4)})

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]

    def filter_by_text_length(
        self,
        min_chars: int = 0,
        max_chars: int | None = None,
        ns: str | None = "default",
        limit: int = 100,
    ) -> list[dict]:
        """Return memories whose text length falls in [min_chars, max_chars].

        Useful for identifying very short stubs (min=0, max=20) or very long
        chunks that may need splitting.  *ns=None* spans all namespaces.
        """
        conds = ["1=1"]
        params: list = []
        if ns is not None:
            conds.append("ns=?")
            params.append(ns)
        conds.append("LENGTH(text) >= ?")
        params.append(min_chars)
        if max_chars is not None:
            conds.append("LENGTH(text) <= ?")
            params.append(max_chars)
        params.append(limit)
        rows = self._db.execute(
            f"SELECT id, ns, text, summary, tier, created FROM memories "
            f"WHERE {' AND '.join(conds)} "
            f"ORDER BY LENGTH(text) DESC LIMIT ?",
            params,
        ).fetchall()
        return [
            {"id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
             "tier": r[4], "created": r[5]}
            for r in rows
        ]

    def multi_tag_filter(
        self,
        tags: list[str],
        ns: str | None = "default",
        mode: str = "all",
        limit: int = 100,
    ) -> list[dict]:
        """Return memories that have ALL (mode='all') or ANY (mode='any') of the given tags.

        *ns=None* spans all namespaces.  Returns empty list when *tags* is empty.
        """
        if not tags:
            return []
        import json as _j_mtf

        if ns is not None:
            rows_all = self._db.execute(
                "SELECT id, ns, text, summary, tier, created, meta FROM memories WHERE ns=?",
                (ns,),
            ).fetchall()
        else:
            rows_all = self._db.execute(
                "SELECT id, ns, text, summary, tier, created, meta FROM memories"
            ).fetchall()

        out = []
        tag_set = set(tags)
        for r in rows_all:
            try:
                meta_obj = _j_mtf.loads(r[6]) if r[6] else {}
                row_tags = set(meta_obj.get("tags") or [])
            except Exception:
                row_tags = set()
            if mode == "all":
                match = tag_set.issubset(row_tags)
            else:
                match = bool(tag_set & row_tags)
            if match:
                out.append({"id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
                            "tier": r[4], "created": r[5]})
                if len(out) >= limit:
                    break
        return out

    def tag_stats(self, ns: str | None = "default") -> list[dict]:
        """Return tag usage statistics: tag name, count, and tier breakdown.

        Each dict has: ``{"tag": str, "count": int, "pinned": int,
        "default": int, "ambient": int}``.
        Ordered by count DESC.  *ns=None* spans all namespaces.
        """
        import json as _j_ts

        if ns is not None:
            rows = self._db.execute(
                "SELECT meta, tier FROM memories WHERE ns=?", (ns,)
            ).fetchall()
        else:
            rows = self._db.execute("SELECT meta, tier FROM memories").fetchall()

        stats: dict[str, dict[str, int]] = {}
        for meta_str, tier in rows:
            try:
                tags = (_j_ts.loads(meta_str) if meta_str else {}).get("tags") or []
            except Exception:
                tags = []
            for tag in tags:
                if tag not in stats:
                    stats[tag] = {"count": 0, "pinned": 0, "default": 0, "ambient": 0}
                stats[tag]["count"] += 1
                label = {0: "pinned", 1: "default", 2: "ambient"}.get(tier, "default")
                stats[tag][label] = stats[tag].get(label, 0) + 1

        return sorted(
            [{"tag": t, **v} for t, v in stats.items()],
            key=lambda x: x["count"],
            reverse=True,
        )

    def split_memory(
        self,
        memory_id: int,
        separator: str | None = None,
        max_chars: int | None = None,
        delete_original: bool = False,
    ) -> list[int] | None:
        """Split a memory's text into two or more parts.

        Two split modes:
        - *separator*: split on the given string (e.g. ``"\\n\\n"``).  Empty parts are dropped.
        - *max_chars*: split into word-boundary chunks of at most *max_chars* characters.

        Parts inherit the original's ns, meta, tier, summary.  Vectors are
        zero-filled; call ``reindex_all`` afterwards.

        Returns the list of new IDs, or ``None`` if *memory_id* not found or
        fewer than 2 non-empty parts.
        """
        import json as _j_sm
        import numpy as _np_sm

        row = self._db.execute(
            "SELECT ns, text, summary, meta, tier FROM memories WHERE id=?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None

        ns, text, summary, meta_str, tier = row
        meta_obj = _j_sm.loads(meta_str) if meta_str else {}

        if separator is not None:
            parts = [p.strip() for p in text.split(separator)]
        elif max_chars is not None and max_chars > 0:
            words = text.split()
            parts = []
            buf: list[str] = []
            for word in words:
                if buf and (sum(len(w) for w in buf) + len(buf) - 1 + len(word) + 1) > max_chars:
                    parts.append(" ".join(buf))
                    buf = [word]
                else:
                    buf.append(word)
            if buf:
                parts.append(" ".join(buf))
        else:
            parts = [text]

        parts = [p for p in parts if p]
        if len(parts) < 2:
            return None

        zero_vec = _np_sm.zeros((len(parts), self.dim), dtype=_np_sm.float32)
        new_ids = self.add(
            parts, zero_vec, ns=ns, meta=[meta_obj] * len(parts),
            summaries=[summary] * len(parts),
        )

        if tier != 1:
            ph = ",".join("?" * len(new_ids))
            self._db.execute(
                f"UPDATE memories SET tier=? WHERE id IN ({ph})", (tier, *new_ids)
            )
            self._db.commit()

        if delete_original:
            self.delete(memory_id)

        return new_ids

    def clone_memory(
        self,
        memory_id: int,
        dst_ns: str | None = None,
    ) -> int | None:
        """Create an exact copy of a memory, optionally in a different namespace.

        The clone gets a fresh ``id`` and ``created`` timestamp but inherits
        the original's text, summary, meta, and tier.  Vectors are zero-filled
        (call ``reindex_all`` afterwards).  Returns the new ``id``, or
        ``None`` if *memory_id* does not exist.
        """
        import json as _j_cm
        import numpy as _np_cm

        row = self._db.execute(
            "SELECT ns, text, summary, meta, tier FROM memories WHERE id=?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None

        target_ns = dst_ns if dst_ns is not None else row[0]
        meta = _j_cm.loads(row[2]) if row[2] else {}  # summary is row[2]?
        # Correct order: ns, text, summary, meta, tier
        ns_orig, text, summary, meta_str, tier = row
        meta_obj = _j_cm.loads(meta_str) if meta_str else {}

        zero_vec = _np_cm.zeros((1, self.dim), dtype=_np_cm.float32)
        new_ids = self.add(
            [text], zero_vec, ns=target_ns, meta=[meta_obj], summaries=[summary]
        )
        new_id = new_ids[0] if new_ids else None
        if new_id is not None and tier != 1:
            self._db.execute("UPDATE memories SET tier=? WHERE id=?", (tier, new_id))
            self._db.commit()
        return new_id

    def memories_without_summary(
        self,
        ns: str | None = "default",
        limit: int = 20,
    ) -> list[dict]:
        """Return memories that have a NULL or empty summary.

        Useful for identifying memories that need summarization.  *ns=None*
        spans all namespaces.  Ordered by created DESC.
        """
        if ns is not None:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created FROM memories "
                "WHERE ns=? AND (summary IS NULL OR summary = '') "
                "ORDER BY created DESC LIMIT ?",
                (ns, limit),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created FROM memories "
                "WHERE (summary IS NULL OR summary = '') "
                "ORDER BY created DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
             "tier": r[4], "created": r[5]}
            for r in rows
        ]

    def pin_by_tag(self, tag: str, ns: str | None = "default") -> int:
        """Set tier=0 (pinned) on all memories in *ns* that have *tag*.

        Returns the number of memories pinned.  *ns=None* spans all
        namespaces.  Complement to ``set_tier_by_tag`` but always pins
        and more explicit in intent.
        """
        return self.set_tier_by_tag(tag, tier=0, ns=ns)

    def promote_by_access(
        self,
        ns: str,
        n: int = 10,
        from_tier: int = 2,
        to_tier: int = 1,
    ) -> int:
        """Promote the *n* most-accessed memories from *from_tier* to *to_tier*.

        Useful for automatically graduating frequently-referenced ambient
        memories to default tier.  Returns the number of memories promoted.
        *ns* is required.
        """
        if from_tier not in (0, 1, 2) or to_tier not in (0, 1, 2):
            raise ValueError("from_tier and to_tier must each be 0, 1, or 2")
        if from_tier == to_tier:
            return 0
        rows = self._db.execute(
            "SELECT id FROM memories WHERE ns=? AND tier=? "
            "ORDER BY access_count DESC, last_accessed DESC LIMIT ?",
            (ns, from_tier, n),
        ).fetchall()
        if not rows:
            return 0
        ids = [r[0] for r in rows]
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            cur = self._db.execute(
                f"UPDATE memories SET tier=? WHERE id IN ({placeholders})",
                (to_tier, *ids),
            )
            self._db.commit()
        return cur.rowcount

    def age_by_ns(self, ns: str) -> dict:
        """Return memory count broken down by age bracket for *ns*.

        Brackets: today (<1 day), this_week (1-7 days), this_month (7-30
        days), this_quarter (30-90 days), older (>90 days).  Useful for
        dashboards that show how fresh a namespace's contents are.
        """
        row = self._db.execute(
            """
            SELECT
                SUM(CASE WHEN julianday('now') - julianday(created) < 1 THEN 1 ELSE 0 END),
                SUM(CASE WHEN julianday('now') - julianday(created) BETWEEN 1 AND 6.999 THEN 1 ELSE 0 END),
                SUM(CASE WHEN julianday('now') - julianday(created) BETWEEN 7 AND 29.999 THEN 1 ELSE 0 END),
                SUM(CASE WHEN julianday('now') - julianday(created) BETWEEN 30 AND 89.999 THEN 1 ELSE 0 END),
                SUM(CASE WHEN julianday('now') - julianday(created) >= 90 THEN 1 ELSE 0 END),
                COUNT(*)
            FROM memories WHERE ns=?
            """,
            (ns,),
        ).fetchone()
        total = row[5] or 0
        if not total:
            return {"ns": ns, "today": 0, "this_week": 0, "this_month": 0,
                    "this_quarter": 0, "older": 0, "total": 0}
        return {
            "ns": ns,
            "today": row[0] or 0,
            "this_week": row[1] or 0,
            "this_month": row[2] or 0,
            "this_quarter": row[3] or 0,
            "older": row[4] or 0,
            "total": row[5] or 0,
        }

    def delete_by_tier(self, ns: str, tier: int) -> int:
        """Delete all memories of a given *tier* from *ns*.

        Returns the number of rows deleted.  Pinned memories (tier=0) can
        be deleted this way, unlike the GC which protects them.  Irreversible.
        """
        if tier not in (0, 1, 2):
            raise ValueError("tier must be 0 (pinned), 1 (default), or 2 (ambient)")
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM memories WHERE ns=? AND tier=?", (ns, tier)
            )
            n = cur.rowcount
            self._db.commit()
        return n

    def untagged_memories(
        self,
        ns: str | None = "default",
        limit: int = 20,
    ) -> list[dict]:
        """Return memories that have no tags (empty or absent meta.tags array).

        Useful for identifying memories that need curation.  *ns=None* spans
        all namespaces.  Ordered by created DESC.
        """
        # Avoid json_extract on potentially-malformed meta; use LIKE to identify
        # rows that definitely have no tags ("tags" key absent) or possibly have
        # empty tags (post-filtered in Python).
        if ns is not None:
            candidates = self._db.execute(
                "SELECT id, ns, text, summary, tier, created, meta FROM memories "
                "WHERE ns=? ORDER BY created DESC",
                (ns,),
            ).fetchall()
        else:
            candidates = self._db.execute(
                "SELECT id, ns, text, summary, tier, created, meta FROM memories "
                "ORDER BY created DESC",
            ).fetchall()

        import json as _j_utm_filter
        rows = []
        for c in candidates:
            meta_str = c[6]
            try:
                meta_obj = _j_utm_filter.loads(meta_str) if meta_str else {}
                tags = meta_obj.get("tags")
                if tags is None or (isinstance(tags, list) and len(tags) == 0):
                    rows.append(c)
            except Exception:
                rows.append(c)
        rows = rows[:limit]
        return [
            {"id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
             "tier": r[4], "created": r[5]}
            for r in rows
        ]

    def set_meta_for_untagged(
        self,
        ns: str,
        key: str,
        value: object,
        limit: int = 100,
    ) -> int:
        """Set a meta key on memories that currently have no tags in *ns*.

        Useful for bulk-labelling untagged memories with a default category,
        owner, or source marker.  Returns the number of memories updated.
        """
        import json as _j_smfu
        untagged = self.untagged_memories(ns=ns, limit=limit)
        if not untagged:
            return 0
        updated = 0
        with self._lock:
            for row in untagged:
                cur_meta_row = self._db.execute(
                    "SELECT meta FROM memories WHERE id=?", (row["id"],)
                ).fetchone()
                try:
                    meta = _j_smfu.loads(cur_meta_row[0]) if cur_meta_row else {}
                except Exception:
                    meta = {}
                meta[key] = value
                self._db.execute(
                    "UPDATE memories SET meta=? WHERE id=?",
                    (_j_smfu.dumps(meta, ensure_ascii=False), row["id"]),
                )
                updated += 1
            if updated:
                self._db.commit()
        return updated

    def rename_tag(
        self,
        old_tag: str,
        new_tag: str,
        ns: str | None = "default",
    ) -> int:
        """Rename *old_tag* to *new_tag* in every memory's ``meta.tags`` array.

        Only memories that currently have *old_tag* in their tags list are
        updated; memories without it are untouched.  *ns=None* spans all
        namespaces.  Returns the number of memories updated.

        Implementation: fetches affected rows, rewrites the JSON in Python
        (sqlite3 has no native array-element-replace), and bulk-updates back.
        This is intentionally row-by-row rather than a single UPDATE so the
        JSON round-trip is correct for all meta shapes.
        """
        import json as _j_rt

        if not old_tag or not new_tag or old_tag == new_tag:
            return 0

        if ns is not None:
            rows = self._db.execute(
                "SELECT id, meta FROM memories WHERE ns=? AND meta LIKE '%\"tags\"%'",
                (ns,),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT id, meta FROM memories WHERE meta LIKE '%\"tags\"%'"
            ).fetchall()

        updated = 0
        with self._lock:
            for row_id, meta_str in rows:
                try:
                    meta = _j_rt.loads(meta_str)
                except Exception:
                    continue
                tags = meta.get("tags")
                if not isinstance(tags, list) or old_tag not in tags:
                    continue
                meta["tags"] = [new_tag if t == old_tag else t for t in tags]
                self._db.execute(
                    "UPDATE memories SET meta=? WHERE id=?",
                    (_j_rt.dumps(meta, ensure_ascii=False), row_id),
                )
                updated += 1
            if updated:
                self._db.commit()
        return updated

    def find_duplicates(
        self,
        ns: str | None = "default",
        limit: int = 20,
    ) -> list[dict]:
        """Find memories that share identical text within *ns*.

        Returns a list of ``{"text": str, "count": int, "ids": list[int]}``
        dicts, ordered by count descending.  Only groups with ``count ≥ 2``
        are returned.  *ns=None* spans all namespaces.  *limit* caps the
        number of duplicate *groups* returned (not the total memory count).
        """
        if ns is not None:
            rows = self._db.execute(
                "SELECT text, COUNT(*) AS cnt, GROUP_CONCAT(id) AS id_list "
                "FROM memories WHERE ns=? GROUP BY text HAVING cnt >= 2 "
                "ORDER BY cnt DESC LIMIT ?",
                (ns, limit),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT text, COUNT(*) AS cnt, GROUP_CONCAT(id) AS id_list "
                "FROM memories GROUP BY text HAVING cnt >= 2 "
                "ORDER BY cnt DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"text": r[0], "count": r[1], "ids": [int(i) for i in r[2].split(",")]}
            for r in rows
        ]

    def swap_tier(
        self,
        ns: str,
        from_tier: int,
        to_tier: int,
    ) -> int:
        """Bulk-move all memories in *ns* from *from_tier* to *to_tier*.

        Returns the number of memories updated.  *ns* is required (no
        all-namespace variant — a cross-namespace bulk tier flip is almost
        always a mistake).
        """
        if from_tier not in (0, 1, 2) or to_tier not in (0, 1, 2):
            raise ValueError("from_tier and to_tier must each be 0, 1, or 2")
        if from_tier == to_tier:
            return 0
        with self._lock:
            cur = self._db.execute(
                "UPDATE memories SET tier=? WHERE ns=? AND tier=?",
                (to_tier, ns, from_tier),
            )
            self._db.commit()
        return cur.rowcount

    def ns_summary(self, ns: str) -> dict:
        """Return a one-call dashboard dict for *ns*.

        Combines count, tier breakdown, text-length stats, access stats, and
        date range into a single response suitable for a CLI ``info`` command
        or a dashboard widget.  Returns an empty summary dict (all zeroes) for
        namespaces that do not exist or have no memories.
        """
        row = self._db.execute(
            "SELECT COUNT(*), "
            "  SUM(CASE WHEN tier=0 THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN tier=1 THEN 1 ELSE 0 END), "
            "  SUM(CASE WHEN tier=2 THEN 1 ELSE 0 END), "
            "  AVG(LENGTH(text)), MIN(LENGTH(text)), MAX(LENGTH(text)), "
            "  AVG(access_count), MAX(access_count), "
            "  SUM(CASE WHEN access_count=0 THEN 1 ELSE 0 END), "
            "  MIN(created), MAX(created) "
            "FROM memories WHERE ns=?",
            (ns,),
        ).fetchone()
        if not row or row[0] == 0:
            return {
                "ns": ns, "count": 0,
                "tiers": {"pinned": 0, "default": 0, "ambient": 0},
                "avg_chars": 0, "min_chars": 0, "max_chars": 0,
                "avg_accesses": 0.0, "max_accesses": 0, "never_accessed": 0,
                "oldest": None, "newest": None,
            }
        return {
            "ns": ns,
            "count": row[0] or 0,
            "tiers": {
                "pinned": row[1] or 0,
                "default": row[2] or 0,
                "ambient": row[3] or 0,
            },
            "avg_chars": round(row[4] or 0, 1),
            "min_chars": row[5] or 0,
            "max_chars": row[6] or 0,
            "avg_accesses": round(row[7] or 0, 2),
            "max_accesses": row[8] or 0,
            "never_accessed": row[9] or 0,
            "oldest": row[10],
            "newest": row[11],
        }

    def search_text(
        self,
        query: str,
        ns: str | None = "default",
        limit: int = 20,
    ) -> list[dict]:
        """Case-insensitive LIKE search on the *text* field.

        Complements :meth:`search_by_summary` which searches summaries.
        *ns=None* spans all namespaces.  Returns dicts with ``id``, ``ns``,
        ``text``, ``summary``, ``tier``, and ``created``.
        """
        pattern = f"%{query}%"
        if ns is not None:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created FROM memories "
                "WHERE ns=? AND text LIKE ? ORDER BY created DESC LIMIT ?",
                (ns, pattern, limit),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created FROM memories "
                "WHERE text LIKE ? ORDER BY created DESC LIMIT ?",
                (pattern, limit),
            ).fetchall()

        return [
            {"id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
             "tier": r[4], "created": r[5]}
            for r in rows
        ]

    def count_by_ns(self) -> dict[str, int]:
        """Return a ``{namespace: count}`` mapping for all namespaces.

        Useful for dashboards and health checks.  Empty namespaces (i.e.
        those that exist in hnswlib but have no SQL rows) are not included.
        """
        rows = self._db.execute(
            "SELECT ns, COUNT(*) FROM memories GROUP BY ns ORDER BY ns"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def clear_ns(self, ns: str) -> int:
        """Delete **all** memories from *ns*.

        Returns the number of rows deleted.  Does NOT purge the hnswlib
        index file — call ``reindex_all(ns)`` afterwards if you plan to
        re-populate the same namespace.  Irreversible.
        """
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM memories WHERE ns=?", (ns,)
            )
            n = cur.rowcount
            self._db.commit()
        return n

    def copy_to_ns(
        self,
        memory_ids: list[int],
        dst_ns: str,
    ) -> int:
        """Copy memories to *dst_ns* without removing them from the source.

        Each copied memory gets a fresh ``id`` and ``created`` timestamp in
        *dst_ns*.  The hnswlib vector for each memory is zero-filled in the
        destination index — call ``reindex_all(dst_ns)`` to rebuild accurate
        vectors after copying.  Returns the number of rows successfully copied.
        """
        import json as _j_ctn

        rows = self._db.execute(
            f"SELECT text, summary, meta, tier "
            f"FROM memories WHERE id IN ({','.join('?' * len(memory_ids))})",
            memory_ids,
        ).fetchall() if memory_ids else []

        if not rows:
            return 0

        import numpy as _np_ctn

        texts = [r[0] for r in rows]
        summaries = [r[1] for r in rows]
        metas = [_j_ctn.loads(r[2]) if r[2] else {} for r in rows]
        tiers = [r[3] for r in rows]

        zero_vecs = _np_ctn.zeros((len(rows), self.dim), dtype=_np_ctn.float32)
        new_ids = self.add(
            texts,
            zero_vecs,
            ns=dst_ns,
            meta=metas,
            summaries=summaries,
        )
        # Apply tiers
        for new_id, tier in zip(new_ids, tiers):
            if tier != 1:
                self._db.execute(
                    "UPDATE memories SET tier=? WHERE id=?", (tier, new_id)
                )
        self._db.commit()
        return len(new_ids)

    def list_by_tier(
        self,
        tier: int,
        ns: str | None = "default",
        limit: int = 100,
    ) -> list[dict]:
        """Return memories with the given *tier*, ordered by ``created DESC``.

        *ns=None* spans all namespaces.  Returns dicts with ``id``, ``ns``,
        ``text``, ``summary``, ``tier``, ``created``, and parsed ``meta``.
        """
        import json as _j_lbt

        if ns is not None:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created, meta "
                "FROM memories WHERE ns=? AND tier=? ORDER BY created DESC LIMIT ?",
                (ns, tier, limit),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created, meta "
                "FROM memories WHERE tier=? ORDER BY created DESC LIMIT ?",
                (tier, limit),
            ).fetchall()

        return [
            {"id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
             "tier": r[4], "created": r[5],
             "meta": _j_lbt.loads(r[6]) if r[6] else {}}
            for r in rows
        ]

    def recent(
        self,
        ns: str | None = "default",
        n: int = 10,
    ) -> list[dict]:
        """Return the *n* most recently created memories.

        *ns=None* spans all namespaces.  Returns dicts with ``id``, ``ns``,
        ``text``, ``summary``, ``tier``, and ``created``.
        """
        if ns is not None:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created FROM memories "
                "WHERE ns=? ORDER BY created DESC LIMIT ?",
                (ns, n),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created FROM memories "
                "ORDER BY created DESC LIMIT ?",
                (n,),
            ).fetchall()

        return [
            {"id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
             "tier": r[4], "created": r[5]}
            for r in rows
        ]

    def oldest(
        self,
        ns: str | None = "default",
        n: int = 10,
    ) -> list[dict]:
        """Return the *n* oldest memories (``created ASC``).

        *ns=None* spans all namespaces.  Same dict shape as :meth:`recent`.
        """
        if ns is not None:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created FROM memories "
                "WHERE ns=? ORDER BY created ASC LIMIT ?",
                (ns, n),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created FROM memories "
                "ORDER BY created ASC LIMIT ?",
                (n,),
            ).fetchall()

        return [
            {"id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
             "tier": r[4], "created": r[5]}
            for r in rows
        ]

    def replace_text(self, memory_id: int, new_text: str) -> bool:
        """Replace the text field of a memory in-place.

        The hnswlib vector embedding is NOT updated — this is a lightweight
        SQL-only operation suitable for fixing typos or metadata corrections
        where re-embedding is not required.  Returns ``True`` on success,
        ``False`` if *memory_id* does not exist.
        """
        row = self._db.execute(
            "SELECT id FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if row is None:
            return False
        with self._lock:
            self._db.execute(
                "UPDATE memories SET text=? WHERE id=?",
                (new_text, memory_id),
            )
            self._db.commit()
        return True

    def set_tier_by_tag(
        self,
        tag: str,
        tier: int,
        ns: str | None = "default",
    ) -> int:
        """Set *tier* on every memory whose meta.tags list contains *tag*.

        *ns=None* spans all namespaces.  Returns the number of rows updated.
        The tag match uses SQLite ``json_each`` so partial tag names are
        *not* matched.
        """
        if ns is not None:
            cur = self._db.execute(
                "UPDATE memories SET tier=? "
                "WHERE ns=? AND EXISTS ("
                "  SELECT 1 FROM json_each(json_extract(meta,'$.tags')) "
                "  WHERE value=?)",
                (tier, ns, tag),
            )
        else:
            cur = self._db.execute(
                "UPDATE memories SET tier=? "
                "WHERE EXISTS ("
                "  SELECT 1 FROM json_each(json_extract(meta,'$.tags')) "
                "  WHERE value=?)",
                (tier, tag),
            )
        n = cur.rowcount
        with self._lock:
            self._db.commit()
        return n

    def rotate_ns(
        self,
        src_ns: str,
        dst_ns: str,
        limit: int = 100,
        tier: int | None = None,
    ) -> int:
        """Move the oldest *limit* memories from *src_ns* to *dst_ns*.

        Optionally filter by *tier*.  Returns the number of rows moved.
        This is a pure SQL operation — the hnswlib index is NOT updated for
        rotated memories (they remain searchable in the old index until
        ``reindex_all`` is called).
        """
        import json as _j_rns

        if tier is not None:
            rows = self._db.execute(
                "SELECT id FROM memories WHERE ns=? AND tier=? ORDER BY created ASC LIMIT ?",
                (src_ns, tier, limit),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT id FROM memories WHERE ns=? ORDER BY created ASC LIMIT ?",
                (src_ns, limit),
            ).fetchall()

        if not rows:
            return 0

        ids = [r[0] for r in rows]
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            self._db.execute(
                f"UPDATE memories SET ns=? WHERE id IN ({placeholders})",
                (dst_ns, *ids),
            )
            self._db.commit()
        return len(ids)

    def compact_meta(
        self,
        ns: str | None = "default",
        keep_keys: list[str] | None = None,
    ) -> int:
        """Remove all meta keys except those in *keep_keys* (or all if None).

        *ns=None* spans all namespaces.  *keep_keys=None* strips meta
        entirely (sets to ``{}``).  Returns the number of rows updated.
        Useful for reducing storage after bulk imports that carried
        unwanted meta fields.
        """
        import json as _j_cm

        if ns is not None:
            rows = self._db.execute(
                "SELECT id, meta FROM memories WHERE ns=?", (ns,)
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT id, meta FROM memories"
            ).fetchall()

        updated = 0
        with self._lock:
            for row_id, raw_meta in rows:
                meta = _j_cm.loads(raw_meta) if raw_meta else {}
                if keep_keys is None:
                    new_meta: dict = {}
                else:
                    new_meta = {k: v for k, v in meta.items() if k in keep_keys}
                if new_meta != meta:
                    self._db.execute(
                        "UPDATE memories SET meta=? WHERE id=?",
                        (_j_cm.dumps(new_meta, ensure_ascii=False), row_id),
                    )
                    updated += 1
            self._db.commit()
        return updated

    def search_by_summary(
        self,
        query: str,
        ns: str | None = "default",
        limit: int = 20,
    ) -> list[dict]:
        """Full-text search on the summary field using SQLite ``LIKE``.

        Returns memories whose ``summary`` contains *query* (case-insensitive).
        *ns=None* spans all namespaces.  Each result dict has ``id``, ``ns``,
        ``text``, ``summary``, ``tier``, and ``created``.
        """
        pattern = f"%{query}%"
        if ns is not None:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created FROM memories "
                "WHERE ns=? AND summary LIKE ? ORDER BY created DESC LIMIT ?",
                (ns, pattern, limit),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created FROM memories "
                "WHERE summary LIKE ? ORDER BY created DESC LIMIT ?",
                (pattern, limit),
            ).fetchall()

        return [
            {"id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
             "tier": r[4], "created": r[5]}
            for r in rows
        ]

    def filter_by_meta(
        self,
        key: str,
        value: object,
        ns: str | None = "default",
        limit: int = 100,
    ) -> list[dict]:
        """Return memories whose ``meta[key] == value``.

        Uses SQLite's ``json_extract`` so no Python-side JSON parsing is
        needed per row.  *ns=None* spans all namespaces.  Returns at most
        *limit* results ordered by ``created DESC``.  Each result dict
        contains ``id``, ``ns``, ``text``, ``summary``, ``tier``,
        ``created``, and ``meta`` (as parsed dict).
        """
        import json as _j_fbm

        json_path = f"$.{key}"
        if ns is not None:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created, meta "
                "FROM memories WHERE ns=? AND json_extract(meta, ?) = ? "
                "ORDER BY created DESC LIMIT ?",
                (ns, json_path, value, limit),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT id, ns, text, summary, tier, created, meta "
                "FROM memories WHERE json_extract(meta, ?) = ? "
                "ORDER BY created DESC LIMIT ?",
                (json_path, value, limit),
            ).fetchall()

        return [
            {
                "id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
                "tier": r[4], "created": r[5],
                "meta": _j_fbm.loads(r[6]) if r[6] else {},
            }
            for r in rows
        ]

    def summary_stats(self, ns: str | None = "default") -> dict:
        """Return summary-field coverage statistics for a namespace.

        Keys: ``total``, ``with_summary``, ``without_summary``,
        ``pct_with_summary`` (float 0-100).  *ns=None* spans all
        namespaces.
        """
        if ns is not None:
            row = self._db.execute(
                "SELECT COUNT(*), SUM(CASE WHEN summary IS NOT NULL AND summary != '' THEN 1 ELSE 0 END) "
                "FROM memories WHERE ns=?",
                (ns,),
            ).fetchone()
        else:
            row = self._db.execute(
                "SELECT COUNT(*), SUM(CASE WHEN summary IS NOT NULL AND summary != '' THEN 1 ELSE 0 END) "
                "FROM memories"
            ).fetchone()

        total = int(row[0]) if row[0] else 0
        with_s = int(row[1]) if row[1] else 0
        without_s = total - with_s
        pct = round(with_s / total * 100, 2) if total > 0 else 0.0
        return {
            "total": total,
            "with_summary": with_s,
            "without_summary": without_s,
            "pct_with_summary": pct,
        }

    def bulk_delete(self, memory_ids: list[int]) -> int:
        """Delete multiple memories by ID in a single transaction.

        Vectors are marked deleted in the hnswlib index for each namespace
        touched.  Returns the count of rows actually deleted from the DB.
        """
        if not memory_ids:
            return 0

        # Determine which namespace each id belongs to (for index cleanup)
        placeholders = ",".join("?" * len(memory_ids))
        rows = self._db.execute(
            f"SELECT id, ns FROM memories WHERE id IN ({placeholders})",
            memory_ids,
        ).fetchall()
        if not rows:
            return 0

        from collections import defaultdict

        by_ns: dict[str, list[int]] = defaultdict(list)
        for mid, ns in rows:
            by_ns[ns].append(mid)

        with self._lock:
            self._db.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})",
                memory_ids,
            )
            self._db.commit()

        # Mark deleted in each namespace's index
        for ns, ids in by_ns.items():
            try:
                idx = self._index_for(ns)
                for mid in ids:
                    try:
                        idx.mark_deleted(mid)
                    except Exception:
                        pass
                self._save_index(ns)
            except Exception:
                pass

        return len(rows)

    def import_records(
        self,
        records: list[dict],
        *,
        ns_override: str | None = None,
        vecs: "list | None" = None,
    ) -> int:
        """Import records produced by :meth:`export_ns` (or hand-crafted dicts).

        Each record must have at least ``text``.  Optional keys: ``ns``,
        ``summary``, ``meta``, ``tier``.  If *ns_override* is given it
        overrides the ``ns`` field in every record.  If *vecs* is given it
        must be a list of numpy arrays (one per record); otherwise records
        are stored with a zero vector (suitable for keyword-only import).

        Returns the count of successfully imported records.
        """
        import numpy as np

        zero = np.zeros(self.dim, dtype=np.float32)
        imported = 0
        for i, rec in enumerate(records):
            text = rec.get("text")
            if not text:
                continue
            ns_val = ns_override or rec.get("ns", "default")
            summary = rec.get("summary")
            meta_raw = rec.get("meta", {})
            meta_dict = meta_raw if isinstance(meta_raw, dict) else {}
            tier = int(rec.get("tier", 1))
            vec = vecs[i] if vecs is not None else zero
            self.add([text], [vec], ns=ns_val, meta=[meta_dict],
                     summaries=[summary], tier=tier)
            imported += 1
        return imported

    def text_stats(self, ns: str | None = "default") -> dict:
        """Return text-length and word-count statistics for a namespace.

        Keys: ``count``, ``total_chars``, ``avg_chars``, ``min_chars``,
        ``max_chars``, ``total_words``, ``avg_words``.  *ns=None* spans
        all namespaces.  Returns zeroed stats for an empty namespace.
        """
        import re as _re_ts

        if ns is not None:
            rows = self._db.execute(
                "SELECT text FROM memories WHERE ns=?", (ns,)
            ).fetchall()
        else:
            rows = self._db.execute("SELECT text FROM memories").fetchall()

        texts = [r[0] for r in rows if r[0]]
        if not texts:
            return {"count": 0, "total_chars": 0, "avg_chars": 0.0,
                    "min_chars": 0, "max_chars": 0, "total_words": 0, "avg_words": 0.0}

        char_lengths = [len(t) for t in texts]
        word_counts = [len(_re_ts.split(r"\s+", t.strip())) for t in texts]
        n = len(texts)
        return {
            "count": n,
            "total_chars": sum(char_lengths),
            "avg_chars": round(sum(char_lengths) / n, 2),
            "min_chars": min(char_lengths),
            "max_chars": max(char_lengths),
            "total_words": sum(word_counts),
            "avg_words": round(sum(word_counts) / n, 2),
        }

    def touch(self, memory_id: int) -> bool:
        """Update ``last_accessed`` to the current UTC time without changing ``access_count``.

        Useful for "mark as viewed" semantics where the caller does not want
        to count an algorithmic access.  Returns ``True`` on success,
        ``False`` if *memory_id* does not exist.
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            cur = self._db.execute(
                "UPDATE memories SET last_accessed=? WHERE id=?", (now, memory_id)
            )
            self._db.commit()
        return cur.rowcount > 0

    def bulk_untag(self, memory_ids: list[int], tags: list[str]) -> int:
        """Remove one or more *tags* from multiple memories at once.

        Each tag is removed from every memory in *memory_ids*.  IDs or tags
        that are absent are silently skipped.  Returns the count of memories
        that were actually updated.
        """
        import json as _j_butag
        updated = 0
        for mid in memory_ids:
            row = self._db.execute(
                "SELECT meta FROM memories WHERE id=?", (mid,)
            ).fetchone()
            if row is None:
                continue
            meta = _j_butag.loads(row[0]) if row[0] else {}
            existing: list = meta.get("tags", [])
            new_list = [t for t in existing if t not in tags]
            if new_list != existing:
                meta["tags"] = new_list
                with self._lock:
                    self._db.execute(
                        "UPDATE memories SET meta=? WHERE id=?",
                        (_j_butag.dumps(meta, ensure_ascii=False), mid),
                    )
                    self._db.commit()
            updated += 1
        return updated

    def count_by_tier(self, ns: str | None = "default") -> dict[int, int]:
        """Return a ``{tier: count}`` dict for *ns* (or all namespaces when *ns* is ``None``).

        Tiers not present in the namespace are omitted from the result.
        """
        if ns is not None:
            rows = self._db.execute(
                "SELECT tier, COUNT(*) FROM memories WHERE ns=? GROUP BY tier", (ns,)
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT tier, COUNT(*) FROM memories GROUP BY tier"
            ).fetchall()
        return {int(r[0]): int(r[1]) for r in rows}

    def export_ns(self, ns: str) -> list[dict]:
        """Export all memories in *ns* as a list of dicts (no vectors).

        Each dict has ``id``, ``ns``, ``text``, ``summary``, ``meta``
        (as parsed dict, not JSON string), ``tier``, ``created``,
        ``last_accessed``, and ``access_count``.  Suitable for JSON
        serialisation with ``json.dumps``.
        """
        import json as _j_exp
        rows = self._db.execute(
            "SELECT id, ns, text, summary, meta, tier, created, last_accessed, access_count "
            "FROM memories WHERE ns=? ORDER BY id",
            (ns,),
        ).fetchall()
        return [
            {
                "id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
                "meta": _j_exp.loads(r[4]) if r[4] else {},
                "tier": r[5], "created": r[6],
                "last_accessed": r[7], "access_count": r[8],
            }
            for r in rows
        ]

    def bulk_tag(self, memory_ids: list[int], tags: list[str]) -> int:
        """Add one or more *tags* to multiple memories at once.

        Each tag is added to every memory in *memory_ids*.  IDs that do
        not exist are silently skipped.  Returns the count of memories that
        were actually updated (had at least one new tag added or confirmed).
        """
        import json as _j_btag
        updated = 0
        for mid in memory_ids:
            row = self._db.execute(
                "SELECT meta FROM memories WHERE id=?", (mid,)
            ).fetchone()
            if row is None:
                continue
            meta = _j_btag.loads(row[0]) if row[0] else {}
            existing: list = meta.get("tags", [])
            new_tags = [t for t in tags if t not in existing]
            existing.extend(new_tags)
            meta["tags"] = existing
            with self._lock:
                self._db.execute(
                    "UPDATE memories SET meta=? WHERE id=?",
                    (_j_btag.dumps(meta, ensure_ascii=False), mid),
                )
                self._db.commit()
            updated += 1
        return updated

    def get_tags(self, memory_id: int) -> list[str] | None:
        """Return the ``tags`` list from a memory's meta JSON.

        Returns an empty list if the memory exists but has no tags, or
        ``None`` if *memory_id* does not exist.
        """
        import json as _j_gt
        row = self._db.execute(
            "SELECT meta FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        meta = _j_gt.loads(row[0]) if row[0] else {}
        return list(meta.get("tags", []))

    def search_date_range(
        self,
        ns: str | None = "default",
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
        tier: int | None = None,
    ) -> list[dict]:
        """Return memories whose ``created`` timestamp falls within a date range.

        *after* and *before* are ISO-8601 strings (``YYYY-MM-DD`` or full
        datetime).  Both are optional — omitting both returns everything in
        *ns* up to *limit*.  When *ns* is ``None`` the search spans all
        namespaces.  Optionally filter by *tier*.
        """
        limit = max(1, min(int(limit), 1000))
        clauses: list[str] = []
        params: list = []
        if ns is not None:
            clauses.append("ns=?")
            params.append(ns)
        if after is not None:
            clauses.append("created >= ?")
            params.append(after)
        if before is not None:
            clauses.append("created <= ?")
            params.append(before)
        if tier is not None:
            clauses.append("tier=?")
            params.append(int(tier))
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self._db.execute(
            f"SELECT id, ns, text, summary, tier, created "
            f"FROM memories {where} ORDER BY created DESC LIMIT ?",
            params,
        ).fetchall()
        return [
            {"id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
             "tier": r[4], "created": r[5]}
            for r in rows
        ]

    def word_frequency(self, ns: str | None = "default", top_n: int = 20) -> list[dict]:
        """Return the most frequent words across all memory texts in *ns*.

        Tokenises each ``text`` column value by splitting on whitespace and
        stripping punctuation.  Stop words are not removed — callers may
        filter the result themselves.  Returns a list of
        ``{"word": str, "count": int}`` dicts sorted by count descending,
        capped at *top_n* (max 500).
        """
        import re as _re_wf
        top_n = max(1, min(int(top_n), 500))
        if ns is not None:
            rows = self._db.execute(
                "SELECT text FROM memories WHERE ns=?", (ns,)
            ).fetchall()
        else:
            rows = self._db.execute("SELECT text FROM memories").fetchall()
        freq: dict[str, int] = {}
        for (text,) in rows:
            for word in _re_wf.split(r"[\s\W]+", text.lower()):
                if len(word) >= 2:
                    freq[word] = freq.get(word, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
        return [{"word": w, "count": c} for w, c in sorted_words[:top_n]]

    def find_by_tag(self, tag: str, ns: str | None = "default", limit: int = 100) -> list[dict]:
        """Return memories whose meta JSON contains *tag* in the ``tags`` list.

        Uses SQLite's ``json_each`` to scan the ``meta.tags`` array.
        When *ns* is ``None`` the search spans all namespaces.
        """
        limit = max(1, min(int(limit), 1000))
        if ns is not None:
            rows = self._db.execute(
                "SELECT m.id, m.ns, m.text, m.summary, m.tier, m.created "
                "FROM memories m, json_each(json_extract(m.meta, '$.tags')) je "
                "WHERE m.ns=? AND je.value=? LIMIT ?",
                (ns, tag, limit),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT m.id, m.ns, m.text, m.summary, m.tier, m.created "
                "FROM memories m, json_each(json_extract(m.meta, '$.tags')) je "
                "WHERE je.value=? LIMIT ?",
                (tag, limit),
            ).fetchall()
        return [
            {"id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
             "tier": r[4], "created": r[5]}
            for r in rows
        ]

    def list_tags(self, ns: str | None = "default") -> list[dict]:
        """Return all distinct tags used in *ns* with their occurrence counts.

        Returns a list of ``{"tag": str, "count": int}`` dicts sorted by count
        descending. When *ns* is ``None`` the query spans all namespaces.
        """
        if ns is not None:
            rows = self._db.execute(
                "SELECT je.value, COUNT(*) as cnt "
                "FROM memories m, json_each(json_extract(m.meta, '$.tags')) je "
                "WHERE m.ns=? "
                "GROUP BY je.value ORDER BY cnt DESC",
                (ns,),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT je.value, COUNT(*) as cnt "
                "FROM memories m, json_each(json_extract(m.meta, '$.tags')) je "
                "GROUP BY je.value ORDER BY cnt DESC",
            ).fetchall()
        return [{"tag": r[0], "count": int(r[1])} for r in rows]

    def tag(self, memory_id: int, tag: str) -> bool:
        """Add *tag* to the ``tags`` list inside a memory's meta JSON.

        Creates the ``tags`` key if absent. Idempotent — adding a tag that
        already exists is a no-op. Returns ``True`` on success, ``False`` if
        *memory_id* does not exist.
        """
        import json as _j_tag
        row = self._db.execute(
            "SELECT meta FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if row is None:
            return False
        meta = _j_tag.loads(row[0]) if row[0] else {}
        tags: list = meta.get("tags", [])
        if tag not in tags:
            tags.append(tag)
            meta["tags"] = tags
            with self._lock:
                self._db.execute(
                    "UPDATE memories SET meta=? WHERE id=?",
                    (_j_tag.dumps(meta, ensure_ascii=False), memory_id),
                )
                self._db.commit()
        return True

    def untag(self, memory_id: int, tag: str) -> bool:
        """Remove *tag* from the ``tags`` list inside a memory's meta JSON.

        Idempotent — removing a tag that is absent is a no-op. Returns
        ``True`` on success, ``False`` if *memory_id* does not exist.
        """
        import json as _j_untag
        row = self._db.execute(
            "SELECT meta FROM memories WHERE id=?", (memory_id,)
        ).fetchone()
        if row is None:
            return False
        meta = _j_untag.loads(row[0]) if row[0] else {}
        tags: list = meta.get("tags", [])
        if tag in tags:
            tags.remove(tag)
            meta["tags"] = tags
            with self._lock:
                self._db.execute(
                    "UPDATE memories SET meta=? WHERE id=?",
                    (_j_untag.dumps(meta, ensure_ascii=False), memory_id),
                )
                self._db.commit()
        return True

    def access_stats(self, ns: str | None = "default") -> dict:
        """Return access-count and last-accessed statistics for *ns*.

        When *ns* is ``None`` the query spans all namespaces.  The returned
        dict has the keys ``ns``, ``total``, ``total_accesses``,
        ``avg_accesses``, ``max_accesses``, ``never_accessed`` (count with
        access_count == 0), and ``most_recent_access`` (ISO string or None).
        """
        clause = "WHERE ns=?" if ns is not None else ""
        params: tuple = (ns,) if ns is not None else ()
        row = self._db.execute(
            f"SELECT COUNT(*), SUM(access_count), AVG(access_count), "
            f"MAX(access_count), MAX(last_accessed), "
            f"SUM(CASE WHEN access_count=0 THEN 1 ELSE 0 END) "
            f"FROM memories {clause}",
            params,
        ).fetchone()
        total = int(row[0]) if row[0] else 0
        return {
            "ns": ns,
            "total": total,
            "total_accesses": int(row[1]) if row[1] is not None else 0,
            "avg_accesses": round(float(row[2]), 3) if row[2] is not None else 0.0,
            "max_accesses": int(row[3]) if row[3] is not None else 0,
            "most_recent_access": row[4],
            "never_accessed": int(row[5]) if row[5] is not None else 0,
        }

    def namespace_info(self, ns: str) -> dict | None:
        """Return a summary dict for a single namespace, or None if it doesn't exist.

        Fields returned:
        - ``ns``: namespace name
        - ``total``: total memory count
        - ``by_tier``: {tier_int: count, ...}
        - ``oldest``: ISO timestamp of the earliest created row
        - ``newest``: ISO timestamp of the most recent created row
        - ``avg_text_len``: average character length of the ``text`` field
        - ``total_words``: approximate total word count (space-split)
        - ``with_summary``: count of memories that have a non-null summary
        """
        row = self._db.execute(
            "SELECT COUNT(*), MIN(created), MAX(created), "
            "AVG(LENGTH(text)), SUM(LENGTH(text) - LENGTH(REPLACE(text, ' ', '')) + 1), "
            "SUM(CASE WHEN summary IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM memories WHERE ns=?",
            (ns,),
        ).fetchone()
        if row is None or row[0] == 0:
            return None
        total, oldest, newest, avg_len, total_words, with_summary = row
        tier_rows = self._db.execute(
            "SELECT tier, COUNT(*) FROM memories WHERE ns=? GROUP BY tier", (ns,)
        ).fetchall()
        by_tier = {int(t): int(c) for t, c in tier_rows}
        return {
            "ns": ns,
            "total": int(total),
            "by_tier": by_tier,
            "oldest": oldest,
            "newest": newest,
            "avg_text_len": round(float(avg_len), 1) if avg_len is not None else 0.0,
            "total_words": int(total_words) if total_words is not None else 0,
            "with_summary": int(with_summary),
        }

    def sample(
        self,
        ns: str = "default",
        n: int = 5,
        tier: int | None = None,
    ) -> list[dict]:
        """Return up to *n* randomly sampled memories from *ns*.

        Uses SQLite's ``ORDER BY RANDOM()`` — not suitable for large-n stats
        but perfect for spot-checking a namespace or building random review
        queues. Pass *tier* to restrict to a specific tier.
        """
        where = "ns = ?"
        params: list = [ns]
        if tier is not None:
            where += " AND tier = ?"
            params.append(tier)
        params.append(max(1, n))
        rows = self._db.execute(
            f"SELECT id, ns, text, summary, tier, created, last_accessed, access_count "
            f"FROM memories WHERE {where} ORDER BY RANDOM() LIMIT ?",
            params,
        ).fetchall()
        return [
            {
                "id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
                "tier": r[4], "created": r[5], "last_accessed": r[6],
                "access_count": r[7],
            }
            for r in rows
        ]

    def update_summary(self, memory_id: int, summary: str | None) -> bool:
        with self._lock:
            cur = self._db.execute(
                "UPDATE memories SET summary=? WHERE id=?", (summary, memory_id)
            )
            self._db.commit()
        return cur.rowcount > 0

    def bulk_update_summary(self, updates: dict[int, str | None]) -> int:
        """Update summary for multiple memories in one transaction.

        ``updates`` maps memory_id → summary (str or None to clear).
        Returns the count of rows actually updated (missing IDs are silently
        skipped).
        """
        if not updates:
            return 0
        updated = 0
        with self._lock:
            for mid, summary in updates.items():
                cur = self._db.execute(
                    "UPDATE memories SET summary=? WHERE id=?", (summary, int(mid))
                )
                updated += cur.rowcount
            self._db.commit()
        return updated

    def search_by_meta(
        self,
        filters: dict,
        ns: str = "default",
        limit: int = 100,
    ) -> list[dict]:
        """Return memories where meta matches all key=value pairs (AND logic).

        Uses SQLite json_extract for scalar values (str, int, float, bool, None).
        Nested keys are supported via dot notation: filters={"a.b": 1} maps to
        json_extract(meta, '$.a.b') — standard JSONPath.
        """
        if not filters:
            return []
        where_parts = ["ns = ?"]
        params: list[Any] = [ns]
        for key, value in filters.items():
            where_parts.append("json_extract(meta, ?) = ?")
            params.extend([f"$.{key}", value])
        where = " AND ".join(where_parts)
        params.append(limit)
        with self._lock:
            rows = self._db.execute(
                f"SELECT id, ns, text, summary, meta, created, tier, last_accessed, access_count "
                f"FROM memories WHERE {where} LIMIT ?",
                params,
            ).fetchall()
        return [
            {
                "id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
                "meta": json.loads(r[4]), "created": r[5],
                "tier": r[6], "last_accessed": r[7], "access_count": r[8],
            }
            for r in rows
        ]

    def update_meta(self, memory_id: int, meta: dict, merge: bool = False) -> bool:
        """Update (or merge into) the metadata for a memory.

        If *merge=True*, provided keys are merged into the existing meta dict —
        existing keys not mentioned are preserved. If *merge=False* (default,
        preserved for backward compatibility), the whole meta dict is replaced.
        Returns True if the row was found and updated, False if the ID does not exist.
        """
        if merge:
            row = self._db.execute("SELECT meta FROM memories WHERE id=?", (memory_id,)).fetchone()
            if row is None:
                return False
            existing = json.loads(row[0]) if row[0] else {}
            existing.update(meta)
            new_meta = json.dumps(existing, ensure_ascii=False)
        else:
            new_meta = json.dumps(meta, ensure_ascii=False)
        with self._lock:
            cur = self._db.execute(
                "UPDATE memories SET meta=? WHERE id=?",
                (new_meta, memory_id),
            )
            self._db.commit()
        return cur.rowcount > 0

    def update_tier_many(self, memory_ids: list[int], tier: int) -> int:
        if tier not in (0, 1, 2):
            raise ValueError("tier must be 0 (pinned), 1 (default), or 2 (ambient)")
        if not memory_ids:
            return 0
        placeholders = ",".join("?" * len(memory_ids))
        with self._lock:
            cur = self._db.execute(
                f"UPDATE memories SET tier=? WHERE id IN ({placeholders})",
                (tier, *memory_ids),
            )
            self._db.commit()
        return cur.rowcount

    def list_memories(
        self, ns: str = "default", limit: int = 20, offset: int = 0,
        tier: int | None = None,
        since: str | None = None,
        before: str | None = None,
    ) -> list[dict]:
        """List memories in ns, newest first.

        since/before: ISO date strings (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        for date-range filtering. Both can be combined for a half-open interval.
        """
        where: list[str] = ["ns=?"]
        params: list = [ns]
        if tier is not None:
            where.append("tier=?")
            params.append(tier)
        if since is not None:
            where.append("created >= ?")
            params.append(since)
        if before is not None:
            where.append("created < ?")
            params.append(before)
        where_clause = " AND ".join(where)
        params.extend([limit, offset])
        rows = self._db.execute(
            f"SELECT id, ns, text, summary, tier, created, last_accessed, access_count "
            f"FROM memories WHERE {where_clause} ORDER BY id DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [
            {
                "id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
                "tier": r[4], "created": r[5], "last_accessed": r[6],
                "access_count": r[7],
            }
            for r in rows
        ]

    def text_search(
        self,
        query: str,
        ns: str | None = "default",
        tier: int | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Case-insensitive substring search over text and summary columns.

        Faster than BM25 for exact keyword lookups; no tokenization, no FTS
        index required. Returns rows ordered newest-first.
        """
        pattern = f"%{query}%"
        where: list[str] = ["(text LIKE ? OR summary LIKE ?)"]
        params: list = [pattern, pattern]
        if ns is not None:
            where.append("ns=?")
            params.append(ns)
        if tier is not None:
            where.append("tier=?")
            params.append(tier)
        params.append(limit)
        rows = self._db.execute(
            f"SELECT id, ns, text, summary, tier, created, last_accessed, access_count "
            f"FROM memories WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT ?",
            params,
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

    def get_many(self, memory_ids: list[int]) -> list[dict]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" * len(memory_ids))
        rows = self._db.execute(
            f"SELECT id, ns, text, summary, tier, created, last_accessed, access_count "
            f"FROM memories WHERE id IN ({placeholders})",
            memory_ids,
        ).fetchall()
        by_id = {
            r[0]: {
                "id": r[0], "ns": r[1], "text": r[2], "summary": r[3],
                "tier": r[4], "created": r[5], "last_accessed": r[6],
                "access_count": r[7],
            }
            for r in rows
        }
        return [by_id[i] for i in memory_ids if i in by_id]

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

    def delete_many(self, memory_ids: list[int]) -> int:
        """Delete multiple memories by ID. Returns count of actually deleted rows.

        Uses a single DELETE … WHERE id IN (…) per namespace to keep the
        per-NS hnswlib index consistent. Missing IDs are silently skipped.
        """
        if not memory_ids:
            return 0
        placeholders = ",".join("?" * len(memory_ids))
        with self._lock:
            ns_rows = self._db.execute(
                f"SELECT id, ns FROM memories WHERE id IN ({placeholders})",
                memory_ids,
            ).fetchall()
            if not ns_rows:
                return 0
            by_ns: dict[str, list[int]] = {}
            for mid, ns in ns_rows:
                by_ns.setdefault(ns, []).append(mid)
            found_ids = [r[0] for r in ns_rows]
            found_ph = ",".join("?" * len(found_ids))
            cur = self._db.execute(
                f"DELETE FROM memories WHERE id IN ({found_ph})", found_ids
            )
            self._db.commit()
        deleted = cur.rowcount
        for ns, ids in by_ns.items():
            with self._ns_file_lock(ns, exclusive=True), self._lock:
                try:
                    idx = self._index_for(ns)
                    for mid in ids:
                        try:
                            idx.mark_deleted(mid)
                        except Exception:  # pragma: no cover
                            pass
                    idx_path = self.root / f"index_{ns}.bin"
                    idx.save_index(str(idx_path))
                    self._index_mtime[ns] = idx_path.stat().st_mtime
                except Exception:  # pragma: no cover
                    pass
        return deleted

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

    def reindex_all(self) -> list[dict]:
        """Rebuild the hnswlib vector index for every namespace.

        Calls :meth:`rebuild_ns_index` for each namespace found in the DB.
        Returns a list of per-namespace result dicts with keys ``ns``,
        ``old_count``, ``new_count``, and optionally ``error`` when a
        namespace rebuild fails.
        """
        results: list[dict] = []
        for ns in self.list_namespaces():
            try:
                old_n, new_n = self.rebuild_ns_index(ns)
                results.append({"ns": ns, "old_count": old_n, "new_count": new_n})
            except Exception as exc:
                results.append({"ns": ns, "error": str(exc)})
        return results

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

    def rename_ns(self, old_ns: str, new_ns: str) -> int:
        """Rename namespace *old_ns* to *new_ns*.

        Updates all rows in SQL and renames the hnswlib .bin index file.
        Returns the number of rows moved. If *old_ns* does not exist, returns 0.
        Raises ValueError if *new_ns* already exists to prevent silent merges.
        """
        if old_ns == new_ns:
            return 0
        count = self._db.execute(
            "SELECT COUNT(*) FROM memories WHERE ns=?", (old_ns,)
        ).fetchone()[0]
        if count == 0:
            return 0
        existing = self._db.execute(
            "SELECT COUNT(*) FROM memories WHERE ns=?", (new_ns,)
        ).fetchone()[0]
        if existing:
            raise ValueError(
                f"rename_ns: target namespace {new_ns!r} already has {existing} memories; "
                "use forget() first or choose a different name"
            )
        with self._lock:
            self._db.execute(
                "UPDATE memories SET ns=? WHERE ns=?", (new_ns, old_ns)
            )
            self._db.commit()
        old_bin = self.root / f"index_{old_ns}.bin"
        new_bin = self.root / f"index_{new_ns}.bin"
        if old_bin.exists():
            old_bin.rename(new_bin)
        if old_ns in self._index:
            self._index[new_ns] = self._index.pop(old_ns)
        if old_ns in self._index_mtime:
            self._index_mtime[new_ns] = self._index_mtime.pop(old_ns)
        return count

    def copy_ns(self, src_ns: str, dst_ns: str) -> int:
        """Copy all memories from *src_ns* into *dst_ns*, preserving text/meta/tier/summary.

        Unlike rename_ns, the source namespace is left intact.
        access_count and last_accessed are reset to 0/NULL on the copies.
        Returns the number of rows copied (0 if src_ns is empty).
        Raises ValueError if dst_ns already exists.
        """
        count = self._db.execute(
            "SELECT COUNT(*) FROM memories WHERE ns=?", (src_ns,)
        ).fetchone()[0]
        if count == 0:
            return 0
        existing = self._db.execute(
            "SELECT COUNT(*) FROM memories WHERE ns=?", (dst_ns,)
        ).fetchone()[0]
        if existing:
            raise ValueError(
                f"copy_ns: target namespace {dst_ns!r} already has {existing} memories; "
                "use forget() first or choose a different name"
            )
        rows = self._db.execute(
            "SELECT text, summary, meta, created, tier FROM memories WHERE ns=?", (src_ns,)
        ).fetchall()
        with self._lock:
            self._db.executemany(
                "INSERT INTO memories (ns, text, summary, meta, created, tier) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(dst_ns, r[0], r[1], r[2], r[3], r[4]) for r in rows],
            )
            self._db.commit()
        return count

    def merge_ns(self, src_ns: str, dst_ns: str) -> int:
        """Move all memories from *src_ns* into *dst_ns*, then delete *src_ns*.

        Unlike rename_ns, *dst_ns* is allowed to already exist — memories from
        *src_ns* are appended to it. access_count and last_accessed are preserved.
        The source namespace (including its .bin index) is removed after the move.
        Returns the number of rows moved (0 if src_ns is empty).
        """
        count = self._db.execute(
            "SELECT COUNT(*) FROM memories WHERE ns=?", (src_ns,)
        ).fetchone()[0]
        if count == 0:
            return 0
        with self._lock:
            self._db.execute("UPDATE memories SET ns=? WHERE ns=?", (dst_ns, src_ns))
            self._db.commit()
        # Remove source index file (dst index is rebuilt lazily on next search)
        src_bin = self.root / f"index_{src_ns}.bin"
        if src_bin.exists():
            src_bin.unlink()
        # Invalidate cached indexes for both namespaces so they are reloaded
        self._index.pop(src_ns, None)
        self._index_mtime.pop(src_ns, None)
        self._index.pop(dst_ns, None)
        self._index_mtime.pop(dst_ns, None)
        return count

    def stats_by_ns(self) -> list[dict]:
        """Return a lightweight per-namespace stats summary (SQL only, no index I/O).

        Each entry: {ns, total, pinned, default, ambient, oldest, newest}
        where pinned/default/ambient are counts per tier.
        """
        rows = self._db.execute(
            "SELECT ns, tier, COUNT(*), MIN(created), MAX(created) "
            "FROM memories GROUP BY ns, tier ORDER BY ns"
        ).fetchall()
        result: dict[str, dict] = {}
        for ns, tier, cnt, oldest, newest in rows:
            if ns not in result:
                result[ns] = {"ns": ns, "total": 0, "pinned": 0, "default": 0, "ambient": 0,
                               "oldest": oldest, "newest": newest}
            entry = result[ns]
            entry["total"] += cnt
            if tier == 0:
                entry["pinned"] = cnt
            elif tier == 1:
                entry["default"] = cnt
            elif tier == 2:
                entry["ambient"] = cnt
            if oldest and (entry["oldest"] is None or oldest < entry["oldest"]):
                entry["oldest"] = oldest
            if newest and (entry["newest"] is None or newest > entry["newest"]):
                entry["newest"] = newest
        return list(result.values())

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

        # Tier breakdown per namespace
        tier_rows = self._db.execute(
            "SELECT ns, tier, COUNT(*) FROM memories GROUP BY ns, tier"
        ).fetchall()
        tier_map: dict[str, dict[int, int]] = {}
        for ns, tier, cnt in tier_rows:
            tier_map.setdefault(ns, {})[tier] = cnt
        for ns_report in ns_reports:
            ns_report["tier_breakdown"] = tier_map.get(ns_report["ns"], {})

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

    def deduplicate(
        self,
        ns: str = "default",
        threshold: float = 0.98,
        dry_run: bool = True,
        keep: str = "newest",
    ) -> dict:
        """Find and optionally delete near-duplicate memories in a namespace.

        Two memories are considered duplicates when their cosine similarity
        (1 - hnswlib distance) is >= *threshold*. For each duplicate pair,
        the *keep* policy decides which to retain:
        - ``"newest"`` (default): keep the higher ID (inserted later)
        - ``"oldest"``: keep the lower ID (inserted earlier)

        When *dry_run* is True (default) no rows are deleted; the list of
        duplicate pairs is returned for inspection. When *dry_run* is False,
        the designated duplicates are deleted.

        Returns a dict with keys:
        - ``pairs``: list of dicts with ``kept_id``, ``removed_id``, ``similarity``
        - ``removed``: count of rows deleted (0 when dry_run=True)
        """
        import numpy as _np_ded
        try:
            idx = self._index_for(ns)
        except Exception:
            return {"pairs": [], "removed": 0}
        n = idx.get_current_count()
        if n < 2:
            return {"pairs": [], "removed": 0}
        rows = self._db.execute(
            "SELECT id FROM memories WHERE ns=? ORDER BY id", (ns,)
        ).fetchall()
        ids_in_db = [r[0] for r in rows]
        if len(ids_in_db) < 2:
            return {"pairs": [], "removed": 0}
        pairs: list[dict] = []
        to_remove: set[int] = set()
        for mem_id in ids_in_db:
            if mem_id in to_remove:
                continue
            try:
                vec = _np_ded.array(idx.get_items([mem_id])[0], dtype="float32")
            except Exception:
                continue
            fetch_n = min(6, n)
            try:
                labels, distances = idx.knn_query(vec, k=fetch_n)
            except RuntimeError:
                continue
            for label, dist in zip(labels[0], distances[0]):
                label = int(label)
                if label == mem_id or label in to_remove:
                    continue
                sim = float(1.0 - dist)
                if sim >= threshold:
                    if keep == "oldest":
                        kept, removed = min(mem_id, label), max(mem_id, label)
                    else:
                        kept, removed = max(mem_id, label), min(mem_id, label)
                    to_remove.add(removed)
                    pairs.append({"kept_id": kept, "removed_id": removed, "similarity": round(sim, 6)})
        removed_count = 0
        if not dry_run:
            for rid in to_remove:
                if self.delete(rid):
                    removed_count += 1
        return {"pairs": pairs, "removed": removed_count}

    def expire(
        self,
        ns: str | None = None,
        age_days: int = 30,
        min_age_days: int | None = None,
    ) -> int:
        """Downgrade tier-1 memories to tier-2 (ambient) based on staleness.

        A tier-1 memory is demoted to tier-2 when it has NOT been accessed in
        the last *age_days* days (based on last_accessed; falls back to created
        if last_accessed is NULL). Pinned memories (tier=0) are never touched.

        If *min_age_days* is given, only memories older than that many days
        (based on the created timestamp) are eligible — useful to protect
        recently-added memories from premature demotion.

        Returns the count of rows demoted.
        """
        age_clause = (
            "COALESCE(last_accessed, created) <= datetime('now', ? || ' days')"
        )
        params: list = [f"-{age_days}"]
        ns_clause = ""
        if ns is not None:
            ns_clause = " AND ns = ?"
            params.append(ns)
        min_age_clause = ""
        if min_age_days is not None:
            min_age_clause = " AND created <= datetime('now', ? || ' days')"
            params.append(f"-{min_age_days}")
        with self._lock:
            cur = self._db.execute(
                f"UPDATE memories SET tier = 2 "
                f"WHERE tier = 1 AND {age_clause}{ns_clause}{min_age_clause}",
                params,
            )
            self._db.commit()
        return cur.rowcount

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
