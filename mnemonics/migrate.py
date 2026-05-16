"""Plain → encrypted DB migration.

One-shot: open the existing ``memories.db`` with stdlib SQLite, copy every
table (memories, FTS5 mirror, sqlite_sequence) into a freshly-created
SQLCipher DB via ``sqlcipher_export``, atomically swap the file, and store
the key in the OS keyring. The HNSW index files (``index_*.bin``) live
outside SQLite and are not touched.

Failure modes are explicit: an already-encrypted DB is detected up front,
a running MCP server aborts the migration unless ``--force``, and the
original DB is preserved at ``memories.db.preencrypt-<timestamp>`` so the
operation is fully reversible.
"""
from __future__ import annotations

import os
import shutil
import sqlite3 as _stdlib_sqlite
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from mnemonics import crypto


def _running_mcp_pids() -> list[int]:
    """Return PIDs of any ``mnemonics mcp`` processes currently running."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "mnemonics mcp"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return []  # pgrep returns 1 when nothing matches; treat as empty
    except FileNotFoundError:
        return []
    return [int(line) for line in out.splitlines() if line.strip().isdigit()]


def _detect_plaintext(db_path: Path) -> bool:
    """Cheap sniff: a plain SQLite file always begins with ``SQLite format``."""
    with open(db_path, "rb") as f:
        return f.read(16).startswith(b"SQLite format")


def encrypt_db(
    path: str | Path = "~/.mnemonics",
    key_hex: str | None = None,
    store_in_keyring: bool = True,
    force: bool = False,
) -> Path:
    """Encrypt ``<path>/memories.db`` in place. Returns the encrypted file path."""
    try:
        import sqlcipher3
    except ImportError as exc:
        raise RuntimeError(
            "sqlcipher3 is not installed. Run `pip install mnemonics[encrypt]` "
            "(macOS users may need `brew install sqlcipher` first)."
        ) from exc

    root = Path(path).expanduser()
    db = root / "memories.db"
    if not db.is_file():
        raise RuntimeError(f"no memories.db at {root}")

    if not _detect_plaintext(db):
        raise RuntimeError(
            f"{db} is already encrypted (or not a SQLite file). "
            "If this is unexpected, restore from a backup before retrying."
        )

    pids = _running_mcp_pids()
    if pids and not force:
        raise RuntimeError(
            f"mnemonics MCP server(s) running: PIDs {pids}. "
            "Stop them first (each will respawn cleanly) or pass --force."
        )

    if key_hex is None:
        key_hex = crypto.generate_key()
    if len(key_hex) != 64 or any(c not in "0123456789abcdefABCDEF" for c in key_hex):
        raise RuntimeError("--key must be a 64-character hex string (256-bit).")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = db.with_suffix(f".db.preencrypt-{ts}")
    encrypted_tmp = db.with_suffix(f".db.encrypted-{ts}")

    # 1. Snapshot the plain DB so this whole operation is reversible.
    #    Store uses WAL journaling; recent writes may live in the -wal file
    #    until checkpoint. Force a TRUNCATE checkpoint so the main DB file
    #    contains every committed row before we copy it.
    flush = _stdlib_sqlite.connect(str(db))
    flush.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    flush.close()
    shutil.copy2(db, backup)
    print(f"[1/4] backup → {backup}")

    # 2. Build an encrypted copy via sqlcipher_export. SQLCipher's export
    #    function lives in the sqlcipher build of libsqlite, not stdlib, so
    #    we open the plain source through sqlcipher3 with an empty key —
    #    SQLCipher recognises an empty key as "no encryption" and reads
    #    the plain file as-is. The ATTACH supplies the target key, and
    #    sqlcipher_export() copies schema (memories + FTS5 + triggers +
    #    indexes) including row IDs that HNSW labels depend on.
    # Open the plain source through sqlcipher3 without any PRAGMA key —
    # SQLCipher reads unencrypted files transparently when no key is set,
    # while still exposing the sqlcipher_export() builtin we need below.
    plain = sqlcipher3.connect(str(db))
    plain.execute(
        f"ATTACH DATABASE '{encrypted_tmp}' AS enc KEY \"x'{key_hex}'\""
    )
    plain.execute("SELECT sqlcipher_export('enc')")
    plain.execute("DETACH DATABASE enc")
    plain.close()
    print(f"[2/4] encrypted copy → {encrypted_tmp}")

    # 3. Sanity-check: open the encrypted copy with sqlcipher3 and confirm
    #    row counts line up. If anything is off we'll bail before swap.
    plain_count = _stdlib_sqlite.connect(str(backup)).execute(
        "SELECT COUNT(*) FROM memories"
    ).fetchone()[0]
    enc_conn = sqlcipher3.connect(str(encrypted_tmp))
    enc_conn.execute(f"PRAGMA key = \"x'{key_hex}'\"")
    enc_count = enc_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    enc_conn.close()
    if enc_count != plain_count:
        encrypted_tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"row-count mismatch after export: plain={plain_count} enc={enc_count}. "
            f"Aborted; original DB untouched, backup at {backup}."
        )
    print(f"[3/4] row count verified: {enc_count} rows")

    # 4. Atomic swap. WAL companion files of the plain DB are stale now —
    #    they belong to a journal mode the encrypted DB can re-establish on
    #    first open, so we remove them rather than risk a half-mixed state.
    encrypted_tmp.replace(db)
    for suffix in ("-wal", "-shm"):
        leftover = db.with_name(db.name + suffix)
        leftover.unlink(missing_ok=True)
    print(f"[4/4] swapped into place → {db}")

    if store_in_keyring:
        try:
            crypto.store_key(key_hex)
            print(
                "\nKey stored in system keyring under "
                f"service '{crypto._KEYRING_SERVICE}' / user '{crypto._KEYRING_USER}'."
            )
        except Exception as exc:
            print(
                f"\nWARNING: failed to store key in keyring ({exc}). "
                f"Save this key yourself or you will lose access:\n  {key_hex}",
                file=sys.stderr,
            )
    else:
        print(f"\nKey (save this somewhere safe — losing it loses the DB):\n  {key_hex}")

    print(
        "\nNext step: export MNEMONICS_ENCRYPT=1 so the running process "
        "picks up the encrypted DB. Restart any open Claude Code / MCP "
        "sessions so their cached store handles reconnect."
    )
    print(f"Rollback: restore from {backup} and `unset MNEMONICS_ENCRYPT`.")
    return db
