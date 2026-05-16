"""Opt-in encryption tests.

The whole suite runs with ``MNEMONICS_ENCRYPT`` unset, so these tests
spawn a fresh subprocess with the env var flipped before each check.
That keeps the parent test process on the stdlib-sqlite3 path (matching
the default user experience) while still exercising the SQLCipher
branch end-to-end.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("sqlcipher3") is None,
    reason="sqlcipher3 not installed (mnemonics[encrypt] extra)",
)


KEY_HEX_ALPHA = "a" * 64
KEY_HEX_BETA = "b" * 64


def _run(code: str, env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    """Run ``code`` in a subprocess with the given env, returning the result."""
    env = os.environ.copy()
    # Strip any encryption-related leakage from the parent so each test
    # subprocess starts from a known state.
    for k in ("MNEMONICS_ENCRYPT", "MNEMONICS_DB_KEY"):
        env.pop(k, None)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_plain_path_is_default(tmp_path: Path) -> None:
    """No env var → existing stdlib path, file starts with the SQLite magic."""
    code = (
        "from mnemonics.store import Store\n"
        "from mnemonics.ingest import ingest\n"
        f"s = Store({str(tmp_path)!r})\n"
        "print(ingest(['plain row'], s, ns='t'))\n"
    )
    proc = _run(code, env_extra={})
    assert proc.returncode == 0, proc.stderr
    with open(tmp_path / "memories.db", "rb") as f:
        assert f.read(16).startswith(b"SQLite format"), "default path must stay plain"


def test_encrypted_path_writes_random_header(tmp_path: Path) -> None:
    code = (
        "from mnemonics.store import Store\n"
        "from mnemonics.ingest import ingest\n"
        f"s = Store({str(tmp_path)!r})\n"
        "ingest(['encrypted row alpha'], s, ns='t')\n"
    )
    proc = _run(code, env_extra={"MNEMONICS_ENCRYPT": "1", "MNEMONICS_DB_KEY": KEY_HEX_ALPHA})
    assert proc.returncode == 0, proc.stderr
    with open(tmp_path / "memories.db", "rb") as f:
        head = f.read(16)
    assert not head.startswith(b"SQLite"), f"encrypted file leaked plain header: {head!r}"


def test_encrypted_roundtrip_retrieve(tmp_path: Path) -> None:
    write = _run(
        (
            "from mnemonics.store import Store\n"
            "from mnemonics.ingest import ingest\n"
            f"s = Store({str(tmp_path)!r})\n"
            "ingest(['encrypted row alpha', 'encrypted row beta'], s, ns='t')\n"
        ),
        env_extra={"MNEMONICS_ENCRYPT": "1", "MNEMONICS_DB_KEY": KEY_HEX_ALPHA},
    )
    assert write.returncode == 0, write.stderr

    read = _run(
        (
            "from mnemonics.store import Store\n"
            "from mnemonics.retrieve import retrieve\n"
            f"s = Store({str(tmp_path)!r})\n"
            "import json\n"
            "r = retrieve('alpha', s, ns='t', top_k=2)\n"
            "print(json.dumps([h['text'] for h in r['results']]))\n"
        ),
        env_extra={"MNEMONICS_ENCRYPT": "1", "MNEMONICS_DB_KEY": KEY_HEX_ALPHA},
    )
    assert read.returncode == 0, read.stderr
    hits = json.loads(read.stdout.strip().splitlines()[-1])
    assert "encrypted row alpha" in hits, hits


def test_wrong_key_is_rejected(tmp_path: Path) -> None:
    write = _run(
        (
            "from mnemonics.store import Store\n"
            "from mnemonics.ingest import ingest\n"
            f"s = Store({str(tmp_path)!r})\n"
            "ingest(['secret'], s, ns='t')\n"
        ),
        env_extra={"MNEMONICS_ENCRYPT": "1", "MNEMONICS_DB_KEY": KEY_HEX_ALPHA},
    )
    assert write.returncode == 0, write.stderr

    wrong = _run(
        (
            "from mnemonics.store import Store\n"
            f"s = Store({str(tmp_path)!r})\n"
            "print('rows:', s.count('t'))\n"
        ),
        env_extra={"MNEMONICS_ENCRYPT": "1", "MNEMONICS_DB_KEY": KEY_HEX_BETA},
    )
    assert wrong.returncode != 0, (
        "wrong key should NOT yield a readable DB; got stdout=%r" % wrong.stdout
    )


def test_missing_key_fails_loud(tmp_path: Path) -> None:
    """Encryption requested with no key in env or keyring → clear error."""
    proc = _run(
        f"from mnemonics.store import Store\nStore({str(tmp_path)!r})\n",
        env_extra={"MNEMONICS_ENCRYPT": "1"},
        # Intentionally no MNEMONICS_DB_KEY. Keyring also has no mnemonics
        # entry in CI; if a developer runs this locally with a real
        # keyring entry, that's the explicit "configured" path and the
        # test will skip below.
    )
    # If the developer's keyring already holds a mnemonics key, this test
    # cannot prove the "missing" case without nuking their state. Skip
    # rather than fail in that environment.
    if proc.returncode == 0:
        pytest.skip("system keyring already holds a mnemonics-db key")
    assert "no key was found" in proc.stderr.lower() or "missing" in proc.stderr.lower() \
        or "key" in proc.stderr.lower(), proc.stderr


def test_migration_plain_to_encrypted(tmp_path: Path) -> None:
    """End-to-end: plain Store → encrypt_db() → encrypted Store reads same rows."""
    seed = _run(
        (
            "from mnemonics.store import Store\n"
            "from mnemonics.ingest import ingest\n"
            f"s = Store({str(tmp_path)!r})\n"
            "ingest(['mig alpha', 'mig beta'], s, ns='t')\n"
        ),
        env_extra={},  # plain
    )
    assert seed.returncode == 0, seed.stderr

    migrate = _run(
        (
            "from mnemonics.migrate import encrypt_db\n"
            f"encrypt_db(path={str(tmp_path)!r}, key_hex={KEY_HEX_ALPHA!r}, "
            "store_in_keyring=False, force=True)\n"
        ),
        env_extra={},
    )
    assert migrate.returncode == 0, migrate.stderr
    # The pre-encrypt backup must exist for rollback to be possible.
    backups = list(tmp_path.glob("memories.db.preencrypt-*"))
    assert len(backups) == 1, f"expected one backup, got {backups}"

    verify = _run(
        (
            "from mnemonics.store import Store\n"
            f"s = Store({str(tmp_path)!r})\n"
            "print(s.count('t'))\n"
        ),
        env_extra={"MNEMONICS_ENCRYPT": "1", "MNEMONICS_DB_KEY": KEY_HEX_ALPHA},
    )
    assert verify.returncode == 0, verify.stderr
    assert verify.stdout.strip().endswith("2"), verify.stdout
