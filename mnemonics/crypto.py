"""Optional at-rest encryption support.

Mnemonics' encryption surface is intentionally opt-in. When the environment
variable ``MNEMONICS_ENCRYPT=1`` is set, ``Store`` swaps stdlib ``sqlite3`` for
``sqlcipher3`` and applies ``PRAGMA key`` to every new connection. The actual
key is resolved by ``resolve_key()`` in the order:

    1. ``MNEMONICS_DB_KEY`` env var (preferred for CI / ephemeral setups)
    2. macOS Keychain / Linux secretstorage / Windows DPAPI via ``keyring``
    3. None — caller must decide whether to error or generate a new one

The encryption path is intentionally orthogonal to all retrieval logic:
schema, triggers, FTS5 and HNSW indexes are untouched. Migration from a
plain DB happens via ``mnemonics encrypt-db`` (one-shot, makes a backup).
"""
from __future__ import annotations

import os
import secrets

_KEYRING_SERVICE = "mnemonics-db"
_KEYRING_USER = "default"


def encryption_requested() -> bool:
    """True iff the caller has opted into encrypted storage for this process."""
    return os.environ.get("MNEMONICS_ENCRYPT") == "1"


def resolve_key() -> str | None:
    """Return the active DB key, or ``None`` if no source is configured.

    Env var wins so CI can override without touching the user's keyring.
    """
    env_key = os.environ.get("MNEMONICS_DB_KEY")
    if env_key:
        return env_key
    try:
        import keyring
    except ImportError:
        return None
    return keyring.get_password(_KEYRING_SERVICE, _KEYRING_USER)


def generate_key() -> str:
    """Cryptographically-strong 256-bit key as hex string."""
    return secrets.token_hex(32)


def store_key(key: str) -> None:
    """Persist ``key`` in the system keyring under the mnemonics service."""
    import keyring
    keyring.set_password(_KEYRING_SERVICE, _KEYRING_USER, key)


def clear_key() -> None:
    """Remove the mnemonics key from the system keyring (no-op if absent)."""
    try:
        import keyring
        keyring.delete_password(_KEYRING_SERVICE, _KEYRING_USER)
    except Exception:
        pass


def require_key() -> str:
    """Like ``resolve_key`` but raises if no key is configured.

    Used by ``Store`` when encryption is explicitly requested but no source
    has supplied a key. Failing loud here is better than silently writing
    plaintext to disk.
    """
    key = resolve_key()
    if not key:
        raise RuntimeError(
            "MNEMONICS_ENCRYPT=1 is set but no key was found. "
            "Set MNEMONICS_DB_KEY=<hex> or run `mnemonics encrypt-db` "
            "to provision a key in the system keyring."
        )
    return key
