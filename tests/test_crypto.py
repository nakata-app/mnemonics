"""Tests for mnemonics.crypto — mocked keyring, no SQLCipher needed."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ── encryption_requested ─────────────────────────────────────────────────────

def test_encryption_requested_true(monkeypatch):
    from mnemonics.crypto import encryption_requested
    monkeypatch.setenv("MNEMONICS_ENCRYPT", "1")
    assert encryption_requested() is True


def test_encryption_requested_false(monkeypatch):
    from mnemonics.crypto import encryption_requested
    monkeypatch.delenv("MNEMONICS_ENCRYPT", raising=False)
    assert encryption_requested() is False


# ── resolve_key ───────────────────────────────────────────────────────────────

def test_resolve_key_from_env(monkeypatch):
    from mnemonics.crypto import resolve_key
    monkeypatch.setenv("MNEMONICS_DB_KEY", "aabbcc")
    assert resolve_key() == "aabbcc"


def test_resolve_key_from_keyring(monkeypatch):
    from mnemonics.crypto import resolve_key
    monkeypatch.delenv("MNEMONICS_DB_KEY", raising=False)
    mock_kr = MagicMock()
    mock_kr.get_password.return_value = "keyring-secret"
    monkeypatch.setitem(sys.modules, "keyring", mock_kr)
    assert resolve_key() == "keyring-secret"


def test_resolve_key_no_keyring_no_env(monkeypatch):
    from mnemonics.crypto import resolve_key
    monkeypatch.delenv("MNEMONICS_DB_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "keyring", None)  # simulate ImportError
    result = resolve_key()
    assert result is None


# ── generate_key ──────────────────────────────────────────────────────────────

def test_generate_key_is_hex_64():
    from mnemonics.crypto import generate_key
    k = generate_key()
    assert len(k) == 64
    int(k, 16)  # valid hex, raises ValueError if not


# ── store_key ─────────────────────────────────────────────────────────────────

def test_store_key_calls_keyring(monkeypatch):
    from mnemonics.crypto import store_key
    mock_kr = MagicMock()
    monkeypatch.setitem(sys.modules, "keyring", mock_kr)
    store_key("myhexkey")
    mock_kr.set_password.assert_called_once_with("mnemonics-db", "default", "myhexkey")


# ── clear_key ─────────────────────────────────────────────────────────────────

def test_clear_key_calls_delete(monkeypatch):
    from mnemonics.crypto import clear_key
    mock_kr = MagicMock()
    monkeypatch.setitem(sys.modules, "keyring", mock_kr)
    clear_key()
    mock_kr.delete_password.assert_called_once()


def test_clear_key_swallows_exception(monkeypatch):
    from mnemonics.crypto import clear_key
    mock_kr = MagicMock()
    mock_kr.delete_password.side_effect = RuntimeError("keyring error")
    monkeypatch.setitem(sys.modules, "keyring", mock_kr)
    clear_key()  # must not raise


# ── require_key ───────────────────────────────────────────────────────────────

def test_require_key_returns_key(monkeypatch):
    from mnemonics.crypto import require_key
    monkeypatch.setenv("MNEMONICS_DB_KEY", "deadbeef")
    assert require_key() == "deadbeef"


def test_require_key_raises_when_missing(monkeypatch):
    from mnemonics.crypto import require_key
    monkeypatch.delenv("MNEMONICS_DB_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "keyring", None)
    with pytest.raises(RuntimeError, match="MNEMONICS_ENCRYPT"):
        require_key()
