"""Tests for mnemonics CLI."""
import sys
from unittest.mock import MagicMock, patch
import pytest

from mnemonics.cli import main


def run_main(*args):
    with patch("sys.argv", ["mnemonics", *args]):
        try:
            main()
        except SystemExit:
            pass


# ── no subcommand ─────────────────────────────────────────────────────────────

def test_no_args_exits():
    with patch("sys.argv", ["mnemonics"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code != 0


# ── ingest ────────────────────────────────────────────────────────────────────

def test_ingest_calls_ingest(tmp_path, capsys):
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=1) as mock_ingest,
        patch("sys.argv", ["mnemonics", "ingest", "hello world", "--path", str(tmp_path)]),
    ):
        main()

    mock_ingest.assert_called_once()
    out = capsys.readouterr().out
    assert "1" in out


def test_ingest_namespace(tmp_path):
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=2) as mock_ingest,
        patch("sys.argv", ["mnemonics", "ingest", "text", "--ns", "myns", "--path", str(tmp_path)]),
    ):
        main()

    call_kwargs = mock_ingest.call_args[1]
    assert call_kwargs["ns"] == "myns"


def test_ingest_multiple_words_joined(tmp_path):
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=1) as mock_ingest,
        patch("sys.argv", ["mnemonics", "ingest", "foo", "bar", "baz", "--path", str(tmp_path)]),
    ):
        main()

    call_kwargs = mock_ingest.call_args[1]
    assert call_kwargs["texts"] == ["foo bar baz"]


# ── retrieve ──────────────────────────────────────────────────────────────────

def test_retrieve_prints_trust_score(tmp_path, capsys):
    fake_result = {
        "trust_score": 0.95,
        "flagged_count": 0,
        "results": [{"score": 0.85, "text": "Paris is in France.", "flagged": False}],
    }
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.retrieve.retrieve", return_value=fake_result),
        patch("sys.argv", ["mnemonics", "retrieve", "France", "--path", str(tmp_path)]),
    ):
        main()

    out = capsys.readouterr().out
    assert "0.95" in out
    assert "Paris" in out


def test_retrieve_no_verify_flag(tmp_path):
    fake_result = {"trust_score": 1.0, "flagged_count": 0, "results": []}
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.retrieve.retrieve", return_value=fake_result) as mock_ret,
        patch("sys.argv", ["mnemonics", "retrieve", "q", "--no-verify", "--path", str(tmp_path)]),
    ):
        main()

    call_kwargs = mock_ret.call_args[1]
    assert call_kwargs["verify"] is False


def test_retrieve_top_k_param(tmp_path):
    fake_result = {"trust_score": 1.0, "flagged_count": 0, "results": []}
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.retrieve.retrieve", return_value=fake_result) as mock_ret,
        patch("sys.argv", ["mnemonics", "retrieve", "q", "--top-k", "10", "--path", str(tmp_path)]),
    ):
        main()

    call_kwargs = mock_ret.call_args[1]
    assert call_kwargs["top_k"] == 10


def test_retrieve_flagged_results_show_warning(tmp_path, capsys):
    fake_result = {
        "trust_score": 0.3,
        "flagged_count": 1,
        "results": [{"score": 0.9, "text": "suspicious claim", "flagged": True}],
    }
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.retrieve.retrieve", return_value=fake_result),
        patch("sys.argv", ["mnemonics", "retrieve", "q", "--path", str(tmp_path)]),
    ):
        main()

    out = capsys.readouterr().out
    assert "⚠" in out


# ── stats ─────────────────────────────────────────────────────────────────────

def test_stats_lists_namespaces(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.list_namespaces.return_value = ["default", "work"]
    mock_store.count.side_effect = lambda ns: {"default": 10, "work": 3}[ns]

    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "stats", "--path", str(tmp_path)]),
    ):
        main()

    out = capsys.readouterr().out
    assert "default" in out
    assert "10" in out
    assert "work" in out
    assert "3" in out


def test_stats_empty_store(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.list_namespaces.return_value = []

    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "stats", "--path", str(tmp_path)]),
    ):
        main()

    out = capsys.readouterr().out
    assert out == ""


# ── serve ─────────────────────────────────────────────────────────────────────

def test_serve_calls_serve_with_port(tmp_path):
    with (
        patch("mnemonics.server.serve") as mock_serve,
        patch("sys.argv", ["mnemonics", "serve", "--port", "9999", "--path", str(tmp_path)]),
        patch("os.environ", {}),
    ):
        main()

    mock_serve.assert_called_once_with(port=9999)


# ── mcp ───────────────────────────────────────────────────────────────────────

def test_mcp_calls_serve_mcp():
    with (
        patch("mnemonics.server.serve") as mock_serve,
        patch("sys.argv", ["mnemonics", "mcp"]),
    ):
        main()

    mock_serve.assert_called_once_with(mcp=True)
