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

def _v2_result(score=0.85, text="Paris is in France.", tier=1, decay=0.95, boost=1.10, age=5):
    return {
        "results": [{
            "id": 1, "score": score, "raw_score": round(score / (decay * boost), 4),
            "decay_factor": decay, "boost": boost, "age_days": age, "tier": tier,
            "text": text,
        }],
    }


def test_retrieve_prints_v2_breakdown(tmp_path, capsys):
    fake_result = _v2_result(score=0.85, text="Paris is in France.")
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.retrieve.retrieve", return_value=fake_result),
        patch("sys.argv", ["mnemonics", "retrieve", "France", "--path", str(tmp_path)]),
    ):
        main()

    out = capsys.readouterr().out
    assert "0.850" in out
    assert "raw=" in out
    assert "decay=" in out
    assert "boost=" in out
    assert "tier=def" in out
    assert "Paris" in out


def test_retrieve_no_decay_flag(tmp_path):
    fake_result = _v2_result()
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.retrieve.retrieve", return_value=fake_result) as mock_ret,
        patch("sys.argv", ["mnemonics", "retrieve", "q", "--no-decay", "--path", str(tmp_path)]),
    ):
        main()

    call_kwargs = mock_ret.call_args[1]
    assert call_kwargs["decay"] is False


def test_retrieve_top_k_param(tmp_path):
    fake_result = {"results": []}
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.retrieve.retrieve", return_value=fake_result) as mock_ret,
        patch("sys.argv", ["mnemonics", "retrieve", "q", "--top-k", "10", "--path", str(tmp_path)]),
    ):
        main()

    call_kwargs = mock_ret.call_args[1]
    assert call_kwargs["top_k"] == 10


def test_retrieve_pinned_label(tmp_path, capsys):
    fake_result = _v2_result(tier=0, decay=1.0, boost=1.0)
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.retrieve.retrieve", return_value=fake_result),
        patch("sys.argv", ["mnemonics", "retrieve", "q", "--path", str(tmp_path)]),
    ):
        main()

    out = capsys.readouterr().out
    assert "tier=pin" in out


# ── pin / tier / gc ──────────────────────────────────────────────────────────

def test_pin_calls_store_pin(tmp_path):
    mock_store = MagicMock()
    mock_store.pin.return_value = True
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "pin", "42", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    mock_store.pin.assert_called_once_with(42)


def test_tier_calls_store_set_tier(tmp_path):
    mock_store = MagicMock()
    mock_store.set_tier.return_value = True
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "tier", "42", "2", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    mock_store.set_tier.assert_called_once_with(42, 2)


def test_gc_dry_run_lists_only(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.gc_candidates.return_value = [
        {"id": 7, "ns": "ambient", "preview": "old log", "age_days": 45},
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "gc", "--path", str(tmp_path)]),
    ):
        main()

    out = capsys.readouterr().out
    assert "id=    7" in out
    assert "Dry-run" in out
    mock_store.gc.assert_not_called()


def test_gc_apply_deletes(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.gc_candidates.return_value = [
        {"id": 7, "ns": "ambient", "preview": "old log", "age_days": 45},
    ]
    mock_store.gc.return_value = 1
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "gc", "--apply", "--path", str(tmp_path)]),
    ):
        main()

    out = capsys.readouterr().out
    assert "Deleted: 1" in out
    mock_store.gc.assert_called_once()


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
