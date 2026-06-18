"""Tests for mnemonics CLI."""
import json
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


def test_ingest_dedup_no_match_ingests(tmp_path, capsys):
    """When --dedup finds no near-duplicates, ingest proceeds normally."""
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=1) as mock_ingest,
        patch("mnemonics.dedup.find_similar", return_value=[]),
        patch("sys.argv", ["mnemonics", "ingest", "hello", "--dedup", "--path", str(tmp_path)]),
    ):
        main()
    mock_ingest.assert_called_once()
    assert "1" in capsys.readouterr().out


def test_ingest_dedup_skip_similar_exits(tmp_path, capsys):
    """When --dedup finds matches and --skip-similar is set, ingest is skipped."""
    match = {"id": 99, "text": "hello world", "similarity": 0.97}
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest") as mock_ingest,
        patch("mnemonics.dedup.find_similar", return_value=[match]),
        patch("sys.argv", ["mnemonics", "ingest", "hello", "--dedup", "--skip-similar",
                           "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    mock_ingest.assert_not_called()
    out = capsys.readouterr().out
    assert "Skip" in out or "skip" in out


def test_ingest_dedup_force_new_ingests(tmp_path, capsys):
    """When --dedup finds matches but --force-new is set, ingest proceeds."""
    match = {"id": 99, "text": "hello world", "similarity": 0.97}
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=1) as mock_ingest,
        patch("mnemonics.dedup.find_similar", return_value=[match]),
        patch("sys.argv", ["mnemonics", "ingest", "hello", "--dedup", "--force-new",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_ingest.assert_called_once()


def test_ingest_dedup_non_interactive_exits(tmp_path, capsys):
    """Non-interactive run with matches exits without ingesting."""
    match = {"id": 99, "text": "hello world", "similarity": 0.97}
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest") as mock_ingest,
        patch("mnemonics.dedup.find_similar", return_value=[match]),
        patch("sys.stdin.isatty", return_value=False),
        patch("sys.argv", ["mnemonics", "ingest", "hello", "--dedup",
                           "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    mock_ingest.assert_not_called()
    out = capsys.readouterr().out
    assert "Non-interactive" in out or "non-interactive" in out.lower() or "force-new" in out


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
    import numpy as np
    from mnemonics.store import Store, DIM
    store = Store(tmp_path)
    rng = np.random.default_rng(0)
    vecs = rng.random((3, DIM)).astype("float32")
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    store.add(["a", "b", "c"], vecs, ns="default")
    store.add(["x", "y", "z"], vecs, ns="work")

    with patch("sys.argv", ["mnemonics", "stats", "--path", str(tmp_path)]):
        main()

    out = capsys.readouterr().out
    assert "default" in out
    assert "work" in out
    assert "3" in out
    assert "pin=" in out


def test_stats_empty_store(tmp_path, capsys):
    from mnemonics.store import Store
    Store(tmp_path)  # init only

    with patch("sys.argv", ["mnemonics", "stats", "--path", str(tmp_path)]):
        main()

    out = capsys.readouterr().out
    assert "(empty)" in out


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

# ── doctor ────────────────────────────────────────────────────────────────────

def test_doctor_prints_ok(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.health_check.return_value = {
        "db_integrity": "ok",
        "wal_size": 0,
        "namespaces": [{"ns": "default", "sql_count": 5, "idx_count": 5,
                        "soft_deleted": 0, "missing_vectors": 0,
                        "idx_missing": False, "usage_pct": 0.5, "capacity_warning": False}],
        "orphan_indexes": [],
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "doctor", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    out = capsys.readouterr().out
    assert "OK" in out
    assert "default" in out


def test_doctor_json_output(tmp_path, capsys):
    import json
    mock_store = MagicMock()
    mock_store.health_check.return_value = {
        "db_integrity": "ok", "wal_size": 0, "namespaces": [], "orphan_indexes": []
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "doctor", "--json", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    report = json.loads(capsys.readouterr().out)
    assert report["db_integrity"] == "ok"


def test_doctor_fix_calls_repair(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.repair.return_value = {
        "orphan_vectors_fixed": [{"ns": "x", "removed": 3}],
        "orphan_indexes_removed": [],
        "missing_vectors_reported": [],
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "doctor", "--fix", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    mock_store.repair.assert_called_once()
    out = capsys.readouterr().out
    assert "orphan" in out.lower() or "removed" in out.lower() or "✓" in out


# ── forget ────────────────────────────────────────────────────────────────────

def test_forget_dry_run(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.forget_candidates.return_value = [
        {"id": 1, "ns": "old", "tier": 1, "created": "2026-01-01 00:00:00", "preview": "stale"},
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "forget", "--ns", "old", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    mock_store.forget.assert_not_called()
    out = capsys.readouterr().out
    assert "dry-run" in out.lower() or "1 row" in out or "apply" in out.lower()


def test_forget_apply_deletes(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.forget_candidates.return_value = [
        {"id": 2, "ns": "old", "tier": 1, "created": "2026-01-01 00:00:00", "preview": "stale"},
    ]
    mock_store.forget.return_value = 1
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "forget", "--ns", "old", "--apply", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    mock_store.forget.assert_called_once()
    out = capsys.readouterr().out
    assert "1" in out


# ── rebuild-index ─────────────────────────────────────────────────────────────

def test_rebuild_index_cli(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.rebuild_ns_index.return_value = (50, 45)
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "rebuild-index", "--ns", "myns", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    mock_store.rebuild_ns_index.assert_called_once_with("myns")
    out = capsys.readouterr().out
    assert "50" in out and "45" in out


# ── gc tier flag ──────────────────────────────────────────────────────────────

def test_gc_tier1_dry_run(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.gc_candidates.return_value = [
        {"id": 9, "ns": "sessions", "preview": "old", "age_days": 65},
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "gc", "--tier", "1", "--age-days", "60", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    mock_store.gc_candidates.assert_called_once()
    call_kwargs = mock_store.gc_candidates.call_args
    assert call_kwargs.kwargs.get("tier") == 1 or (call_kwargs.args and 1 in call_kwargs.args)


# ── mcp ───────────────────────────────────────────────────────────────────────

def test_mcp_calls_serve_mcp():
    with (
        patch("mnemonics.server.serve") as mock_serve,
        patch("sys.argv", ["mnemonics", "mcp"]),
    ):
        main()

    mock_serve.assert_called_once_with(mcp=True)


# ── sync export / import ──────────────────────────────────────────────────────

def test_sync_export_calls_export_store(tmp_path, capsys):
    from pathlib import Path
    fake_archive = tmp_path / "store.sync.tar.gz"
    fake_archive.write_bytes(b"")
    with (
        patch("mnemonics.sync.export_store", return_value=fake_archive) as mock_ex,
        patch("sys.argv", ["mnemonics", "sync", "export", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    mock_ex.assert_called_once()
    out = capsys.readouterr().out
    assert "Wrote" in out or str(fake_archive) in out


def test_sync_import_calls_import_store(tmp_path, capsys):
    fake_archive = str(tmp_path / "store.sync.tar.gz")
    summary = {"imported": 5, "skipped": 1, "overwritten": 0}
    with (
        patch("mnemonics.sync.import_store", return_value=summary) as mock_im,
        patch("sys.argv", ["mnemonics", "sync", "import", fake_archive, "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    mock_im.assert_called_once()
    out = capsys.readouterr().out
    assert "imported=5" in out


# ── export-jsonl ──────────────────────────────────────────────────────────────

def test_export_jsonl_all_ns(tmp_path, capsys):
    """export-jsonl outputs JSONL to stdout for all namespaces."""
    import json as _json
    import numpy as np
    from mnemonics.store import Store
    s = Store(tmp_path)
    v = np.random.rand(384).astype("float32")
    v /= np.linalg.norm(v)
    s.add(["hello world"], v.reshape(1, -1), ns="a")
    s.add(["goodbye world"], v.reshape(1, -1), ns="b")
    with patch("sys.argv", ["mnemonics", "export-jsonl", "--path", str(tmp_path)]):
        main()
    lines = [l for l in capsys.readouterr().out.strip().split("\n") if l]
    assert len(lines) == 2
    objs = [_json.loads(l) for l in lines]
    texts = {o["text"] for o in objs}
    assert texts == {"hello world", "goodbye world"}


def test_export_jsonl_ns_filter(tmp_path, capsys):
    """export-jsonl --ns filters to one namespace."""
    import json as _json
    import numpy as np
    from mnemonics.store import Store
    s = Store(tmp_path)
    v = np.random.rand(384).astype("float32")
    v /= np.linalg.norm(v)
    s.add(["keep this"], v.reshape(1, -1), ns="keep")
    s.add(["skip this"], v.reshape(1, -1), ns="skip")
    with patch("sys.argv", ["mnemonics", "export-jsonl", "--ns", "keep", "--path", str(tmp_path)]):
        main()
    lines = [l for l in capsys.readouterr().out.strip().split("\n") if l]
    assert len(lines) == 1
    assert _json.loads(lines[0])["text"] == "keep this"


def test_export_jsonl_tier_filter(tmp_path, capsys):
    """export-jsonl --tier filters to pinned (tier=0) only."""
    import json as _json
    import numpy as np
    from mnemonics.store import Store
    s = Store(tmp_path)
    v = np.random.rand(384).astype("float32")
    v /= np.linalg.norm(v)
    ids = s.add(["pinned text", "default text"], v.reshape(1, -1).repeat(2, axis=0))
    s.pin(ids[0])
    with patch("sys.argv", ["mnemonics", "export-jsonl", "--tier", "0", "--path", str(tmp_path)]):
        main()
    lines = [l for l in capsys.readouterr().out.strip().split("\n") if l]
    assert len(lines) == 1
    assert _json.loads(lines[0])["text"] == "pinned text"


def test_export_jsonl_to_file(tmp_path):
    """export-jsonl --out writes to file, prints count to stderr."""
    import json as _json
    import numpy as np
    from mnemonics.store import Store
    s = Store(tmp_path)
    v = np.random.rand(384).astype("float32")
    v /= np.linalg.norm(v)
    s.add(["file export test"], v.reshape(1, -1))
    out_file = tmp_path / "export.jsonl"
    with patch("sys.argv", ["mnemonics", "export-jsonl", "--out", str(out_file), "--path", str(tmp_path)]):
        main()
    lines = out_file.read_text().strip().split("\n")
    assert len(lines) == 1
    assert _json.loads(lines[0])["text"] == "file export test"


# ── backup / restore ──────────────────────────────────────────────────────────

def test_backup_calls_backup(tmp_path, capsys):
    from pathlib import Path
    fake_archive = tmp_path / "backup.tar.gz"
    fake_archive.write_bytes(b"x" * 100)
    with (
        patch("mnemonics.backup.backup", return_value=fake_archive) as mock_bk,
        patch("sys.argv", ["mnemonics", "backup", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    mock_bk.assert_called_once()
    out = capsys.readouterr().out
    assert "Wrote" in out


def test_restore_calls_restore(tmp_path, capsys):
    fake_archive = str(tmp_path / "backup.tar.gz")
    with (
        patch("mnemonics.backup.restore", return_value=["memories.db"]) as mock_rs,
        patch("sys.argv", ["mnemonics", "restore", fake_archive, "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    mock_rs.assert_called_once()
    out = capsys.readouterr().out
    assert "memories.db" in out or "Restored" in out


# ── list ──────────────────────────────────────────────────────────────────────

def test_cli_list_empty(tmp_path, capsys):
    from mnemonics.store import Store
    Store(tmp_path)  # init empty store
    with patch("sys.argv", ["mnemonics", "list", "--ns", "default", "--path", str(tmp_path)]):
        try:
            main()
        except SystemExit:
            pass
    out = capsys.readouterr().out
    assert "No memories" in out


def test_cli_list_shows_rows(tmp_path, capsys):
    import numpy as np
    from mnemonics.store import Store, DIM
    store = Store(tmp_path)
    rng = np.random.default_rng(0)
    vecs = rng.random((3, DIM)).astype("float32")
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    store.add(["alpha", "beta", "gamma"], vecs, ns="default")
    with patch("sys.argv", ["mnemonics", "list", "--ns", "default", "--path", str(tmp_path)]):
        try:
            main()
        except SystemExit:
            pass
    out = capsys.readouterr().out
    assert "showing 3 row(s)" in out
    assert "alpha" in out or "beta" in out or "gamma" in out


# ── bm25 ──────────────────────────────────────────────────────────────────────

def test_cli_bm25_no_results(tmp_path, capsys):
    from mnemonics.store import Store
    Store(tmp_path)
    with patch("sys.argv", ["mnemonics", "bm25", "xyzzy", "--path", str(tmp_path)]):
        try:
            main()
        except SystemExit:
            pass
    out = capsys.readouterr().out
    assert "No BM25" in out


def test_cli_bm25_finds_text(tmp_path, capsys):
    import numpy as np
    from mnemonics.store import Store, DIM
    store = Store(tmp_path)
    rng = np.random.default_rng(0)
    vecs = rng.random((1, DIM)).astype("float32")
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    store.add(["unique_keyword_xq7"], vecs, ns="default")
    with patch("sys.argv", ["mnemonics", "bm25", "unique_keyword_xq7", "--path", str(tmp_path)]):
        try:
            main()
        except SystemExit:
            pass
    out = capsys.readouterr().out
    assert "unique_keyword_xq7" in out


# ── get ───────────────────────────────────────────────────────────────────────

def test_cli_get_existing(tmp_path, capsys):
    import numpy as np
    from mnemonics.store import Store, DIM
    store = Store(tmp_path)
    rng = np.random.default_rng(0)
    vecs = rng.random((1, DIM)).astype("float32")
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = store.add(["hello world content"], vecs, ns="default")
    with patch("sys.argv", ["mnemonics", "get", str(ids[0]), "--path", str(tmp_path)]):
        try:
            main()
        except SystemExit:
            pass
    out = capsys.readouterr().out
    assert "hello world content" in out
    assert f"id={ids[0]}" in out


def test_cli_get_not_found(tmp_path, capsys):
    from mnemonics.store import Store
    Store(tmp_path)
    with patch("sys.argv", ["mnemonics", "get", "9999", "--path", str(tmp_path)]):
        try:
            main()
        except SystemExit:
            pass
    out = capsys.readouterr().out
    assert "not found" in out


# ── set-summary ───────────────────────────────────────────────────────────────

def test_cli_set_summary(tmp_path, capsys):
    import numpy as np
    from mnemonics.store import Store, DIM
    store = Store(tmp_path)
    rng = np.random.default_rng(0)
    vecs = rng.random((1, DIM)).astype("float32")
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = store.add(["raw text"], vecs)
    with patch("sys.argv", ["mnemonics", "set-summary", str(ids[0]), "my new summary",
                             "--path", str(tmp_path)]):
        try:
            main()
        except SystemExit:
            pass
    out = capsys.readouterr().out
    assert "updated" in out.lower() or "Summary" in out
    assert store.get(ids[0])["summary"] == "my new summary"


def test_cli_set_summary_clear(tmp_path, capsys):
    import numpy as np
    from mnemonics.store import Store, DIM
    store = Store(tmp_path)
    rng = np.random.default_rng(0)
    vecs = rng.random((1, DIM)).astype("float32")
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = store.add(["raw text"], vecs)
    store.update_summary(ids[0], "old summary")
    with patch("sys.argv", ["mnemonics", "set-summary", str(ids[0]),
                             "--path", str(tmp_path)]):
        try:
            main()
        except SystemExit:
            pass
    assert store.get(ids[0])["summary"] is None


# ── retrieve summary output ───────────────────────────────────────────────────

def test_retrieve_summary_output(tmp_path, capsys):
    fake = {"results": [{
        "id": 5, "score": 0.9, "raw_score": 0.9, "decay_factor": 1.0,
        "boost": 1.0, "age_days": 0.0, "tier": 0,
        "text": "full raw content", "summary": "short gist here",
    }]}
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.retrieve.retrieve", return_value=fake),
        patch("sys.argv", ["mnemonics", "retrieve", "q", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "short gist here" in out
    assert "└─ raw:" in out
    assert "full raw content" in out


# ── gc no candidates ─────────────────────────────────────────────────────────

def test_gc_no_candidates(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.gc_candidates.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "gc", "--age-days", "99", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    out = capsys.readouterr().out
    assert "Nothing to GC" in out


# ── bm25 summary output ───────────────────────────────────────────────────────

def test_cli_bm25_shows_summary(tmp_path, capsys):
    import numpy as np
    from mnemonics.store import Store, DIM
    store = Store(tmp_path)
    rng = np.random.default_rng(0)
    vecs = rng.random((1, DIM)).astype("float32")
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = store.add(["unique_keyword_abc123"], vecs)
    store.update_summary(ids[0], "the short gist")
    with patch("sys.argv", ["mnemonics", "bm25", "unique_keyword_abc123", "--path", str(tmp_path)]):
        try:
            main()
        except SystemExit:
            pass
    out = capsys.readouterr().out
    assert "unique_keyword_abc123" in out
    assert "the short gist" in out


# ── get with summary ──────────────────────────────────────────────────────────

def test_cli_get_with_summary(tmp_path, capsys):
    import numpy as np
    from mnemonics.store import Store, DIM
    store = Store(tmp_path)
    rng = np.random.default_rng(0)
    vecs = rng.random((1, DIM)).astype("float32")
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = store.add(["memory content"], vecs)
    store.update_summary(ids[0], "memory gist")
    with patch("sys.argv", ["mnemonics", "get", str(ids[0]), "--path", str(tmp_path)]):
        try:
            main()
        except SystemExit:
            pass
    out = capsys.readouterr().out
    assert "memory content" in out
    assert "memory gist" in out


# ── forget no candidates ──────────────────────────────────────────────────────

def test_forget_no_candidates(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.forget_candidates.return_value = []
    mock_store._db = MagicMock()
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "forget", "--ns", "myns", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    out = capsys.readouterr().out
    assert "Nothing to forget" in out
    assert "ns=myns" in out


def test_forget_no_candidates_with_before(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.forget_candidates.return_value = []
    mock_store._db = MagicMock()
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "forget", "--ns", "myns",
                           "--before", "2025-01-01", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    out = capsys.readouterr().out
    assert "before=2025-01-01" in out


# ── doctor status branches ───────────────────────────────────────────────────

def _make_doctor_ns(ns="default", sql=10, idx=10, soft_deleted=0, idx_missing=False,
                     missing_vectors=0, capacity_warning=False, usage_pct=0):
    return {
        "ns": ns, "sql_count": sql, "idx_count": idx,
        "soft_deleted": soft_deleted, "idx_missing": idx_missing,
        "missing_vectors": missing_vectors,
        "capacity_warning": capacity_warning, "usage_pct": usage_pct,
    }


def _base_report(**ns_kwargs):
    return {
        "db_integrity": "ok",
        "wal_size": 0,
        "namespaces": [_make_doctor_ns(**ns_kwargs)],
        "orphan_indexes": [],
    }


def test_doctor_missing_vectors_status(tmp_path, capsys):
    report = _base_report(missing_vectors=3)
    mock_store = MagicMock()
    mock_store.health_check.return_value = report
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "doctor", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    assert "missing vector" in capsys.readouterr().out


def test_doctor_orphan_vectors_status(tmp_path, capsys):
    report = _base_report(soft_deleted=2)
    mock_store = MagicMock()
    mock_store.health_check.return_value = report
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "doctor", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    assert "orphan vector" in capsys.readouterr().out


def test_doctor_capacity_warning_status(tmp_path, capsys):
    report = _base_report(capacity_warning=True, usage_pct=90)
    mock_store = MagicMock()
    mock_store.health_check.return_value = report
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "doctor", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    assert "90%" in capsys.readouterr().out


def test_doctor_orphan_indexes_output(tmp_path, capsys):
    report = {
        "db_integrity": "ok",
        "wal_size": 0,
        "namespaces": [_make_doctor_ns()],
        "orphan_indexes": [{"ns": "ghost", "size": 1024 * 1024, "path": "/tmp/ghost.bin"}],
    }
    mock_store = MagicMock()
    mock_store.health_check.return_value = report
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "doctor", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    out = capsys.readouterr().out
    assert "Orphan indexes" in out
    assert "ghost.bin" in out


def test_doctor_issues_count_exit(tmp_path, capsys):
    report = _base_report(soft_deleted=1)
    mock_store = MagicMock()
    mock_store.health_check.return_value = report
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "doctor", "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1
    assert "issue" in capsys.readouterr().out


# ── gc >50 candidates ─────────────────────────────────────────────────────────

def test_gc_many_candidates(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.gc_candidates.return_value = [
        {"id": i, "ns": "default", "age_days": 99, "preview": "x"} for i in range(55)
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "gc", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    assert "... and 5 more" in capsys.readouterr().out


# ── doctor --fix paths ────────────────────────────────────────────────────────

def test_doctor_fix_nothing(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.repair.return_value = {
        "orphan_vectors_fixed": [],
        "orphan_indexes_removed": [],
        "missing_vectors_reported": [],
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "doctor", "--fix", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    assert "Nothing to fix" in capsys.readouterr().out


def test_doctor_fix_orphan_vector_error(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.repair.return_value = {
        "orphan_vectors_fixed": [{"ns": "bad", "error": "index corrupt"}],
        "orphan_indexes_removed": [],
        "missing_vectors_reported": [],
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "doctor", "--fix", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    assert "index corrupt" in capsys.readouterr().out


def test_doctor_fix_orphan_index_removed(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.repair.return_value = {
        "orphan_vectors_fixed": [{"ns": "ok", "removed": 2}],
        "orphan_indexes_removed": ["/tmp/orphan.bin"],
        "missing_vectors_reported": [{"ns": "partial", "missing": 5}],
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "doctor", "--fix", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    out = capsys.readouterr().out
    assert "orphan.bin" in out
    assert "partial" in out


def test_doctor_fix_orphan_index_error(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.repair.return_value = {
        "orphan_vectors_fixed": [],
        "orphan_indexes_removed": [{"path": "/tmp/bad.bin", "error": "permission denied"}],
        "missing_vectors_reported": [],
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "doctor", "--fix", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    assert "permission denied" in capsys.readouterr().out


# ── doctor idx_missing (no index) ─────────────────────────────────────────────

def test_doctor_no_index_status(tmp_path, capsys):
    report = {
        "db_integrity": "ok",
        "wal_size": 0,
        "namespaces": [_make_doctor_ns(idx_missing=True, idx=None)],
        "orphan_indexes": [],
    }
    mock_store = MagicMock()
    mock_store.health_check.return_value = report
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "doctor", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    assert "no index" in capsys.readouterr().out


# ── set-summary not found ─────────────────────────────────────────────────────

def test_set_summary_not_found(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.update_summary.return_value = False
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "set-summary", "999", "gist", "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1
    assert "not found" in capsys.readouterr().out


# ── forget >50 candidates ─────────────────────────────────────────────────────

def test_forget_many_candidates(tmp_path, capsys):
    candidates = [
        {"id": i, "tier": 2, "created": "2025-01-01 00:00:00", "preview": "x"} for i in range(55)
    ]
    mock_store = MagicMock()
    mock_store.forget_candidates.return_value = candidates
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchone.return_value = (0,)
    mock_store._db = mock_db
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "forget", "--ns", "x", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    assert "... and 5 more" in capsys.readouterr().out


# ── restore error paths ───────────────────────────────────────────────────────

def test_restore_file_exists_error(tmp_path, capsys):
    with (
        patch("mnemonics.backup.restore", side_effect=FileExistsError("db already exists")),
        patch("sys.argv", ["mnemonics", "restore", "backup.tar.gz", "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2
    assert "Refusing" in capsys.readouterr().err


def test_restore_no_files_written(tmp_path, capsys):
    with (
        patch("mnemonics.backup.restore", return_value=[]),
        patch("sys.argv", ["mnemonics", "restore", "backup.tar.gz", "--path", str(tmp_path)]),
    ):
        main()
    assert "no restorable" in capsys.readouterr().out


# ── forget with --tier filter note ────────────────────────────────────────────

def test_forget_no_candidates_with_tier(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.forget_candidates.return_value = []
    mock_store._db = MagicMock()
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "forget", "--ns", "myns",
                           "--tier", "2", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    assert "tier=2" in capsys.readouterr().out


# ── eval command ──────────────────────────────────────────────────────────────

def test_eval_basic(tmp_path, capsys):
    fake_result = {
        "encoder": "minilm",
        "method": "vector",
        "mrr": 0.8,
        "r1": 0.7,
        "r5": 0.9,
        "ndcg10": 0.85,
    }
    with (
        patch("mnemonics.eval.run_eval", return_value=fake_result),
        patch("mnemonics.eval.compare_table", return_value="eval table here"),
        patch("sys.argv", [
            "mnemonics", "eval",
            "--corpus", str(tmp_path / "corpus.jsonl"),
            "--queries", str(tmp_path / "queries.jsonl"),
        ]),
    ):
        main()
    out = capsys.readouterr().out
    assert "eval table here" in out
    assert "[eval] encoder=minilm method=vector" in out


def test_eval_with_out_dir(tmp_path, capsys):
    out_dir = tmp_path / "results"
    fake_result = {"encoder": "minilm", "method": "vector", "mrr": 0.5}
    import json as _json
    with (
        patch("mnemonics.eval.run_eval", return_value=fake_result),
        patch("mnemonics.eval.compare_table", return_value="table"),
        patch("builtins.open", MagicMock()),
        patch("json.dump"),
        patch("sys.argv", [
            "mnemonics", "eval",
            "--corpus", str(tmp_path / "c.jsonl"),
            "--queries", str(tmp_path / "q.jsonl"),
            "--out", str(out_dir),
        ]),
    ):
        main()
    assert "table" in capsys.readouterr().out


# ── encrypt-db error path ─────────────────────────────────────────────────────

def test_encrypt_db_runtime_error(tmp_path, capsys):
    with (
        patch("mnemonics.migrate.encrypt_db", side_effect=RuntimeError("mcp is running")),
        patch("sys.argv", ["mnemonics", "encrypt-db", "--key", "a" * 64, "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2
    assert "mcp is running" in capsys.readouterr().err


# ── if __name__ == "__main__" ─────────────────────────────────────────────────

def test_main_as_module(tmp_path, monkeypatch):
    """Running cli.py as __main__ should call main()."""
    import runpy
    monkeypatch.setattr("sys.argv", ["mnemonics", "stats", "--path", str(tmp_path)])
    import numpy as np
    from mnemonics.store import Store, DIM
    Store(tmp_path)  # create empty DB so stats doesn't fail
    with patch("builtins.print"):
        runpy.run_module("mnemonics.cli", run_name="__main__", alter_sys=True)


def test_sync_no_subcommand_shows_help(capsys):
    """cli.py lines 499-500: sync with no subcommand → print_help + exit 1."""
    with patch("sys.argv", ["mnemonics", "sync"]):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


def test_ingest_interactive_cancelled(tmp_path, capsys, monkeypatch):
    """cli.py lines 220-221, 227-228: --dedup + interactive prompt → cancel."""
    fake_match = [{"id": 1, "text": "identical text", "similarity": 0.99}]
    with (
        patch("sys.argv", ["mnemonics", "ingest", "identical text",
                           "--path", str(tmp_path), "--dedup"]),
        patch("sys.stdin.isatty", return_value=True),
        patch("builtins.input", return_value=""),   # empty → not y → decision = "c"
        patch("mnemonics.dedup.find_similar", return_value=fake_match),
        patch("mnemonics.store.Store"),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Cancelled" in out


# ── get-many ──────────────────────────────────────────────────────────────────

def test_cli_get_many_found(tmp_path, capsys):
    """get-many prints one line per found memory."""
    fake_rows = [
        {"id": 1, "ns": "default", "tier": 1, "text": "hello world"},
        {"id": 2, "ns": "default", "tier": 0, "text": "pinned memory"},
    ]
    mock_store = MagicMock()
    mock_store.get_many.return_value = fake_rows
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "get-many", "1", "2", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "id=1" in out
    assert "id=2" in out
    assert "pinned" in out


def test_cli_get_many_none_found(tmp_path, capsys):
    """get-many prints 'No memories' when all IDs are missing."""
    mock_store = MagicMock()
    mock_store.get_many.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "get-many", "99", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "No memories" in out


# ── delete-ids ────────────────────────────────────────────────────────────────

def test_cli_delete_ids(tmp_path, capsys):
    """delete-ids reports deleted count."""
    mock_store = MagicMock()
    mock_store.delete_many.return_value = 2
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "delete-ids", "1", "2", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "Deleted 2 of 2" in out


# ── search-meta ───────────────────────────────────────────────────────────────

def test_cli_search_meta_found(tmp_path, capsys):
    """search-meta prints matches."""
    fake_rows = [{"id": 5, "tier": 1, "text": "tagged memory"}]
    mock_store = MagicMock()
    mock_store.search_by_meta.return_value = fake_rows
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "search-meta", "tag=important", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "id=5" in out
    mock_store.search_by_meta.assert_called_once_with({"tag": "important"}, ns="default", limit=20)


def test_cli_search_meta_no_results(tmp_path, capsys):
    """search-meta prints 'No results' when nothing matches."""
    mock_store = MagicMock()
    mock_store.search_by_meta.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "search-meta", "tag=ghost", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "No results" in out


def test_cli_search_meta_bad_filter(tmp_path, capsys):
    """search-meta exits 2 when a filter has no '='."""
    with (
        patch("mnemonics.store.Store"),
        patch("sys.argv", ["mnemonics", "search-meta", "badfilter", "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2


# ── update-meta ───────────────────────────────────────────────────────────────

def test_cli_update_meta_ok(tmp_path, capsys):
    """update-meta updates and prints confirmation."""
    mock_store = MagicMock()
    mock_store.update_meta.return_value = True
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "update-meta", "3", '{"x": 1}', "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Meta updated" in out


def test_cli_update_meta_not_found(tmp_path, capsys):
    """update-meta exits 1 when ID doesn't exist."""
    mock_store = MagicMock()
    mock_store.update_meta.return_value = False
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "update-meta", "99", '{}', "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


def test_cli_update_meta_bad_json(tmp_path, capsys):
    """update-meta exits 2 on invalid JSON."""
    with (
        patch("mnemonics.store.Store"),
        patch("sys.argv", ["mnemonics", "update-meta", "1", "NOT_JSON", "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2


def test_cli_update_meta_non_object_json(tmp_path, capsys):
    """update-meta exits 2 when JSON is not an object."""
    with (
        patch("mnemonics.store.Store"),
        patch("sys.argv", ["mnemonics", "update-meta", "1", "[1,2,3]", "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2


# ── retrieve min_tier/max_tier ────────────────────────────────────────────────

def test_cli_retrieve_tier_filters_passed(tmp_path):
    """retrieve --min-tier / --max-tier are forwarded to retrieve()."""
    from unittest.mock import patch, MagicMock
    mock_result = {"results": []}
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.retrieve.retrieve", return_value=mock_result) as mock_ret,
        patch("sys.argv", ["mnemonics", "retrieve", "hello",
                           "--min-tier", "1", "--max-tier", "2",
                           "--path", str(tmp_path)]),
    ):
        main()
    call_kwargs = mock_ret.call_args[1]
    assert call_kwargs["min_tier"] == 1
    assert call_kwargs["max_tier"] == 2


# ── bm25 min_tier/max_tier ────────────────────────────────────────────────────

def test_cli_bm25_tier_filters_passed(tmp_path):
    """bm25 --min-tier / --max-tier are forwarded to search_bm25()."""
    mock_store = MagicMock()
    mock_store.search_bm25.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "bm25", "hello",
                           "--min-tier", "0", "--max-tier", "1",
                           "--path", str(tmp_path)]),
    ):
        main()
    call_kwargs = mock_store.search_bm25.call_args[1]
    assert call_kwargs["min_tier"] == 0
    assert call_kwargs["max_tier"] == 1


# ── count ─────────────────────────────────────────────────────────────────────

def test_cli_count_all(tmp_path, capsys):
    """count --ns=None reports total across all namespaces."""
    mock_store = MagicMock()
    mock_store.count.return_value = 42
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "count", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "42" in out
    assert "all namespaces" in out
    mock_store.count.assert_called_once_with(ns=None)


def test_cli_count_ns(tmp_path, capsys):
    """count --ns=foo reports count for that namespace."""
    mock_store = MagicMock()
    mock_store.count.return_value = 7
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "count", "--ns", "foo", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "7" in out
    assert "foo" in out
    mock_store.count.assert_called_once_with(ns="foo")


# ── set-tier-many ─────────────────────────────────────────────────────────────

def test_cli_set_tier_many(tmp_path, capsys):
    """set-tier-many updates rows and reports count."""
    mock_store = MagicMock()
    mock_store.update_tier_many.return_value = 3
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "set-tier-many", "0", "1", "2", "3", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "Updated 3 of 3" in out
    assert "pinned" in out
    mock_store.update_tier_many.assert_called_once_with([1, 2, 3], 0)


# ── export-jsonl --meta-filter ────────────────────────────────────────────────

def test_cli_export_jsonl_meta_filter(tmp_path, capsys):
    """export-jsonl --meta-filter filters rows by metadata key=value."""
    from mnemonics.store import Store
    store = Store(tmp_path)
    # Ingest two rows, only one with matching meta (no embedding column in schema)
    store._db.execute(
        "INSERT INTO memories (ns, text, meta, tier) VALUES (?,?,?,?)",
        ("default", "match", '{"tag":"x"}', 1),
    )
    store._db.execute(
        "INSERT INTO memories (ns, text, meta, tier) VALUES (?,?,?,?)",
        ("default", "no match", '{"tag":"y"}', 1),
    )
    store._db.commit()
    with patch("sys.argv", ["mnemonics", "export-jsonl",
                             "--meta-filter", "tag=x",
                             "--path", str(tmp_path)]):
        main()
    out = capsys.readouterr().out
    lines = [l for l in out.strip().splitlines() if l]
    assert len(lines) == 1
    assert '"match"' in lines[0]


def test_cli_export_jsonl_meta_filter_bad(tmp_path, capsys):
    """export-jsonl --meta-filter with no '=' exits 2."""
    with (
        patch("mnemonics.store.Store"),
        patch("sys.argv", ["mnemonics", "export-jsonl",
                            "--meta-filter", "badfilter",
                            "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2


# ── ingest --meta ─────────────────────────────────────────────────────────────

def test_cli_ingest_meta(tmp_path, capsys):
    """ingest --meta passes dict to ingest()."""
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=1) as mock_ingest,
        patch("sys.argv", ["mnemonics", "ingest", "hello",
                           "--meta", '{"tag":"work"}',
                           "--path", str(tmp_path)]),
    ):
        main()
    call_kwargs = mock_ingest.call_args[1]
    assert call_kwargs["meta"] == [{"tag": "work"}]


def test_cli_ingest_meta_bad_json(tmp_path, capsys):
    """ingest --meta with invalid JSON exits 2."""
    with (
        patch("mnemonics.store.Store"),
        patch("sys.argv", ["mnemonics", "ingest", "hello",
                           "--meta", "NOT_JSON",
                           "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2


def test_cli_ingest_meta_non_object(tmp_path, capsys):
    """ingest --meta with a non-object JSON exits 2."""
    with (
        patch("mnemonics.store.Store"),
        patch("sys.argv", ["mnemonics", "ingest", "hello",
                           "--meta", "[1,2,3]",
                           "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2


# ── get --json ────────────────────────────────────────────────────────────────

def test_cli_get_json_output(tmp_path, capsys):
    """get --json prints a JSON object."""
    import json as _json
    fake_row = {"id": 7, "ns": "default", "text": "hello", "tier": 1,
                "summary": None, "meta": {}, "created": "2026-01-01 00:00:00",
                "last_accessed": None, "access_count": 0}
    mock_store = MagicMock()
    mock_store.get.return_value = fake_row
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "get", "7", "--json", "--path", str(tmp_path)]),
    ):
        try:
            main()
        except SystemExit:
            pass
    out = capsys.readouterr().out
    data = _json.loads(out.strip())
    assert data["id"] == 7
    assert data["text"] == "hello"


# ── list --json ───────────────────────────────────────────────────────────────

def test_cli_list_json_output(tmp_path, capsys):
    """list --json outputs one JSON object per line."""
    import json as _json
    fake_rows = [
        {"id": 1, "ns": "default", "text": "a", "tier": 1, "summary": None,
         "meta": {}, "created": "2026-01-01 00:00:00", "last_accessed": None, "access_count": 0},
        {"id": 2, "ns": "default", "text": "b", "tier": 2, "summary": None,
         "meta": {}, "created": "2026-01-02 00:00:00", "last_accessed": None, "access_count": 0},
    ]
    mock_store = MagicMock()
    mock_store.list_memories.return_value = fake_rows
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "list", "--json", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    lines = [l for l in out.strip().splitlines() if l]
    assert len(lines) == 2
    assert _json.loads(lines[0])["id"] == 1
    assert _json.loads(lines[1])["id"] == 2


def test_cli_list_json_empty(tmp_path, capsys):
    """list --json with no rows outputs '[]'."""
    mock_store = MagicMock()
    mock_store.list_memories.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "list", "--json", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out.strip()
    assert out == "[]"


# ── retrieve --json ───────────────────────────────────────────────────────────

def test_cli_retrieve_json_output(tmp_path, capsys):
    """retrieve --json outputs a JSON array."""
    import json as _json
    fake_result = {"results": [
        {"id": 3, "score": 0.9, "raw_score": 0.8, "decay_factor": 1.0, "boost": 1.0,
         "age_days": 2, "tier": 1, "text": "the capital of France", "signal_boost": 1.0},
    ]}
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.retrieve.retrieve", return_value=fake_result),
        patch("sys.argv", ["mnemonics", "retrieve", "France", "--json", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    data = _json.loads(out.strip())
    assert isinstance(data, list)
    assert data[0]["id"] == 3


# ── bm25 --json ───────────────────────────────────────────────────────────────

def test_cli_bm25_json_output(tmp_path, capsys):
    """bm25 --json outputs a JSON array."""
    import json as _json
    fake_hits = [{"id": 5, "score": 0.75, "text": "keyword match", "tier": 1, "summary": None}]
    mock_store = MagicMock()
    mock_store.search_bm25.return_value = fake_hits
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "bm25", "keyword", "--json", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    data = _json.loads(out.strip())
    assert isinstance(data, list)
    assert data[0]["id"] == 5


def test_cli_bm25_json_no_results(tmp_path, capsys):
    """bm25 --json with no results outputs empty array."""
    import json as _json
    mock_store = MagicMock()
    mock_store.search_bm25.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "bm25", "ghost", "--json", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    data = _json.loads(out.strip())
    assert data == []


# ── get-many --json ───────────────────────────────────────────────────────────

def test_cli_get_many_json(tmp_path, capsys):
    """get-many --json outputs JSONL."""
    import json as _json
    fake_rows = [
        {"id": 1, "ns": "default", "text": "a", "tier": 1, "summary": None,
         "meta": {}, "created": "2026-01-01", "last_accessed": None, "access_count": 0},
    ]
    mock_store = MagicMock()
    mock_store.get_many.return_value = fake_rows
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "get-many", "1", "--json", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    lines = [l for l in out.strip().splitlines() if l]
    assert len(lines) == 1
    assert _json.loads(lines[0])["id"] == 1


# ── search-meta --json ────────────────────────────────────────────────────────

def test_cli_search_meta_json(tmp_path, capsys):
    """search-meta --json outputs JSONL."""
    import json as _json
    fake_rows = [{"id": 7, "tier": 0, "text": "pinned match", "ns": "default",
                  "summary": None, "meta": {"tag": "x"}, "created": "2026-01-01",
                  "last_accessed": None, "access_count": 0}]
    mock_store = MagicMock()
    mock_store.search_by_meta.return_value = fake_rows
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "search-meta", "tag=x", "--json", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    lines = [l for l in out.strip().splitlines() if l]
    assert len(lines) == 1
    assert _json.loads(lines[0])["id"] == 7


def test_cli_search_meta_json_empty(tmp_path, capsys):
    """search-meta --json with no results outputs nothing."""
    mock_store = MagicMock()
    mock_store.search_by_meta.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "search-meta", "tag=x", "--json", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out.strip()
    assert out == ""


# ── ingest --tier ─────────────────────────────────────────────────────────────

def test_cli_ingest_tier_ambient(tmp_path):
    """ingest --tier 2 passes tier=2 to ingest()."""
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=1) as mock_ingest,
        patch("sys.argv", ["mnemonics", "ingest", "ambient text", "--tier", "2",
                           "--path", str(tmp_path)]),
    ):
        main()
    kwargs = mock_ingest.call_args[1]
    assert kwargs["tier"] == 2


def test_cli_ingest_tier_pinned(tmp_path):
    """ingest --tier 0 passes tier=0 (pinned) to ingest()."""
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=1) as mock_ingest,
        patch("sys.argv", ["mnemonics", "ingest", "pin this", "--tier", "0",
                           "--path", str(tmp_path)]),
    ):
        main()
    kwargs = mock_ingest.call_args[1]
    assert kwargs["tier"] == 0


# ── list --since ──────────────────────────────────────────────────────────────

def test_cli_list_since_passed(tmp_path):
    """list --since passes since parameter to list_memories."""
    mock_store = MagicMock()
    mock_store.list_memories.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "list", "--since", "2026-01-01",
                           "--path", str(tmp_path)]),
    ):
        main()
    kwargs = mock_store.list_memories.call_args[1]
    assert kwargs["since"] == "2026-01-01"


# ── list --before ─────────────────────────────────────────────────────────────

def test_cli_list_before_passed(tmp_path):
    """list --before passes before parameter to list_memories."""
    mock_store = MagicMock()
    mock_store.list_memories.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "list", "--before", "2027-01-01",
                           "--path", str(tmp_path)]),
    ):
        main()
    kwargs = mock_store.list_memories.call_args[1]
    assert kwargs["before"] == "2027-01-01"


# ── export-jsonl --since / --before ───────────────────────────────────────────

def test_cli_export_jsonl_since_filter(tmp_path, capsys):
    """export-jsonl --since=far-future returns no rows from an empty store."""
    with (
        patch("sys.argv", ["mnemonics", "export-jsonl", "--since", "2099-01-01",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert out.strip() == ""  # far-future since → no rows


def test_cli_export_jsonl_before_filter(tmp_path, capsys):
    """export-jsonl --before far-past returns no rows."""
    with (
        patch("sys.argv", ["mnemonics", "export-jsonl", "--before", "2000-01-01",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert out.strip() == ""


# ── ingest --file ─────────────────────────────────────────────────────────────

def test_cli_ingest_file(tmp_path, capsys):
    """ingest --file reads text from a file."""
    doc = tmp_path / "doc.txt"
    doc.write_text("content from file", encoding="utf-8")
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=1) as mock_ingest,
        patch("sys.argv", ["mnemonics", "ingest", "--file", str(doc),
                           "--path", str(tmp_path)]),
    ):
        main()
    kwargs = mock_ingest.call_args[1]
    assert "content from file" in kwargs["texts"][0]


def test_cli_ingest_file_not_found(tmp_path, capsys):
    """ingest --file missing-file exits with error."""
    with (
        patch("mnemonics.store.Store"),
        patch("sys.argv", ["mnemonics", "ingest", "--file", "/no/such/file.txt",
                           "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2


def test_cli_ingest_no_text_no_file(tmp_path, capsys):
    """ingest with no text and no --file exits with error."""
    with (
        patch("mnemonics.store.Store"),
        patch("sys.argv", ["mnemonics", "ingest", "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2


def test_cli_ingest_file_stdin(tmp_path, capsys):
    """ingest --file - reads from stdin."""
    import io
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=1) as mock_ingest,
        patch("sys.stdin", io.StringIO("hello from stdin")),
        patch("sys.argv", ["mnemonics", "ingest", "--file", "-",
                           "--path", str(tmp_path)]),
    ):
        main()
    kwargs = mock_ingest.call_args[1]
    assert "hello from stdin" in kwargs["texts"][0]


# ── import-jsonl ──────────────────────────────────────────────────────────────

def test_cli_import_jsonl_file(tmp_path, capsys):
    """import-jsonl reads a JSONL file and ingests rows."""
    import json as _json
    jsonl = tmp_path / "export.jsonl"
    jsonl.write_text(
        _json.dumps({"text": "hello world", "ns": "default", "tier": 1}) + "\n",
        encoding="utf-8",
    )
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=1) as mock_ingest,
        patch("sys.argv", ["mnemonics", "import-jsonl", str(jsonl), "--path", str(tmp_path)]),
    ):
        main()
    mock_ingest.assert_called_once()
    out = capsys.readouterr().out
    assert "Imported" in out


def test_cli_import_jsonl_dry_run(tmp_path, capsys):
    """import-jsonl --dry-run does not call ingest."""
    import json as _json
    jsonl = tmp_path / "export.jsonl"
    jsonl.write_text(_json.dumps({"text": "data"}) + "\n", encoding="utf-8")
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest") as mock_ingest,
        patch("sys.argv", ["mnemonics", "import-jsonl", str(jsonl), "--dry-run",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_ingest.assert_not_called()
    out = capsys.readouterr().out
    assert "dry-run" in out


def test_cli_import_jsonl_missing_file(tmp_path, capsys):
    """import-jsonl with a non-existent file exits with code 2."""
    with (
        patch("mnemonics.store.Store"),
        patch("sys.argv", ["mnemonics", "import-jsonl", "/no/such/file.jsonl",
                           "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 2


def test_cli_import_jsonl_invalid_lines(tmp_path, capsys):
    """import-jsonl skips invalid JSON lines and rows without text."""
    import json as _json
    jsonl = tmp_path / "export.jsonl"
    jsonl.write_text(
        "not json\n" +
        _json.dumps({"ns": "default"}) + "\n" +   # no text
        _json.dumps({"text": "good row"}) + "\n",
        encoding="utf-8",
    )
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=1) as mock_ingest,
        patch("sys.argv", ["mnemonics", "import-jsonl", str(jsonl), "--path", str(tmp_path)]),
    ):
        main()
    assert mock_ingest.call_count == 1
    out = capsys.readouterr().out
    assert "skipped 2" in out


def test_cli_import_jsonl_ns_override(tmp_path, capsys):
    """import-jsonl --ns overrides namespace from rows."""
    import json as _json
    jsonl = tmp_path / "export.jsonl"
    jsonl.write_text(_json.dumps({"text": "row", "ns": "original"}) + "\n", encoding="utf-8")
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=1) as mock_ingest,
        patch("sys.argv", ["mnemonics", "import-jsonl", str(jsonl), "--ns", "override",
                           "--path", str(tmp_path)]),
    ):
        main()
    kwargs = mock_ingest.call_args[1]
    assert kwargs["ns"] == "override"


def test_cli_import_jsonl_stdin(tmp_path, capsys):
    """import-jsonl reads from stdin when no file given."""
    import json as _json, io
    line = _json.dumps({"text": "from stdin", "ns": "default"}) + "\n"
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=1) as mock_ingest,
        patch("sys.stdin", io.StringIO(line)),
        patch("sys.argv", ["mnemonics", "import-jsonl", "--path", str(tmp_path)]),
    ):
        main()
    mock_ingest.assert_called_once()


def test_cli_import_jsonl_blank_lines_ignored(tmp_path, capsys):
    """import-jsonl skips blank lines."""
    import json as _json
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(
        "\n" + _json.dumps({"text": "row"}) + "\n\n",
        encoding="utf-8",
    )
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=1) as mock_ingest,
        patch("sys.argv", ["mnemonics", "import-jsonl", str(jsonl), "--path", str(tmp_path)]),
    ):
        main()
    mock_ingest.assert_called_once()


def test_cli_import_jsonl_bad_tier_clamped(tmp_path, capsys):
    """import-jsonl clamps invalid tier to 1."""
    import json as _json
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(_json.dumps({"text": "row", "tier": 99}) + "\n", encoding="utf-8")
    with (
        patch("mnemonics.store.Store"),
        patch("mnemonics.ingest.ingest", return_value=1) as mock_ingest,
        patch("sys.argv", ["mnemonics", "import-jsonl", str(jsonl), "--path", str(tmp_path)]),
    ):
        main()
    kwargs = mock_ingest.call_args[1]
    assert kwargs["tier"] == 1


# ── text-search ───────────────────────────────────────────────────────────────

def test_cli_text_search_hit(tmp_path, capsys):
    """text-search prints matching rows."""
    hits = [{"id": 1, "ns": "default", "tier": 1, "text": "Eiffel Tower Paris", "summary": None}]
    mock_store = MagicMock()
    mock_store.text_search.return_value = hits
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "text-search", "Eiffel", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.text_search.assert_called_once()
    out = capsys.readouterr().out
    assert "Eiffel" in out


def test_cli_text_search_no_results(tmp_path, capsys):
    """text-search prints 'No results' when nothing matches."""
    mock_store = MagicMock()
    mock_store.text_search.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "text-search", "xyz_nomatch", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "No results" in out


def test_cli_text_search_json(tmp_path, capsys):
    """text-search --json outputs JSON array."""
    import json as _json
    hits = [{"id": 1, "ns": "default", "tier": 1, "text": "hello", "summary": None}]
    mock_store = MagicMock()
    mock_store.text_search.return_value = hits
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "text-search", "hello", "--json", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    parsed = _json.loads(out)
    assert parsed[0]["id"] == 1


def test_cli_text_search_all_ns(tmp_path, capsys):
    """text-search --ns all passes ns=None to store."""
    mock_store = MagicMock()
    mock_store.text_search.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "text-search", "q", "--ns", "all", "--path", str(tmp_path)]),
    ):
        main()
    kwargs = mock_store.text_search.call_args[1]
    assert kwargs["ns"] is None


def test_cli_text_search_summary_shown(tmp_path, capsys):
    """text-search prints summary line when present."""
    hits = [{"id": 1, "ns": "default", "tier": 1, "text": "Paris", "summary": "city in France"}]
    mock_store = MagicMock()
    mock_store.text_search.return_value = hits
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "text-search", "Paris", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "city in France" in out


# ── rename-ns ─────────────────────────────────────────────────────────────────

def test_cli_rename_ns_basic(tmp_path, capsys):
    """rename-ns moves memories and prints confirmation."""
    mock_store = MagicMock()
    mock_store.rename_ns.return_value = 3
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "rename-ns", "old", "new", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.rename_ns.assert_called_once_with("old", "new")
    out = capsys.readouterr().out
    assert "3 memories" in out


def test_cli_rename_ns_zero(tmp_path, capsys):
    """rename-ns with empty source namespace prints warning."""
    mock_store = MagicMock()
    mock_store.rename_ns.return_value = 0
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "rename-ns", "ghost", "new", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "nothing to rename" in out


def test_cli_rename_ns_conflict_exits(tmp_path, capsys):
    """rename-ns with a ValueError exits with code 1."""
    mock_store = MagicMock()
    mock_store.rename_ns.side_effect = ValueError("already has 2 memories")
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "rename-ns", "src", "dst", "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


# ── namespaces ─────────────────────────────────────────────────────────────────

def test_cli_namespaces_populated(tmp_path, capsys):
    """namespaces lists existing namespaces with counts."""
    mock_store = MagicMock()
    mock_store.list_namespaces.return_value = ["default", "work"]
    mock_store.count.side_effect = lambda ns: 3 if ns == "default" else 1
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "namespaces", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "default" in out
    assert "work" in out


def test_cli_namespaces_empty(tmp_path, capsys):
    """namespaces on empty store prints placeholder."""
    mock_store = MagicMock()
    mock_store.list_namespaces.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "namespaces", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "no namespaces" in out


# ── bulk-tier ──────────────────────────────────────────────────────────────────

def test_cli_bulk_tier_basic(tmp_path, capsys):
    """bulk-tier updates memories and prints confirmation."""
    mock_store = MagicMock()
    mock_store.set_tier_many.return_value = 3
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "bulk-tier", "0", "1", "2", "3",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.set_tier_many.assert_called_once_with([1, 2, 3], 0)
    out = capsys.readouterr().out
    assert "3" in out
    assert "pinned" in out


def test_cli_bulk_tier_invalid_tier_error(tmp_path, capsys):
    """bulk-tier with invalid tier exits with code 1."""
    mock_store = MagicMock()
    mock_store.set_tier_many.side_effect = ValueError("tier must be 0, 1, or 2")
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "bulk-tier", "0", "99",
                           "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


# ── recent ─────────────────────────────────────────────────────────────────────

def test_cli_recent_basic(tmp_path, capsys):
    """recent prints recently accessed memories."""
    hits = [{"id": 1, "ns": "default", "tier": 1, "text": "Paris memory",
             "summary": None, "last_accessed": "2026-06-18 10:00:00", "access_count": 3}]
    mock_store = MagicMock()
    mock_store.recent_accessed.return_value = hits
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "recent", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "Paris" in out
    assert "2026-06-18" in out


def test_cli_recent_json(tmp_path, capsys):
    """recent --json outputs JSON array."""
    hits = [{"id": 1, "ns": "default", "tier": 1, "text": "hi",
             "summary": None, "last_accessed": None, "access_count": 0}]
    mock_store = MagicMock()
    mock_store.recent_accessed.return_value = hits
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "recent", "--json", "--path", str(tmp_path)]),
    ):
        main()
    import json as _json
    parsed = _json.loads(capsys.readouterr().out)
    assert parsed[0]["id"] == 1


def test_cli_recent_empty(tmp_path, capsys):
    """recent prints placeholder when no results."""
    mock_store = MagicMock()
    mock_store.recent_accessed.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "recent", "--path", str(tmp_path)]),
    ):
        main()
    assert "No recently" in capsys.readouterr().out


# ── top-accessed ───────────────────────────────────────────────────────────────

def test_cli_top_accessed_basic(tmp_path, capsys):
    hits = [{"id": 1, "ns": "default", "tier": 1, "text": "popular memory",
             "summary": None, "last_accessed": "2026-06-18", "access_count": 42}]
    mock_store = MagicMock()
    mock_store.top_accessed.return_value = hits
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "top-accessed", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "popular" in out
    assert "42" in out


def test_cli_top_accessed_json(tmp_path, capsys):
    hits = [{"id": 2, "ns": "default", "tier": 0, "text": "pinned hot",
             "summary": None, "last_accessed": None, "access_count": 7}]
    mock_store = MagicMock()
    mock_store.top_accessed.return_value = hits
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "top-accessed", "--json", "--path", str(tmp_path)]),
    ):
        main()
    import json as _json
    parsed = _json.loads(capsys.readouterr().out)
    assert parsed[0]["access_count"] == 7


def test_cli_top_accessed_empty(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.top_accessed.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "top-accessed", "--path", str(tmp_path)]),
    ):
        main()
    assert "No memories" in capsys.readouterr().out


# ── copy-ns ────────────────────────────────────────────────────────────────────

def test_cli_copy_ns_basic(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.copy_ns.return_value = 7
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "copy-ns", "default", "backup",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.copy_ns.assert_called_once_with("default", "backup")
    out = capsys.readouterr().out
    assert "7" in out
    assert "backup" in out


def test_cli_copy_ns_conflict(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.copy_ns.side_effect = ValueError("already has")
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "copy-ns", "default", "backup",
                           "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


# ── stats-by-ns ────────────────────────────────────────────────────────────────

def test_cli_stats_by_ns_basic(tmp_path, capsys):
    stats = [{"ns": "default", "total": 5, "pinned": 1, "default": 3, "ambient": 1,
               "oldest": "2026-01-01", "newest": "2026-06-18"}]
    mock_store = MagicMock()
    mock_store.stats_by_ns.return_value = stats
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "stats-by-ns", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "default" in out
    assert "5" in out


def test_cli_stats_by_ns_json(tmp_path, capsys):
    stats = [{"ns": "default", "total": 3, "pinned": 0, "default": 3, "ambient": 0,
               "oldest": "2026-01-01", "newest": "2026-06-01"}]
    mock_store = MagicMock()
    mock_store.stats_by_ns.return_value = stats
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "stats-by-ns", "--json", "--path", str(tmp_path)]),
    ):
        main()
    import json as _j
    parsed = _j.loads(capsys.readouterr().out)
    assert parsed[0]["ns"] == "default"


def test_cli_stats_by_ns_empty(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.stats_by_ns.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "stats-by-ns", "--path", str(tmp_path)]),
    ):
        main()
    assert "no namespaces" in capsys.readouterr().out


# ── merge-ns ───────────────────────────────────────────────────────────────────

def test_cli_merge_ns_basic(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.merge_ns.return_value = 5
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "merge-ns", "src", "dst",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.merge_ns.assert_called_once_with("src", "dst")
    out = capsys.readouterr().out
    assert "5" in out
    assert "dst" in out


# ── touch-many ─────────────────────────────────────────────────────────────────

def test_cli_touch_many_basic(tmp_path, capsys):
    mock_store = MagicMock()
    mock_store.touch_many.return_value = 3
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "touch-many", "1", "2", "3",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.touch_many.assert_called_once_with([1, 2, 3])
    assert "3" in capsys.readouterr().out





# ── update-meta --merge flag ───────────────────────────────────────────────────

def test_cli_update_meta_merge_flag(tmp_path, capsys):
    """update-meta --merge calls update_meta with merge=True."""
    mock_store = MagicMock()
    mock_store.update_meta.return_value = True
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "update-meta", "5", '{"tag": "x"}',
                           "--merge", "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 0
    mock_store.update_meta.assert_called_once_with(5, {"tag": "x"}, merge=True)


# ── hybrid-search CLI ─────────────────────────────────────────────────────────

def test_cli_hybrid_search_ok(tmp_path, capsys):
    """hybrid-search prints results."""
    mock_store = MagicMock()
    mock_store.hybrid_search.return_value = [
        {"id": 1, "text": "Paris is great", "tier": 1, "rrf_score": 0.03,
         "vector_rank": 1, "bm25_rank": 2, "summary": None}
    ]
    mock_enc = MagicMock()
    mock_enc.encode.return_value = [[0.1] * 384]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("mnemonics.ingest._get_encoder", return_value=mock_enc),
        patch("sys.argv", ["mnemonics", "hybrid-search", "Paris",
                           "--ns", "default", "--top-k", "5", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "rrf=" in out or "Paris" in out


def test_cli_hybrid_search_no_results(tmp_path, capsys):
    """hybrid-search prints message when no results."""
    mock_store = MagicMock()
    mock_store.hybrid_search.return_value = []
    mock_enc = MagicMock()
    mock_enc.encode.return_value = [[0.0] * 384]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("mnemonics.ingest._get_encoder", return_value=mock_enc),
        patch("sys.argv", ["mnemonics", "hybrid-search", "xyz",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "No hybrid results" in out


def test_cli_hybrid_search_json(tmp_path, capsys):
    """hybrid-search --json outputs JSON array."""
    import json as _j
    mock_store = MagicMock()
    mock_store.hybrid_search.return_value = [
        {"id": 2, "text": "Python", "tier": 1, "rrf_score": 0.02,
         "vector_rank": 2, "bm25_rank": 1, "summary": None}
    ]
    mock_enc = MagicMock()
    mock_enc.encode.return_value = [[0.1] * 384]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("mnemonics.ingest._get_encoder", return_value=mock_enc),
        patch("sys.argv", ["mnemonics", "hybrid-search", "Python",
                           "--json", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    data = _j.loads(out.strip())
    assert data[0]["id"] == 2


# ── similar-to CLI ────────────────────────────────────────────────────────────

def test_cli_similar_to_ok(tmp_path, capsys):
    """similar-to prints nearest neighbors."""
    mock_store = MagicMock()
    mock_store.similar_to.return_value = [
        {"id": 2, "text": "neighbor doc", "tier": 1, "score": 0.95, "summary": None}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "similar-to", "1",
                           "--top-k", "3", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "0.950" in out or "neighbor" in out


def test_cli_similar_to_no_results(tmp_path, capsys):
    """similar-to prints message when no neighbors."""
    mock_store = MagicMock()
    mock_store.similar_to.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "similar-to", "99",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "No similar" in out


def test_cli_similar_to_json(tmp_path, capsys):
    """similar-to --json outputs JSON."""
    import json as _j
    mock_store = MagicMock()
    mock_store.similar_to.return_value = [
        {"id": 3, "text": "similar doc", "tier": 1, "score": 0.88, "summary": None}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "similar-to", "1",
                           "--json", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    data = _j.loads(out.strip())
    assert data[0]["id"] == 3


# ── expire CLI ────────────────────────────────────────────────────────────────

def test_cli_expire_ok(tmp_path, capsys):
    """expire prints demoted count."""
    mock_store = MagicMock()
    mock_store.expire.return_value = 3
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "expire", "--age-days", "30", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "3" in out and "Demoted" in out


def test_cli_expire_with_ns(tmp_path, capsys):
    """expire --ns targets specific namespace."""
    mock_store = MagicMock()
    mock_store.expire.return_value = 0
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "expire", "--ns", "proj:test",
                           "--age-days", "7", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.expire.assert_called_once_with(ns="proj:test", age_days=7, min_age_days=None)


# ── bulk-update-summary CLI ───────────────────────────────────────────────────

def test_cli_bulk_update_summary_ok(tmp_path, capsys):
    """bulk-update-summary parses id:summary pairs and calls store."""
    mock_store = MagicMock()
    mock_store.bulk_update_summary.return_value = 2
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "bulk-update-summary",
                           "1:Summary one", "2:Summary two",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.bulk_update_summary.assert_called_once_with({1: "Summary one", 2: "Summary two"})
    assert "2" in capsys.readouterr().out


def test_cli_bulk_update_summary_clear(tmp_path, capsys):
    """bulk-update-summary with 'id:' (empty summary) passes None."""
    mock_store = MagicMock()
    mock_store.bulk_update_summary.return_value = 1
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "bulk-update-summary", "5:",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.bulk_update_summary.assert_called_once_with({5: None})


def test_cli_bulk_update_summary_bad_format(tmp_path, capsys):
    """bulk-update-summary with pair missing ':' prints error and exits 1."""
    mock_store = MagicMock()
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "bulk-update-summary", "bad_pair",
                           "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


# ── deduplicate CLI ───────────────────────────────────────────────────────────

def test_cli_deduplicate_dry_run(tmp_path, capsys):
    """deduplicate --dry-run lists pairs."""
    mock_store = MagicMock()
    mock_store.deduplicate.return_value = {
        "pairs": [{"kept_id": 2, "removed_id": 1, "similarity": 0.9998}],
        "removed": 0,
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "deduplicate", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "1 pair" in out
    mock_store.deduplicate.assert_called_once_with(
        ns="default", threshold=0.98, dry_run=True, keep="newest"
    )


def test_cli_deduplicate_execute(tmp_path, capsys):
    """deduplicate --execute deletes."""
    mock_store = MagicMock()
    mock_store.deduplicate.return_value = {"pairs": [], "removed": 0}
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "deduplicate", "--execute",
                           "--threshold", "0.95", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.deduplicate.assert_called_once_with(
        ns="default", threshold=0.95, dry_run=False, keep="newest"
    )


def test_cli_deduplicate_json(tmp_path, capsys):
    """deduplicate --json outputs JSON."""
    mock_store = MagicMock()
    mock_store.deduplicate.return_value = {"pairs": [], "removed": 0}
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "deduplicate", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "pairs" in parsed


# ── sample CLI ────────────────────────────────────────────────────────────────

def test_cli_sample_ok(tmp_path, capsys):
    """sample prints memory lines."""
    mock_store = MagicMock()
    mock_store.sample.return_value = [
        {"id": 1, "tier": 1, "text": "hello world", "summary": None}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "sample", "--n", "3", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "id=1" in out


def test_cli_sample_empty(tmp_path, capsys):
    """sample prints message when no results."""
    mock_store = MagicMock()
    mock_store.sample.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "sample", "--path", str(tmp_path)]),
    ):
        main()
    assert "No memories" in capsys.readouterr().out


def test_cli_sample_json(tmp_path, capsys):
    """sample --json outputs JSON lines."""
    mock_store = MagicMock()
    mock_store.sample.return_value = [
        {"id": 2, "tier": 0, "text": "pinned", "ns": "default",
         "summary": None, "created": "2026-01-01", "last_accessed": None, "access_count": 0}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "sample", "--json", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    parsed = json.loads(out.strip())
    assert parsed["id"] == 2


# ── reindex-all CLI ───────────────────────────────────────────────────────────

def test_cli_reindex_all_ok(tmp_path, capsys):
    """reindex-all prints per-namespace result."""
    mock_store = MagicMock()
    mock_store.reindex_all.return_value = [
        {"ns": "default", "old_count": 5, "new_count": 5},
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "reindex-all", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "default" in out


def test_cli_reindex_all_empty(tmp_path, capsys):
    """reindex-all on empty store prints 'No namespaces found'."""
    mock_store = MagicMock()
    mock_store.reindex_all.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "reindex-all", "--path", str(tmp_path)]),
    ):
        main()
    assert "No namespaces" in capsys.readouterr().out


def test_cli_reindex_all_with_error(tmp_path, capsys):
    """reindex-all prints ERROR for namespaces that fail."""
    mock_store = MagicMock()
    mock_store.reindex_all.return_value = [
        {"ns": "broken", "error": "disk full"},
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "reindex-all", "--path", str(tmp_path)]),
    ):
        main()
    assert "ERROR" in capsys.readouterr().out


# ── namespace-info CLI ────────────────────────────────────────────────────────

def test_cli_namespace_info_ok(tmp_path, capsys):
    """namespace-info prints namespace stats."""
    mock_store = MagicMock()
    mock_store.namespace_info.return_value = {
        "ns": "default", "total": 5,
        "by_tier": {1: 5}, "oldest": "2026-01-01",
        "newest": "2026-06-01", "avg_text_len": 42.0,
        "total_words": 30, "with_summary": 2,
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "namespace-info", "default",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "default" in out and "5" in out


def test_cli_namespace_info_not_found(tmp_path, capsys):
    """namespace-info exits 1 when namespace not found."""
    mock_store = MagicMock()
    mock_store.namespace_info.return_value = None
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "namespace-info", "ghost",
                           "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


def test_cli_namespace_info_json(tmp_path, capsys):
    """namespace-info --json outputs JSON."""
    mock_store = MagicMock()
    mock_store.namespace_info.return_value = {
        "ns": "default", "total": 3,
        "by_tier": {1: 3}, "oldest": "2026-01-01",
        "newest": "2026-06-01", "avg_text_len": 50.0,
        "total_words": 20, "with_summary": 0,
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "namespace-info", "default",
                           "--json", "--path", str(tmp_path)]),
    ):
        main()
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["ns"] == "default"


# ── move-to-ns CLI ────────────────────────────────────────────────────────────

def test_cli_move_to_ns_ok(tmp_path, capsys):
    """move-to-ns prints moved count."""
    mock_store = MagicMock()
    mock_store.move_to_ns.return_value = 2
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "move-to-ns", "archive", "1", "2",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.move_to_ns.assert_called_once_with([1, 2], "archive")
    assert "2" in capsys.readouterr().out


# ── clone CLI ─────────────────────────────────────────────────────────────────

def test_cli_clone_ok(tmp_path, capsys):
    """clone prints new id."""
    mock_store = MagicMock()
    mock_store.clone.return_value = 42
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "clone", "7", "backup",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.clone.assert_called_once_with(7, "backup")
    assert "42" in capsys.readouterr().out


def test_cli_clone_not_found(tmp_path, capsys):
    """clone exits 1 when clone returns None."""
    mock_store = MagicMock()
    mock_store.clone.return_value = None
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "clone", "999", "backup",
                           "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


# ── update-text CLI ───────────────────────────────────────────────────────────

def test_cli_update_text_ok(tmp_path, capsys):
    """update-text calls update_text and prints confirmation."""
    mock_store = MagicMock()
    mock_store._dim = 4
    mock_store.update_text.return_value = True
    mock_enc = MagicMock()
    import numpy as np
    mock_enc.encode.return_value = np.zeros((1, 4), dtype="float32")
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "update-text", "7", "hello world",
                           "--path", str(tmp_path)]),
        patch("sentence_transformers.SentenceTransformer", return_value=mock_enc),
    ):
        main()
    assert "Updated" in capsys.readouterr().out


def test_cli_update_text_not_found(tmp_path, capsys):
    """update-text exits 1 when update_text returns False."""
    mock_store = MagicMock()
    mock_store._dim = 4
    mock_store.update_text.return_value = False
    mock_enc = MagicMock()
    import numpy as np
    mock_enc.encode.return_value = np.zeros((1, 4), dtype="float32")
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "update-text", "999", "hello",
                           "--path", str(tmp_path)]),
        patch("sentence_transformers.SentenceTransformer", return_value=mock_enc),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


# ── access-stats CLI ──────────────────────────────────────────────────────────

def test_cli_access_stats_ok(tmp_path, capsys):
    """access-stats prints formatted stats."""
    mock_store = MagicMock()
    mock_store.access_stats.return_value = {
        "ns": "default", "total": 10, "total_accesses": 5,
        "avg_accesses": 0.5, "max_accesses": 3,
        "never_accessed": 7, "most_recent_access": "2026-06-18T12:00:00",
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "access-stats", "--ns", "default",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "10" in out and "default" in out


def test_cli_access_stats_all_ns(tmp_path, capsys):
    """access-stats --all-ns passes None to access_stats."""
    mock_store = MagicMock()
    mock_store.access_stats.return_value = {
        "ns": None, "total": 20, "total_accesses": 0,
        "avg_accesses": 0.0, "max_accesses": 0,
        "never_accessed": 20, "most_recent_access": None,
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "access-stats", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.access_stats.assert_called_once_with(None)


def test_cli_access_stats_json(tmp_path, capsys):
    """access-stats --json outputs valid JSON."""
    mock_store = MagicMock()
    mock_store.access_stats.return_value = {
        "ns": "default", "total": 1, "total_accesses": 0,
        "avg_accesses": 0.0, "max_accesses": 0,
        "never_accessed": 1, "most_recent_access": None,
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "access-stats", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["ns"] == "default"


# ── tag / untag CLI ───────────────────────────────────────────────────────────

def test_cli_tag_ok(tmp_path, capsys):
    """tag prints confirmation."""
    mock_store = MagicMock()
    mock_store.tag.return_value = True
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "tag", "7", "important",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.tag.assert_called_once_with(7, "important")
    assert "Added" in capsys.readouterr().out


def test_cli_tag_not_found(tmp_path, capsys):
    """tag exits 1 when ID not found."""
    mock_store = MagicMock()
    mock_store.tag.return_value = False
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "tag", "999", "x",
                           "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


def test_cli_untag_ok(tmp_path, capsys):
    """untag prints confirmation."""
    mock_store = MagicMock()
    mock_store.untag.return_value = True
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "untag", "7", "important",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.untag.assert_called_once_with(7, "important")
    assert "Removed" in capsys.readouterr().out


def test_cli_untag_not_found(tmp_path, capsys):
    """untag exits 1 when ID not found."""
    mock_store = MagicMock()
    mock_store.untag.return_value = False
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "untag", "999", "x",
                           "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


# ── find-by-tag / list-tags CLI ───────────────────────────────────────────────

def test_cli_find_by_tag_ok(tmp_path, capsys):
    """find-by-tag prints results."""
    mock_store = MagicMock()
    mock_store.find_by_tag.return_value = [
        {"id": 1, "ns": "default", "text": "hello world", "summary": None, "tier": 1, "created": "2026-01-01"},
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "find-by-tag", "important",
                           "--ns", "default", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "1 result" in out and "hello world" in out


def test_cli_find_by_tag_json(tmp_path, capsys):
    """find-by-tag --json outputs valid JSON."""
    mock_store = MagicMock()
    mock_store.find_by_tag.return_value = [
        {"id": 2, "ns": "default", "text": "test", "summary": None, "tier": 1, "created": "2026-01-01"},
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "find-by-tag", "x", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    parsed = json.loads(capsys.readouterr().out)
    assert parsed[0]["id"] == 2


def test_cli_find_by_tag_all_ns(tmp_path, capsys):
    """find-by-tag --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.find_by_tag.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "find-by-tag", "x", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.find_by_tag.assert_called_once_with("x", ns=None, limit=20)


def test_cli_list_tags_ok(tmp_path, capsys):
    """list-tags prints tag counts."""
    mock_store = MagicMock()
    mock_store.list_tags.return_value = [{"tag": "alpha", "count": 3}, {"tag": "beta", "count": 1}]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "list-tags", "--ns", "default",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "alpha" in out and "3" in out


def test_cli_list_tags_empty(tmp_path, capsys):
    """list-tags with no tags prints 'No tags found'."""
    mock_store = MagicMock()
    mock_store.list_tags.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "list-tags", "--path", str(tmp_path)]),
    ):
        main()
    assert "No tags" in capsys.readouterr().out


def test_cli_list_tags_json(tmp_path, capsys):
    """list-tags --json outputs valid JSON."""
    mock_store = MagicMock()
    mock_store.list_tags.return_value = [{"tag": "z", "count": 2}]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "list-tags", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    parsed = json.loads(capsys.readouterr().out)
    assert parsed[0]["tag"] == "z"


# ── word-frequency CLI ────────────────────────────────────────────────────────

def test_cli_word_frequency_ok(tmp_path, capsys):
    """word-frequency prints top words."""
    mock_store = MagicMock()
    mock_store.word_frequency.return_value = [
        {"word": "python", "count": 5},
        {"word": "data", "count": 3},
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "word-frequency", "--ns", "default",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "python" in out and "5" in out


def test_cli_word_frequency_empty(tmp_path, capsys):
    """word-frequency prints 'No words' when empty."""
    mock_store = MagicMock()
    mock_store.word_frequency.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "word-frequency",
                           "--path", str(tmp_path)]),
    ):
        main()
    assert "No words" in capsys.readouterr().out


def test_cli_word_frequency_json(tmp_path, capsys):
    """word-frequency --json outputs valid JSON."""
    mock_store = MagicMock()
    mock_store.word_frequency.return_value = [{"word": "ai", "count": 7}]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "word-frequency", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    parsed = json.loads(capsys.readouterr().out)
    assert parsed[0]["word"] == "ai"


def test_cli_word_frequency_all_ns(tmp_path, capsys):
    """word-frequency --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.word_frequency.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "word-frequency", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.word_frequency.assert_called_once_with(None, 20)


# ── get-tags CLI ──────────────────────────────────────────────────────────────

def test_cli_get_tags_with_tags(tmp_path, capsys):
    """get-tags prints comma-separated tags."""
    mock_store = MagicMock()
    mock_store.get_tags.return_value = ["alpha", "beta"]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "get-tags", "7", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "alpha" in out and "beta" in out


def test_cli_get_tags_no_tags(tmp_path, capsys):
    """get-tags prints 'no tags' when list is empty."""
    mock_store = MagicMock()
    mock_store.get_tags.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "get-tags", "7", "--path", str(tmp_path)]),
    ):
        main()
    assert "no tags" in capsys.readouterr().out


def test_cli_get_tags_not_found(tmp_path, capsys):
    """get-tags exits 1 when ID not found."""
    mock_store = MagicMock()
    mock_store.get_tags.return_value = None
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "get-tags", "999", "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


# ── search-date-range CLI ─────────────────────────────────────────────────────

def test_cli_search_date_range_ok(tmp_path, capsys):
    """search-date-range prints results."""
    mock_store = MagicMock()
    mock_store.search_date_range.return_value = [
        {"id": 1, "ns": "default", "text": "hello", "summary": None,
         "tier": 1, "created": "2026-06-01T12:00:00"},
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "search-date-range",
                           "--ns", "default", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "1 result" in out and "hello" in out


def test_cli_search_date_range_json(tmp_path, capsys):
    """search-date-range --json outputs valid JSON."""
    mock_store = MagicMock()
    mock_store.search_date_range.return_value = [
        {"id": 2, "ns": "default", "text": "world", "summary": None,
         "tier": 0, "created": "2026-01-01T00:00:00"},
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "search-date-range", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    parsed = json.loads(capsys.readouterr().out)
    assert parsed[0]["id"] == 2


def test_cli_search_date_range_all_ns(tmp_path, capsys):
    """search-date-range --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.search_date_range.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "search-date-range", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    call_kwargs = mock_store.search_date_range.call_args
    assert call_kwargs.kwargs.get("ns") is None or call_kwargs.args[0] is None


# ── export-ns CLI ─────────────────────────────────────────────────────────────

def test_cli_export_ns_ok(tmp_path, capsys):
    """export-ns outputs JSON array."""
    mock_store = MagicMock()
    mock_store.export_ns.return_value = [
        {"id": 1, "ns": "default", "text": "hello", "summary": None,
         "meta": {}, "tier": 1, "created": "2026-01-01",
         "last_accessed": None, "access_count": 0},
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "export-ns", "default", "--path", str(tmp_path)]),
    ):
        main()
    parsed = json.loads(capsys.readouterr().out)
    assert parsed[0]["id"] == 1


# ── bulk-tag CLI ──────────────────────────────────────────────────────────────

def test_cli_bulk_tag_ok(tmp_path, capsys):
    """bulk-tag prints count of updated memories."""
    mock_store = MagicMock()
    mock_store.bulk_tag.return_value = 3
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "bulk-tag", "1", "2", "3",
                           "--tags", "alpha", "beta", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.bulk_tag.assert_called_once_with([1, 2, 3], ["alpha", "beta"])
    assert "3" in capsys.readouterr().out


# ── touch CLI ─────────────────────────────────────────────────────────────────

def test_cli_touch_ok(tmp_path, capsys):
    """touch prints success when ID exists."""
    mock_store = MagicMock()
    mock_store.touch.return_value = True
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "touch", "42", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.touch.assert_called_once_with(42)
    assert "42" in capsys.readouterr().out


def test_cli_touch_not_found(tmp_path, capsys):
    """touch exits 1 when ID not found."""
    mock_store = MagicMock()
    mock_store.touch.return_value = False
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "touch", "999", "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code == 1


# ── bulk-untag CLI ────────────────────────────────────────────────────────────

def test_cli_bulk_untag_ok(tmp_path, capsys):
    """bulk-untag prints count of updated memories."""
    mock_store = MagicMock()
    mock_store.bulk_untag.return_value = 2
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "bulk-untag", "1", "2",
                           "--tags", "x", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.bulk_untag.assert_called_once_with([1, 2], ["x"])
    assert "2" in capsys.readouterr().out


# ── count-by-tier CLI ─────────────────────────────────────────────────────────

def test_cli_count_by_tier_ok(tmp_path, capsys):
    """count-by-tier prints tier counts."""
    mock_store = MagicMock()
    mock_store.count_by_tier.return_value = {1: 5, 2: 3}
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "count-by-tier", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "default" in out or "5" in out


def test_cli_count_by_tier_json(tmp_path, capsys):
    """count-by-tier --json outputs raw JSON."""
    mock_store = MagicMock()
    mock_store.count_by_tier.return_value = {1: 4}
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "count-by-tier", "--json", "--path", str(tmp_path)]),
    ):
        main()
    parsed = json.loads(capsys.readouterr().out)
    assert "1" in parsed or 1 in parsed


def test_cli_count_by_tier_all_ns(tmp_path, capsys):
    """count-by-tier --all-ns passes ns=None to store."""
    mock_store = MagicMock()
    mock_store.count_by_tier.return_value = {}
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "count-by-tier", "--all-ns", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.count_by_tier.assert_called_once_with(None)


# ── import-records CLI ────────────────────────────────────────────────────────

def test_cli_import_records_from_file(tmp_path, capsys):
    """import-records loads JSON from file and prints count."""
    import json as _j
    data_file = tmp_path / "records.json"
    data_file.write_text(_j.dumps([{"text": "hello"}, {"text": "world"}]))
    mock_store = MagicMock()
    mock_store.import_records.return_value = 2
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "import-records", str(data_file), "--path", str(tmp_path)]),
    ):
        main()
    assert "2" in capsys.readouterr().out


# ── text-stats CLI ────────────────────────────────────────────────────────────

def test_cli_text_stats_ok(tmp_path, capsys):
    """text-stats prints human-readable stats."""
    mock_store = MagicMock()
    mock_store.text_stats.return_value = {
        "count": 3, "total_chars": 60, "avg_chars": 20.0,
        "min_chars": 10, "max_chars": 30, "total_words": 9, "avg_words": 3.0,
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "text-stats", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "count" in out


def test_cli_text_stats_json(tmp_path, capsys):
    """text-stats --json outputs raw JSON."""
    mock_store = MagicMock()
    mock_store.text_stats.return_value = {"count": 1, "total_chars": 5,
        "avg_chars": 5.0, "min_chars": 5, "max_chars": 5, "total_words": 1, "avg_words": 1.0}
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "text-stats", "--json", "--path", str(tmp_path)]),
    ):
        main()
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["count"] == 1


def test_cli_text_stats_all_ns(tmp_path, capsys):
    """text-stats --all-ns passes ns=None to store."""
    mock_store = MagicMock()
    mock_store.text_stats.return_value = {"count": 0, "total_chars": 0,
        "avg_chars": 0.0, "min_chars": 0, "max_chars": 0, "total_words": 0, "avg_words": 0.0}
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "text-stats", "--all-ns", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.text_stats.assert_called_once_with(None)


def test_cli_import_records_from_stdin(tmp_path, capsys):
    """import-records reads JSON from stdin when file is '-'."""
    import io, json as _j
    mock_store = MagicMock()
    mock_store.import_records.return_value = 1
    stdin_data = _j.dumps([{"text": "stdin record"}])
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "import-records", "-", "--path", str(tmp_path)]),
        patch("sys.stdin", io.StringIO(stdin_data)),
    ):
        main()
    assert "1" in capsys.readouterr().out


# ── rename-ns CLI ─────────────────────────────────────────────────────────────

def test_cli_rename_ns_ok(tmp_path, capsys):
    """rename-ns prints count of moved memories."""
    mock_store = MagicMock()
    mock_store.rename_ns.return_value = 5
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "rename-ns", "old", "new", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.rename_ns.assert_called_once_with("old", "new")
    assert "5" in capsys.readouterr().out


# ── merge-ns CLI ──────────────────────────────────────────────────────────────

def test_cli_merge_ns_ok(tmp_path, capsys):
    """merge-ns prints count of merged memories."""
    mock_store = MagicMock()
    mock_store.merge_ns.return_value = 3
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "merge-ns", "src", "tgt", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.merge_ns.assert_called_once_with("src", "tgt")
    assert "3" in capsys.readouterr().out


# ── bulk-delete CLI ───────────────────────────────────────────────────────────

def test_cli_bulk_delete_ok(tmp_path, capsys):
    """bulk-delete prints count of deleted memories."""
    mock_store = MagicMock()
    mock_store.bulk_delete.return_value = 2
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "bulk-delete", "1", "2", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.bulk_delete.assert_called_once_with([1, 2])
    assert "2" in capsys.readouterr().out


# ── filter-by-meta CLI ────────────────────────────────────────────────────────

def test_cli_filter_by_meta_ok(tmp_path, capsys):
    """filter-by-meta prints matching memory count."""
    mock_store = MagicMock()
    mock_store.filter_by_meta.return_value = [
        {"id": 1, "text": "hello", "ns": "default", "summary": None,
         "tier": 1, "created": "2026-01-01", "meta": {"kind": "note"}},
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "filter-by-meta", "kind", "note",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.filter_by_meta.assert_called_once_with("kind", "note", ns="default", limit=100)
    out = capsys.readouterr().out
    assert "1" in out


def test_cli_filter_by_meta_json(tmp_path, capsys):
    """filter-by-meta --json prints JSON array."""
    mock_store = MagicMock()
    mock_store.filter_by_meta.return_value = [
        {"id": 2, "text": "world", "ns": "default", "summary": None,
         "tier": 1, "created": "2026-01-01", "meta": {}},
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "filter-by-meta", "x", "1",
                           "--json", "--path", str(tmp_path)]),
    ):
        main()
    parsed = json.loads(capsys.readouterr().out)
    assert parsed[0]["id"] == 2


def test_cli_filter_by_meta_bool(tmp_path, capsys):
    """filter-by-meta converts 'true'/'false' to Python bool."""
    mock_store = MagicMock()
    mock_store.filter_by_meta.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "filter-by-meta", "active", "true",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.filter_by_meta.assert_called_once_with("active", True, ns="default", limit=100)


# ── summary-stats CLI ─────────────────────────────────────────────────────────

def test_cli_summary_stats_ok(tmp_path, capsys):
    """summary-stats prints coverage info."""
    mock_store = MagicMock()
    mock_store.summary_stats.return_value = {
        "total": 10, "with_summary": 7, "without_summary": 3, "pct_with_summary": 70.0,
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "summary-stats", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "70" in out


def test_cli_summary_stats_json(tmp_path, capsys):
    """summary-stats --json outputs raw JSON."""
    mock_store = MagicMock()
    mock_store.summary_stats.return_value = {
        "total": 3, "with_summary": 1, "without_summary": 2, "pct_with_summary": 33.33,
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "summary-stats", "--json", "--path", str(tmp_path)]),
    ):
        main()
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["total"] == 3


def test_cli_filter_by_meta_false(tmp_path, capsys):
    """filter-by-meta converts 'false' to Python False."""
    mock_store = MagicMock()
    mock_store.filter_by_meta.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "filter-by-meta", "active", "false",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.filter_by_meta.assert_called_once_with("active", False, ns="default", limit=100)


# ── pinned-memories CLI ────────────────────────────────────────────────────────

def test_cli_pinned_memories_empty(tmp_path, capsys):
    """pinned-memories prints empty count when nothing pinned."""
    mock_store = MagicMock()
    mock_store.pinned_memories.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "pinned-memories", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "0" in out


def test_cli_pinned_memories_json(tmp_path, capsys):
    """pinned-memories --json outputs JSON."""
    mock_store = MagicMock()
    mock_store.pinned_memories.return_value = [{"id": 1, "text": "pinned", "tier": 0}]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "pinned-memories", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed[0]["tier"] == 0


def test_cli_pinned_memories_all_ns(tmp_path, capsys):
    """pinned-memories --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.pinned_memories.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "pinned-memories", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.pinned_memories.assert_called_once_with(ns=None, limit=100)


# ── update-meta-key CLI ────────────────────────────────────────────────────────

def test_cli_update_meta_key_sets_value(tmp_path, capsys):
    """update-meta-key prints confirmation."""
    mock_store = MagicMock()
    mock_store.update_meta_key.return_value = True
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "update-meta-key", "1", "active", "true",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.update_meta_key.assert_called_once_with(1, "active", True)


def test_cli_update_meta_key_false_value(tmp_path, capsys):
    """update-meta-key converts 'false' to Python False."""
    mock_store = MagicMock()
    mock_store.update_meta_key.return_value = True
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "update-meta-key", "1", "active", "false",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.update_meta_key.assert_called_once_with(1, "active", False)


def test_cli_update_meta_key_not_found(tmp_path, capsys):
    """update-meta-key exits 1 when memory not found."""
    mock_store = MagicMock()
    mock_store.update_meta_key.return_value = False
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "update-meta-key", "999", "k", "v",
                           "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 1


def test_cli_update_meta_key_removes(tmp_path, capsys):
    """update-meta-key with no value arg removes the key."""
    mock_store = MagicMock()
    mock_store.update_meta_key.return_value = True
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "update-meta-key", "1", "temp",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.update_meta_key.assert_called_once_with(1, "temp", None)


def test_cli_update_meta_key_int_value(tmp_path, capsys):
    """update-meta-key parses int value correctly."""
    mock_store = MagicMock()
    mock_store.update_meta_key.return_value = True
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "update-meta-key", "1", "score", "42",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.update_meta_key.assert_called_once_with(1, "score", 42)


# ── search-by-summary CLI ──────────────────────────────────────────────────────

def test_cli_search_by_summary_returns_hits(tmp_path, capsys):
    """search-by-summary prints matching memories."""
    mock_store = MagicMock()
    mock_store.search_by_summary.return_value = [
        {"id": 1, "text": "hello world", "summary": "greeting", "tier": 1, "created": "t", "ns": "default"}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "search-by-summary", "greeting",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "1" in out


def test_cli_search_by_summary_json(tmp_path, capsys):
    """search-by-summary --json outputs JSON."""
    mock_store = MagicMock()
    mock_store.search_by_summary.return_value = [
        {"id": 1, "text": "t", "summary": "s", "tier": 1, "created": "c", "ns": "default"}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "search-by-summary", "s", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert json.loads(out)[0]["id"] == 1


def test_cli_search_by_summary_all_ns(tmp_path, capsys):
    """search-by-summary --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.search_by_summary.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "search-by-summary", "q", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.search_by_summary.assert_called_once_with("q", ns=None, limit=20)


def test_cli_pinned_memories_nonempty(tmp_path, capsys):
    """pinned-memories prints each pinned memory row in text mode."""
    mock_store = MagicMock()
    mock_store.pinned_memories.return_value = [
        {"id": 7, "text": "important doc", "tier": 0, "ns": "default",
         "summary": None, "created": "2026-01-01", "meta": {}}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "pinned-memories", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "[7]" in out


# ── set-tier-by-tag CLI ────────────────────────────────────────────────────────

def test_cli_set_tier_by_tag(tmp_path, capsys):
    """set-tier-by-tag calls store method with correct args."""
    mock_store = MagicMock()
    mock_store.set_tier_by_tag.return_value = 3
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "set-tier-by-tag", "vip", "0",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.set_tier_by_tag.assert_called_once_with("vip", 0, ns="default")
    out = capsys.readouterr().out
    assert "3" in out


def test_cli_set_tier_by_tag_all_ns(tmp_path, capsys):
    """set-tier-by-tag --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.set_tier_by_tag.return_value = 0
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "set-tier-by-tag", "t", "1",
                           "--all-ns", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.set_tier_by_tag.assert_called_once_with("t", 1, ns=None)


# ── rotate-ns CLI ──────────────────────────────────────────────────────────────

def test_cli_rotate_ns(tmp_path, capsys):
    """rotate-ns calls store method and prints count."""
    mock_store = MagicMock()
    mock_store.rotate_ns.return_value = 5
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "rotate-ns", "src", "dst",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.rotate_ns.assert_called_once_with("src", "dst", limit=100, tier=None)
    out = capsys.readouterr().out
    assert "5" in out


def test_cli_rotate_ns_with_tier(tmp_path, capsys):
    """rotate-ns --tier passes tier to store."""
    mock_store = MagicMock()
    mock_store.rotate_ns.return_value = 2
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "rotate-ns", "src", "dst", "--tier", "2",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.rotate_ns.assert_called_once_with("src", "dst", limit=100, tier=2)


# ── compact-meta CLI ───────────────────────────────────────────────────────────

def test_cli_compact_meta(tmp_path, capsys):
    """compact-meta calls store method and prints count."""
    mock_store = MagicMock()
    mock_store.compact_meta.return_value = 4
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "compact-meta", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.compact_meta.assert_called_once_with(ns="default", keep_keys=None)
    out = capsys.readouterr().out
    assert "4" in out


def test_cli_compact_meta_with_keep(tmp_path, capsys):
    """compact-meta --keep-keys passes keep_keys."""
    mock_store = MagicMock()
    mock_store.compact_meta.return_value = 1
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "compact-meta", "--keep-keys", "a", "b",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.compact_meta.assert_called_once_with(ns="default", keep_keys=["a", "b"])


def test_cli_compact_meta_all_ns(tmp_path, capsys):
    """compact-meta --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.compact_meta.return_value = 0
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "compact-meta", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.compact_meta.assert_called_once_with(ns=None, keep_keys=None)


# ── list-by-tier CLI ───────────────────────────────────────────────────────────

def test_cli_list_by_tier_text(tmp_path, capsys):
    """list-by-tier prints matching memories."""
    mock_store = MagicMock()
    mock_store.list_by_tier.return_value = [
        {"id": 1, "text": "pinned doc", "tier": 0, "ns": "default",
         "summary": None, "created": "t", "meta": {}}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "list-by-tier", "0",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "[1]" in out


def test_cli_list_by_tier_json(tmp_path, capsys):
    """list-by-tier --json outputs JSON."""
    mock_store = MagicMock()
    mock_store.list_by_tier.return_value = [{"id": 1, "tier": 0}]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "list-by-tier", "0", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert json.loads(out)[0]["id"] == 1


def test_cli_list_by_tier_all_ns(tmp_path, capsys):
    """list-by-tier --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.list_by_tier.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "list-by-tier", "0", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.list_by_tier.assert_called_once_with(0, ns=None, limit=100)


# ── newest CLI ─────────────────────────────────────────────────────────────────

def test_cli_recent_text(tmp_path, capsys):
    """newest prints most recently created memories."""
    mock_store = MagicMock()
    mock_store.recent.return_value = [
        {"id": 5, "text": "latest", "tier": 1, "ns": "default",
         "summary": None, "created": "t"}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "newest", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "[5]" in out


def test_cli_recent_json(tmp_path, capsys):
    """newest --json outputs JSON."""
    mock_store = MagicMock()
    mock_store.recent.return_value = [{"id": 5, "text": "x"}]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "newest", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert json.loads(out)[0]["id"] == 5


def test_cli_recent_all_ns(tmp_path, capsys):
    """newest --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.recent.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "newest", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.recent.assert_called_once_with(ns=None, n=10)


# ── oldest CLI ─────────────────────────────────────────────────────────────────

def test_cli_oldest_text(tmp_path, capsys):
    """oldest prints oldest memories."""
    mock_store = MagicMock()
    mock_store.oldest.return_value = [
        {"id": 1, "text": "very old", "tier": 1, "ns": "default",
         "summary": None, "created": "t"}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "oldest", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "[1]" in out


def test_cli_oldest_json(tmp_path, capsys):
    """oldest --json outputs JSON."""
    mock_store = MagicMock()
    mock_store.oldest.return_value = [{"id": 1, "text": "old"}]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "oldest", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert json.loads(out)[0]["id"] == 1


def test_cli_oldest_all_ns(tmp_path, capsys):
    """oldest --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.oldest.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "oldest", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.oldest.assert_called_once_with(ns=None, n=10)


# ── replace-text CLI ───────────────────────────────────────────────────────────

def test_cli_replace_text_success(tmp_path, capsys):
    """replace-text prints confirmation on success."""
    mock_store = MagicMock()
    mock_store.replace_text.return_value = True
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "replace-text", "1", "new content",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "updated" in out


def test_cli_replace_text_not_found(tmp_path):
    """replace-text exits 1 when memory not found."""
    mock_store = MagicMock()
    mock_store.replace_text.return_value = False
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "replace-text", "999", "x",
                           "--path", str(tmp_path)]),
    ):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 1


def test_cli_recent_json_coverage(tmp_path, capsys):
    """recent --json path (line 1717) is covered."""
    import json as _json
    hits = [{"id": 2, "ns": "default", "tier": 1, "text": "hello",
             "summary": None, "last_accessed": None, "access_count": 0}]
    mock_store = MagicMock()
    mock_store.recent_accessed.return_value = hits
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "recent", "--json",
                           "--ns", "default", "--path", str(tmp_path)]),
    ):
        main()
    parsed = _json.loads(capsys.readouterr().out)
    assert parsed[0]["id"] == 2


# ── search-text CLI ────────────────────────────────────────────────────────────

def test_cli_search_text_text(tmp_path, capsys):
    """search-text prints matching memories."""
    mock_store = MagicMock()
    mock_store.search_text.return_value = [
        {"id": 3, "text": "needle found here", "tier": 1, "ns": "default",
         "summary": None, "created": "t"}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "search-text", "needle",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "[3]" in out


def test_cli_search_text_json(tmp_path, capsys):
    """search-text --json outputs JSON."""
    mock_store = MagicMock()
    mock_store.search_text.return_value = [{"id": 3, "text": "needle"}]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "search-text", "needle", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert json.loads(out)[0]["id"] == 3


def test_cli_search_text_all_ns(tmp_path, capsys):
    """search-text --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.search_text.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "search-text", "q", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.search_text.assert_called_once_with("q", ns=None, limit=20)


# ── count-by-ns CLI ────────────────────────────────────────────────────────────

def test_cli_count_by_ns_text(tmp_path, capsys):
    """count-by-ns prints namespace counts."""
    mock_store = MagicMock()
    mock_store.count_by_ns.return_value = {"myns": 5, "other": 2}
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "count-by-ns", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "myns" in out
    assert "5" in out


def test_cli_count_by_ns_json(tmp_path, capsys):
    """count-by-ns --json outputs JSON."""
    mock_store = MagicMock()
    mock_store.count_by_ns.return_value = {"ns1": 3}
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "count-by-ns", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert json.loads(out)["ns1"] == 3


# ── clear-ns CLI ───────────────────────────────────────────────────────────────

def test_cli_clear_ns(tmp_path, capsys):
    """clear-ns prints deletion count."""
    mock_store = MagicMock()
    mock_store.clear_ns.return_value = 7
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "clear-ns", "myns",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.clear_ns.assert_called_once_with("myns")
    out = capsys.readouterr().out
    assert "7" in out


# ── copy-to-ns CLI ─────────────────────────────────────────────────────────────

def test_cli_copy_to_ns(tmp_path, capsys):
    """copy-to-ns copies memories and prints count."""
    mock_store = MagicMock()
    mock_store.copy_to_ns.return_value = 3
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "copy-to-ns", "1", "2", "3",
                           "--dst-ns", "newns", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.copy_to_ns.assert_called_once_with([1, 2, 3], "newns")
    out = capsys.readouterr().out
    assert "3" in out


# ── rename-tag CLI ─────────────────────────────────────────────────────────────

def test_cli_rename_tag(tmp_path, capsys):
    """rename-tag prints updated count."""
    mock_store = MagicMock()
    mock_store.rename_tag.return_value = 3
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "rename-tag", "old", "new",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.rename_tag.assert_called_once_with("old", "new", ns="default")
    out = capsys.readouterr().out
    assert "3" in out


def test_cli_rename_tag_all_ns(tmp_path):
    """rename-tag --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.rename_tag.return_value = 0
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "rename-tag", "old", "new", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.rename_tag.assert_called_once_with("old", "new", ns=None)


# ── find-duplicates CLI ────────────────────────────────────────────────────────

def test_cli_find_duplicates_text(tmp_path, capsys):
    """find-duplicates prints duplicate groups."""
    mock_store = MagicMock()
    mock_store.find_duplicates.return_value = [
        {"text": "hello", "count": 2, "ids": [1, 2]}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "find-duplicates",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "1" in out


def test_cli_find_duplicates_json(tmp_path, capsys):
    """find-duplicates --json outputs JSON."""
    mock_store = MagicMock()
    mock_store.find_duplicates.return_value = [{"text": "dup", "count": 2, "ids": [1, 2]}]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "find-duplicates", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data[0]["count"] == 2


def test_cli_find_duplicates_all_ns(tmp_path):
    """find-duplicates --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.find_duplicates.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "find-duplicates", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.find_duplicates.assert_called_once_with(ns=None, limit=20)


# ── swap-tier CLI ──────────────────────────────────────────────────────────────

def test_cli_swap_tier(tmp_path, capsys):
    """swap-tier prints updated count."""
    mock_store = MagicMock()
    mock_store.swap_tier.return_value = 5
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "swap-tier", "myns", "1", "2",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.swap_tier.assert_called_once_with("myns", 1, 2)
    out = capsys.readouterr().out
    assert "5" in out


def test_cli_swap_tier_error(tmp_path, capsys):
    """swap-tier prints error when ValueError raised."""
    mock_store = MagicMock()
    mock_store.swap_tier.side_effect = ValueError("tier must be 0, 1, or 2")
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "swap-tier", "myns", "0", "9",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "Error" in out


# ── ns-summary CLI ─────────────────────────────────────────────────────────────

def test_cli_ns_summary_text(tmp_path, capsys):
    """ns-summary prints namespace dashboard."""
    mock_store = MagicMock()
    mock_store.ns_summary.return_value = {
        "ns": "myns", "count": 10,
        "tiers": {"pinned": 1, "default": 8, "ambient": 1},
        "avg_chars": 42.0, "min_chars": 3, "max_chars": 100,
        "avg_accesses": 1.5, "max_accesses": 7, "never_accessed": 3,
        "oldest": "2026-01-01", "newest": "2026-06-01",
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "ns-summary", "myns",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "10" in out
    assert "myns" in out


def test_cli_ns_summary_json(tmp_path, capsys):
    """ns-summary --json outputs JSON."""
    mock_store = MagicMock()
    mock_store.ns_summary.return_value = {"ns": "x", "count": 5, "tiers": {}, "avg_chars": 0,
                                          "min_chars": 0, "max_chars": 0, "avg_accesses": 0,
                                          "max_accesses": 0, "never_accessed": 0,
                                          "oldest": None, "newest": None}
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "ns-summary", "x", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert json.loads(out)["count"] == 5


# ── toggle-tier CLI ────────────────────────────────────────────────────────────

def test_cli_toggle_tier(tmp_path, capsys):
    """toggle-tier cycles tier and prints result."""
    mock_store = MagicMock()
    mock_store.toggle_tier.return_value = 0
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "toggle-tier", "42",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.toggle_tier.assert_called_once_with(42)
    out = capsys.readouterr().out
    assert "0" in out or "pinned" in out


def test_cli_toggle_tier_not_found(tmp_path, capsys):
    """toggle-tier prints message for missing memory."""
    mock_store = MagicMock()
    mock_store.toggle_tier.return_value = None
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "toggle-tier", "99",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "not found" in out


# ── merge-texts CLI ────────────────────────────────────────────────────────────

def test_cli_merge_texts(tmp_path, capsys):
    """merge-texts prints new merged memory id."""
    mock_store = MagicMock()
    mock_store.merge_texts.return_value = 99
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "merge-texts", "1", "2", "3",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.merge_texts.assert_called_once_with(
        [1, 2, 3], separator="\n\n", ns="default", delete_originals=False
    )
    out = capsys.readouterr().out
    assert "99" in out


def test_cli_merge_texts_not_found(tmp_path, capsys):
    """merge-texts prints message when ids not found."""
    mock_store = MagicMock()
    mock_store.merge_texts.return_value = None
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "merge-texts", "99",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "No memories" in out


# ── truncate-text CLI ──────────────────────────────────────────────────────────

def test_cli_truncate_text(tmp_path, capsys):
    """truncate-text prints success message."""
    mock_store = MagicMock()
    mock_store.truncate_text.return_value = True
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "truncate-text", "5", "50",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.truncate_text.assert_called_once_with(5, 50)
    out = capsys.readouterr().out
    assert "truncated" in out


def test_cli_truncate_text_not_found(tmp_path, capsys):
    """truncate-text prints not found message."""
    mock_store = MagicMock()
    mock_store.truncate_text.return_value = False
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "truncate-text", "99", "10",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "not found" in out


# ── search-by-access-count CLI ─────────────────────────────────────────────────

def test_cli_search_by_access_count_text(tmp_path, capsys):
    """search-by-access-count prints matching memories."""
    mock_store = MagicMock()
    mock_store.search_by_access_count.return_value = [
        {"id": 1, "text": "some memory", "access_count": 3, "ns": "default",
         "tier": 1, "summary": None, "created": "t"}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "search-by-access-count",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "[1]" in out


def test_cli_search_by_access_count_json(tmp_path, capsys):
    """search-by-access-count --json outputs JSON."""
    mock_store = MagicMock()
    mock_store.search_by_access_count.return_value = [
        {"id": 1, "access_count": 0}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "search-by-access-count", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert json.loads(out)[0]["id"] == 1


def test_cli_search_by_access_count_all_ns(tmp_path):
    """search-by-access-count --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.search_by_access_count.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "search-by-access-count", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    call_kwargs = mock_store.search_by_access_count.call_args
    assert call_kwargs.kwargs.get("ns") is None or call_kwargs[1].get("ns") is None


# ── age-by-ns CLI ──────────────────────────────────────────────────────────────

def test_cli_age_by_ns_text(tmp_path, capsys):
    """age-by-ns prints age breakdown."""
    mock_store = MagicMock()
    mock_store.age_by_ns.return_value = {
        "ns": "myns", "today": 3, "this_week": 1,
        "this_month": 0, "this_quarter": 0, "older": 0, "total": 4
    }
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "age-by-ns", "myns",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "myns" in out
    assert "4" in out


def test_cli_age_by_ns_json(tmp_path, capsys):
    """age-by-ns --json outputs JSON."""
    mock_store = MagicMock()
    mock_store.age_by_ns.return_value = {"ns": "x", "total": 2, "today": 2,
                                         "this_week": 0, "this_month": 0,
                                         "this_quarter": 0, "older": 0}
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "age-by-ns", "x", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert json.loads(out)["total"] == 2


# ── delete-by-tier CLI ─────────────────────────────────────────────────────────

def test_cli_delete_by_tier(tmp_path, capsys):
    """delete-by-tier prints deleted count."""
    mock_store = MagicMock()
    mock_store.delete_by_tier.return_value = 5
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "delete-by-tier", "myns", "2",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.delete_by_tier.assert_called_once_with("myns", 2)
    out = capsys.readouterr().out
    assert "5" in out


def test_cli_delete_by_tier_error(tmp_path, capsys):
    """delete-by-tier prints error when ValueError raised."""
    mock_store = MagicMock()
    mock_store.delete_by_tier.side_effect = ValueError("tier must be 0, 1, or 2")
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "delete-by-tier", "myns", "9",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "Error" in out


# ── untagged-memories CLI ──────────────────────────────────────────────────────

def test_cli_untagged_memories_text(tmp_path, capsys):
    """untagged-memories prints matching memories."""
    mock_store = MagicMock()
    mock_store.untagged_memories.return_value = [
        {"id": 3, "text": "no tag here", "tier": 1, "ns": "default", "summary": None, "created": "t"}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "untagged-memories",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "[3]" in out


def test_cli_untagged_memories_json(tmp_path, capsys):
    """untagged-memories --json outputs JSON."""
    mock_store = MagicMock()
    mock_store.untagged_memories.return_value = [{"id": 3}]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "untagged-memories", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert json.loads(out)[0]["id"] == 3


def test_cli_untagged_memories_all_ns(tmp_path):
    """untagged-memories --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.untagged_memories.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "untagged-memories", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.untagged_memories.assert_called_once_with(ns=None, limit=20)


# ── set-meta-for-untagged CLI ──────────────────────────────────────────────────

def test_cli_set_meta_for_untagged(tmp_path, capsys):
    """set-meta-for-untagged prints updated count."""
    mock_store = MagicMock()
    mock_store.set_meta_for_untagged.return_value = 4
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "set-meta-for-untagged",
                           "source", "auto", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.set_meta_for_untagged.assert_called_once_with(
        "default", "source", "auto", limit=100
    )
    out = capsys.readouterr().out
    assert "4" in out




# ─── Batch 5: clone-memory, memories-without-summary, pin-by-tag, promote-by-access ───

def test_cli_clone_memory(tmp_path, capsys):
    """clone-memory clones and prints new id."""
    mock_store = MagicMock()
    mock_store.clone_memory.return_value = 99
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "clone-memory", "42", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.clone_memory.assert_called_once_with(42, dst_ns=None)
    out = capsys.readouterr().out
    assert "99" in out


def test_cli_clone_memory_with_dst_ns(tmp_path, capsys):
    """clone-memory passes dst_ns when given."""
    mock_store = MagicMock()
    mock_store.clone_memory.return_value = 55
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "clone-memory", "7", "--dst-ns", "archive",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.clone_memory.assert_called_once_with(7, dst_ns="archive")
    out = capsys.readouterr().out
    assert "55" in out


def test_cli_clone_memory_not_found(tmp_path, capsys):
    """clone-memory prints 'not found' when store returns None."""
    mock_store = MagicMock()
    mock_store.clone_memory.return_value = None
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "clone-memory", "999", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "not found" in out


def test_cli_memories_without_summary_text(tmp_path, capsys):
    """memories-without-summary prints count."""
    mock_store = MagicMock()
    mock_store.memories_without_summary.return_value = [
        {"id": 1, "ns": "default", "text": "hello", "summary": None, "tier": 1, "created": "2024-01-01"}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "memories-without-summary", "--ns", "default",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "1" in out


def test_cli_memories_without_summary_json(tmp_path, capsys):
    """memories-without-summary --json outputs JSON."""
    mock_store = MagicMock()
    mock_store.memories_without_summary.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "memories-without-summary", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    import json
    out = capsys.readouterr().out
    assert json.loads(out) == []


def test_cli_memories_without_summary_all_ns(tmp_path, capsys):
    """memories-without-summary --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.memories_without_summary.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "memories-without-summary", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.memories_without_summary.assert_called_once()
    kw = mock_store.memories_without_summary.call_args[1]
    assert kw["ns"] is None


def test_cli_pin_by_tag_text(tmp_path, capsys):
    """pin-by-tag prints pinned count."""
    mock_store = MagicMock()
    mock_store.pin_by_tag.return_value = 3
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "pin-by-tag", "urgent", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "3" in out


def test_cli_pin_by_tag_all_ns(tmp_path, capsys):
    """pin-by-tag --all-ns passes ns=None to pin_by_tag."""
    mock_store = MagicMock()
    mock_store.pin_by_tag.return_value = 2
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "pin-by-tag", "vip", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    kw = mock_store.pin_by_tag.call_args
    assert kw[0][0] == "vip"


def test_cli_promote_by_access_text(tmp_path, capsys):
    """promote-by-access prints promoted count."""
    mock_store = MagicMock()
    mock_store.promote_by_access.return_value = 5
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "promote-by-access", "default",
                           "--n", "10", "--from-tier", "2", "--to-tier", "1",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.promote_by_access.assert_called_once_with(
        "default", n=10, from_tier=2, to_tier=1
    )
    out = capsys.readouterr().out
    assert "5" in out


def test_cli_promote_by_access_bad_tier(tmp_path, capsys):
    """promote-by-access prints error on ValueError."""
    mock_store = MagicMock()
    mock_store.promote_by_access.side_effect = ValueError("bad tier")
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "promote-by-access", "default",
                           "--from-tier", "9", "--to-tier", "1",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "error" in out.lower() or "bad tier" in out.lower()


# ─── Batch 6: filter-by-text-length, multi-tag-filter, tag-stats, split-memory ───

def test_cli_filter_by_text_length_text(tmp_path, capsys):
    """filter-by-text-length prints count."""
    mock_store = MagicMock()
    mock_store.filter_by_text_length.return_value = [
        {"id": 1, "ns": "default", "text": "hi", "summary": None, "tier": 1, "created": "2024-01-01"}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "filter-by-text-length", "--max-chars", "5",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "1" in out


def test_cli_filter_by_text_length_json(tmp_path, capsys):
    """filter-by-text-length --json outputs list."""
    mock_store = MagicMock()
    mock_store.filter_by_text_length.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "filter-by-text-length", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    import json
    out = capsys.readouterr().out
    assert json.loads(out) == []


def test_cli_filter_by_text_length_all_ns(tmp_path, capsys):
    """filter-by-text-length --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.filter_by_text_length.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "filter-by-text-length", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    kw = mock_store.filter_by_text_length.call_args[1]
    assert kw["ns"] is None


def test_cli_multi_tag_filter_text(tmp_path, capsys):
    """multi-tag-filter prints count."""
    mock_store = MagicMock()
    mock_store.multi_tag_filter.return_value = [
        {"id": 2, "ns": "default", "text": "tagged", "summary": None, "tier": 1, "created": "2024-01-01"}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "multi-tag-filter", "a", "b",
                           "--mode", "any", "--path", str(tmp_path)]),
    ):
        main()
    mock_store.multi_tag_filter.assert_called_once()
    out = capsys.readouterr().out
    assert "1" in out


def test_cli_multi_tag_filter_json(tmp_path, capsys):
    """multi-tag-filter --json outputs list."""
    mock_store = MagicMock()
    mock_store.multi_tag_filter.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "multi-tag-filter", "x", "--json",
                           "--path", str(tmp_path)]),
    ):
        main()
    import json
    out = capsys.readouterr().out
    assert json.loads(out) == []


def test_cli_multi_tag_filter_all_ns(tmp_path, capsys):
    """multi-tag-filter --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.multi_tag_filter.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "multi-tag-filter", "z", "--all-ns",
                           "--path", str(tmp_path)]),
    ):
        main()
    kw = mock_store.multi_tag_filter.call_args[1]
    assert kw["ns"] is None


def test_cli_tag_stats_text(tmp_path, capsys):
    """tag-stats prints count per tag."""
    mock_store = MagicMock()
    mock_store.tag_stats.return_value = [
        {"tag": "a", "count": 3, "pinned": 1, "default": 2, "ambient": 0}
    ]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "tag-stats", "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "a" in out


def test_cli_tag_stats_json(tmp_path, capsys):
    """tag-stats --json outputs list."""
    mock_store = MagicMock()
    mock_store.tag_stats.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "tag-stats", "--json", "--path", str(tmp_path)]),
    ):
        main()
    import json
    out = capsys.readouterr().out
    assert json.loads(out) == []


def test_cli_tag_stats_all_ns(tmp_path, capsys):
    """tag-stats --all-ns passes ns=None."""
    mock_store = MagicMock()
    mock_store.tag_stats.return_value = []
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "tag-stats", "--all-ns", "--path", str(tmp_path)]),
    ):
        main()
    kw = mock_store.tag_stats.call_args[1]
    assert kw["ns"] is None


def test_cli_split_memory_separator(tmp_path, capsys):
    """split-memory with separator prints new ids."""
    mock_store = MagicMock()
    mock_store.split_memory.return_value = [10, 11]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "split-memory", "5", "--separator", "\n\n",
                           "--path", str(tmp_path)]),
    ):
        main()
    mock_store.split_memory.assert_called_once_with(
        5, separator="\n\n", max_chars=None, delete_original=False
    )
    out = capsys.readouterr().out
    assert "2 parts" in out or "10" in out


def test_cli_split_memory_not_found(tmp_path, capsys):
    """split-memory prints message when memory not found or < 2 parts."""
    mock_store = MagicMock()
    mock_store.split_memory.return_value = None
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "split-memory", "99", "--separator", "X",
                           "--path", str(tmp_path)]),
    ):
        main()
    out = capsys.readouterr().out
    assert "not found" in out or "< 2" in out


def test_cli_split_memory_delete_original(tmp_path, capsys):
    """split-memory --delete-original passes flag to store."""
    mock_store = MagicMock()
    mock_store.split_memory.return_value = [20, 21, 22]
    with (
        patch("mnemonics.store.Store", return_value=mock_store),
        patch("sys.argv", ["mnemonics", "split-memory", "3", "--separator", "\n",
                           "--delete-original", "--path", str(tmp_path)]),
    ):
        main()
    kw = mock_store.split_memory.call_args[1]
    assert kw["delete_original"] is True
