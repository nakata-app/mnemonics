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
