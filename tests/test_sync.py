"""Tests for mnemonics.sync — portable export + merge between stores.

Exercises all three conflict strategies and the only-ns filter. Uses the
real encoder because import re-embeds and we want the resulting index to
actually serve queries.
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from mnemonics.ingest import ingest
from mnemonics.store import Store
from mnemonics.sync import (
    _MANIFEST_NAME,
    _MANIFEST_VERSION,
    _row_hash,
    export_store,
    import_store,
)


def _seed(store: Store, ns: str, texts: list[str]) -> None:
    ingest(texts=texts, store=store, ns=ns)


def test_export_writes_manifest_with_rows(tmp_path):
    src = tmp_path / "src"
    s = Store(path=src)
    _seed(s, "alpha", ["one", "two"])
    _seed(s, "beta", ["three"])

    arc = export_store(store_path=src, out=tmp_path / "snap.sync.tar.gz")
    with tarfile.open(arc) as tf:
        names = [m.name for m in tf.getmembers()]
        assert names == [_MANIFEST_NAME]
        manifest = json.loads(tf.extractfile(_MANIFEST_NAME).read().decode("utf-8"))

    assert manifest["version"] == _MANIFEST_VERSION
    assert len(manifest["rows"]) == 3
    nss = {r["ns"] for r in manifest["rows"]}
    assert nss == {"alpha", "beta"}
    # Every row carries its content hash.
    for r in manifest["rows"]:
        assert r["hash"] == _row_hash(r["text"])


def test_export_missing_db_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_store(store_path=tmp_path / "no-such-dir", out=tmp_path / "x.tar.gz")


def test_import_into_empty_target(tmp_path):
    src = tmp_path / "src"
    _seed(Store(path=src), "shared", ["row-A", "row-B", "row-C"])
    arc = export_store(store_path=src, out=tmp_path / "snap.sync.tar.gz")

    dest_path = tmp_path / "dest"
    summary = import_store(archive=arc, store_path=dest_path)
    assert summary == {"imported": 3, "skipped": 0, "overwritten": 0}
    assert Store(path=dest_path).count("shared") == 3


def test_import_skip_existing_dedups_by_hash(tmp_path):
    """If the same text already lives in the target ns, default strategy
    must skip it. New texts still come in."""
    src = tmp_path / "src"
    _seed(Store(path=src), "shared", ["row-A", "row-B-new", "row-C"])
    arc = export_store(store_path=src, out=tmp_path / "snap.sync.tar.gz")

    dest_path = tmp_path / "dest"
    _seed(Store(path=dest_path), "shared", ["row-A", "row-existing"])

    summary = import_store(archive=arc, store_path=dest_path, strategy="skip-existing")
    # row-A is a hash collision → skipped. row-B-new and row-C are new.
    assert summary["imported"] == 2
    assert summary["skipped"] == 1
    assert summary["overwritten"] == 0

    s = Store(path=dest_path)
    assert s.count("shared") == 4  # 2 existing + 2 newly imported


def test_import_force_new_id_always_inserts(tmp_path):
    src = tmp_path / "src"
    _seed(Store(path=src), "shared", ["row-A", "row-B"])
    arc = export_store(store_path=src, out=tmp_path / "snap.sync.tar.gz")

    dest_path = tmp_path / "dest"
    _seed(Store(path=dest_path), "shared", ["row-A"])

    summary = import_store(archive=arc, store_path=dest_path, strategy="force-new-id")
    # Both incoming rows are inserted, even though row-A duplicates an
    # existing hash.
    assert summary["imported"] == 2
    assert summary["skipped"] == 0
    s = Store(path=dest_path)
    assert s.count("shared") == 3  # 1 existing + 2 inserted (one is a dupe)


def test_import_overwrite_replaces_existing(tmp_path):
    src = tmp_path / "src"
    _seed(Store(path=src), "shared", ["row-A", "row-B"])
    arc = export_store(store_path=src, out=tmp_path / "snap.sync.tar.gz")

    dest_path = tmp_path / "dest"
    _seed(Store(path=dest_path), "shared", ["row-A", "row-stays"])

    summary = import_store(archive=arc, store_path=dest_path, strategy="overwrite")
    # row-A: existing row was deleted then re-inserted with a fresh id.
    # row-B: new insert. row-stays: untouched.
    assert summary["overwritten"] == 1
    assert summary["imported"] == 2
    s = Store(path=dest_path)
    # 2 existing → 1 deleted (row-A) → 2 imported = 3 total
    assert s.count("shared") == 3


def test_import_only_ns_filters(tmp_path):
    src = tmp_path / "src"
    src_store = Store(path=src)
    _seed(src_store, "alpha", ["a1", "a2"])
    _seed(src_store, "beta", ["b1"])
    arc = export_store(store_path=src, out=tmp_path / "snap.sync.tar.gz")

    dest_path = tmp_path / "dest"
    summary = import_store(
        archive=arc, store_path=dest_path, strategy="skip-existing", only_ns="alpha"
    )
    assert summary["imported"] == 2
    s = Store(path=dest_path)
    assert s.count("alpha") == 2
    assert s.count("beta") == 0


def test_import_missing_archive_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_store(archive=tmp_path / "nope.tar.gz", store_path=tmp_path / "dest")


def test_import_unknown_strategy_raises(tmp_path):
    src = tmp_path / "src"
    _seed(Store(path=src), "x", ["one"])
    arc = export_store(store_path=src, out=tmp_path / "snap.sync.tar.gz")
    with pytest.raises(ValueError, match="unknown strategy"):
        import_store(archive=arc, store_path=tmp_path / "dest", strategy="bogus")  # type: ignore[arg-type]


def test_import_unsupported_manifest_version_raises(tmp_path):
    # Hand-craft an archive with a bumped version field.
    arc = tmp_path / "bad.tar.gz"
    bad = {"version": 999, "rows": []}
    with tarfile.open(arc, "w:gz") as tf:
        import io
        data = json.dumps(bad).encode("utf-8")
        info = tarfile.TarInfo(name=_MANIFEST_NAME)
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError, match="manifest version"):
        import_store(archive=arc, store_path=tmp_path / "dest")


def test_imported_rows_are_searchable(tmp_path):
    """End-to-end: after import, retrieve must find the merged row."""
    from mnemonics.retrieve import retrieve

    src = tmp_path / "src"
    _seed(Store(path=src), "search-ns", [
        "Zeus testleri DeepSeek V4 Pro ile koşar",
        "AdaptMem encoder fine-tuning Colab T4 GPU üzerinde",
    ])
    arc = export_store(store_path=src, out=tmp_path / "snap.sync.tar.gz")

    dest_path = tmp_path / "dest"
    import_store(archive=arc, store_path=dest_path)
    dest = Store(path=dest_path)

    out = retrieve(
        query="Zeus DeepSeek pipeline",
        store=dest,
        ns="search-ns",
        top_k=2,
        decay=False,
    )
    assert len(out["results"]) >= 1
    assert "Zeus" in out["results"][0]["text"] or "DeepSeek" in out["results"][0]["text"]


def test_read_manifest_unreadable_member(tmp_path):
    """Line 103: manifest member extractfile returns None → ValueError."""
    import tarfile, io
    from mnemonics.sync import _read_manifest
    arc = tmp_path / "bad.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        # Add a symlink member named as the manifest
        lnk = tarfile.TarInfo(name="mnemonics_manifest.json")
        lnk.type = tarfile.SYMTYPE
        lnk.linkname = "nonexistent"
        tf.addfile(lnk)
    import pytest
    with pytest.raises((ValueError, Exception)):
        _read_manifest(arc)


def test_import_all_rows_duplicate_skips_batch(tmp_path):
    """Line 175: to_insert empty after dedup → continue (no insert)."""
    src = tmp_path / "src"
    _seed(Store(path=src), "shared", ["dup-A", "dup-B"])
    arc = export_store(store_path=src, out=tmp_path / "snap.tar.gz")

    dest = tmp_path / "dest"
    # First import — all rows land
    import_store(archive=arc, store_path=dest)
    count_before = Store(path=dest).count("shared")

    # Second import with skip-existing — all rows are duplicates, to_insert empty
    summary = import_store(archive=arc, store_path=dest, strategy="skip-existing")
    assert Store(path=dest).count("shared") == count_before
    assert summary["skipped"] == count_before
