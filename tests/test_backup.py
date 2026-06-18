"""Tests for mnemonics.backup — bundle and restore canonical store files."""
from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from mnemonics.backup import backup, restore, _is_allowed


def _make_store(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "memories.db").write_bytes(b"FAKE-SQLITE-BYTES-1234567890")
    (root / "index_default.bin").write_bytes(b"hnsw-default")
    (root / "index_proj_zeus.bin").write_bytes(b"hnsw-zeus")
    # Junk that must NOT end up in the archive.
    (root / "live").mkdir(exist_ok=True)
    (root / "live" / "scratch").write_bytes(b"junk")
    (root / "memories.db.bak").write_bytes(b"older backup, not core")


def test_backup_writes_archive_with_only_canonical_files(tmp_path):
    store = tmp_path / "store"
    _make_store(store)
    archive = backup(store_path=store, out=tmp_path / "snap.tar.gz")
    assert archive.exists()
    with tarfile.open(archive) as tf:
        names = sorted(m.name for m in tf.getmembers())
    assert "memories.db" in names
    assert "index_default.bin" in names
    assert "index_proj_zeus.bin" in names
    # Non-canonical files must not be archived.
    assert "memories.db.bak" not in names
    assert all("live" not in n for n in names)


def test_backup_default_path_uses_home_subdir(tmp_path, monkeypatch):
    store = tmp_path / "store"
    _make_store(store)
    home = tmp_path / "fakehome"
    monkeypatch.setenv("HOME", str(home))
    archive = backup(store_path=store, out=None)
    assert str(archive).startswith(str(home / ".mnemonics-backups"))
    assert archive.suffix == ".gz"


def test_backup_missing_store_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        backup(store_path=tmp_path / "does-not-exist")


def test_backup_empty_store_produces_valid_archive(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    archive = backup(store_path=empty, out=tmp_path / "snap.tar.gz")
    with tarfile.open(archive) as tf:
        assert tf.getmembers() == []


def test_restore_into_empty_destination(tmp_path):
    src = tmp_path / "src"
    _make_store(src)
    archive = backup(store_path=src, out=tmp_path / "snap.tar.gz")

    dest = tmp_path / "dest"
    written = restore(archive=archive, store_path=dest)
    assert set(written) == {"memories.db", "index_default.bin", "index_proj_zeus.bin"}
    assert (dest / "memories.db").read_bytes() == b"FAKE-SQLITE-BYTES-1234567890"
    assert (dest / "index_default.bin").read_bytes() == b"hnsw-default"


def test_restore_refuses_to_overwrite_without_force(tmp_path):
    src = tmp_path / "src"
    _make_store(src)
    archive = backup(store_path=src, out=tmp_path / "snap.tar.gz")

    dest = tmp_path / "dest"
    _make_store(dest)  # already populated
    with pytest.raises(FileExistsError):
        restore(archive=archive, store_path=dest, force=False)


def test_restore_with_force_overwrites(tmp_path):
    src = tmp_path / "src"
    _make_store(src)
    (src / "memories.db").write_bytes(b"NEW-PAYLOAD")
    archive = backup(store_path=src, out=tmp_path / "snap.tar.gz")

    dest = tmp_path / "dest"
    _make_store(dest)
    restore(archive=archive, store_path=dest, force=True)
    assert (dest / "memories.db").read_bytes() == b"NEW-PAYLOAD"


def test_restore_missing_archive_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        restore(archive=tmp_path / "nope.tar.gz", store_path=tmp_path / "dest")


def test_is_allowed_rejects_traversal_and_absolute_paths():
    assert not _is_allowed("../etc/passwd")
    assert not _is_allowed("/etc/passwd")
    assert not _is_allowed("live/scratch")
    assert _is_allowed("memories.db")
    assert _is_allowed("index_default.bin")


def test_restore_skips_malicious_archive_members(tmp_path):
    """Defense-in-depth: a hand-crafted tarball with a `../` member must not escape."""
    arc = tmp_path / "evil.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        # legit member
        legit = tarfile.TarInfo(name="memories.db")
        legit.size = 5
        tf.addfile(legit, fileobj=__import__("io").BytesIO(b"hello"))
        # path-traversal attempt
        evil = tarfile.TarInfo(name="../escape.txt")
        evil.size = 3
        tf.addfile(evil, fileobj=__import__("io").BytesIO(b"BAD"))

    dest = tmp_path / "dest"
    written = restore(archive=arc, store_path=dest)
    assert written == ["memories.db"]
    # The traversal target must not exist anywhere outside dest.
    assert not (tmp_path / "escape.txt").exists()


def test_restore_skips_directory_member(tmp_path):
    """Line 106: non-file member (directory) → skipped, no crash."""
    arc = tmp_path / "withdir.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        # Add a real file
        fi = tarfile.TarInfo(name="memories.db")
        fi.size = 4
        tf.addfile(fi, fileobj=__import__("io").BytesIO(b"data"))
        # Add a directory entry (isfile() → False)
        di = tarfile.TarInfo(name="somedir")
        di.type = tarfile.DIRTYPE
        tf.addfile(di)

    dest = tmp_path / "dest"
    written = restore(archive=arc, store_path=dest)
    assert "memories.db" in written
    assert "somedir" not in written


def test_restore_skips_member_with_none_extractfile(tmp_path):
    """Line 114: extractfile() returns None for symlinks — must be skipped."""
    arc = tmp_path / "withlink.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        # Real file first
        fi = tarfile.TarInfo(name="memories.db")
        fi.size = 4
        tf.addfile(fi, fileobj=__import__("io").BytesIO(b"data"))
        # Symlink — extractfile() returns None for these
        lnk = tarfile.TarInfo(name="index_default.bin")
        lnk.type = tarfile.SYMTYPE
        lnk.linkname = "memories.db"
        tf.addfile(lnk)

    dest = tmp_path / "dest"
    written = restore(archive=arc, store_path=dest)
    assert written == ["memories.db"]  # symlink skipped, not extracted
