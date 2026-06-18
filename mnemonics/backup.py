"""Backup + restore of a mnemonics store.

A store is just a directory containing:
  memories.db        SQLite metadata (rows, FTS5 mirror, tier, access_count)
  index_<ns>.bin     One hnswlib index file per namespace

This module bundles those into a single .tar.gz and unbundles them. Anything
else in the directory (legacy `.bak` files, `live/`, `adaptmem-model/`,
`backups/`) is left alone — restore only writes the canonical files.
"""
from __future__ import annotations

import tarfile
from datetime import datetime
from pathlib import Path


# Files we ship in a backup. Index files are namespace-scoped (`index_<ns>.bin`)
# and discovered at runtime, so this list is the *fixed* portion only.
_CORE_FILES = ("memories.db",)

# Substring filter for archive members: a file may be restored only if its
# arcname starts with one of these prefixes. Protects against tarball traversal
# (`../`) and stray files (model checkpoints, backups of backups, etc).
_ALLOWED_PREFIXES = ("memories.db", "index_")


def _store_files(store_root: Path) -> list[Path]:
    """Canonical files we put into a backup. Only those that actually exist."""
    files: list[Path] = []
    for name in _CORE_FILES:
        p = store_root / name
        if p.is_file():
            files.append(p)
    for p in sorted(store_root.glob("index_*.bin")):
        if p.is_file():
            files.append(p)
    return files


def _default_out_path() -> Path:
    return Path.home() / ".mnemonics-backups" / (
        datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".tar.gz"
    )


def backup(store_path: str | Path = "~/.mnemonics", out: str | Path | None = None) -> Path:
    """Bundle the store's canonical files into a .tar.gz. Returns the archive path.

    Raises FileNotFoundError if the store directory itself doesn't exist.
    Empty stores (no memories.db yet) produce a valid but empty archive.
    """
    root = Path(store_path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"store path not found: {root}")

    out_path = Path(out).expanduser() if out else _default_out_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = _store_files(root)
    with tarfile.open(out_path, mode="w:gz") as tf:
        for f in files:
            # arcname = filename only — restore writes back into the dest root
            # without any nested directory structure.
            tf.add(str(f), arcname=f.name)
    return out_path


def _is_allowed(arcname: str) -> bool:
    # Reject absolute paths and parent-traversal up front.
    if arcname.startswith("/") or ".." in Path(arcname).parts:
        return False
    return any(arcname.startswith(pfx) for pfx in _ALLOWED_PREFIXES)


def restore(
    archive: str | Path,
    store_path: str | Path = "~/.mnemonics",
    force: bool = False,
) -> list[str]:
    """Extract a backup archive into store_path. Returns the list of files written.

    By default, refuses to overwrite an existing non-empty store (any canonical
    file already present). Pass force=True to overwrite. Members outside the
    canonical set (../path, model dirs, etc.) are silently skipped.
    """
    arc = Path(archive).expanduser()
    if not arc.is_file():
        raise FileNotFoundError(f"archive not found: {arc}")

    dest = Path(store_path).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    if not force:
        existing = _store_files(dest)
        if existing:
            raise FileExistsError(
                f"store at {dest} already contains {len(existing)} file(s); "
                "pass force=True to overwrite"
            )

    written: list[str] = []
    with tarfile.open(arc, mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            if not _is_allowed(member.name):
                continue
            # Force a flat extraction — discard any directory prefix that
            # snuck into the arcname despite the prefix check.
            target = dest / Path(member.name).name
            src = tf.extractfile(member)
            if src is None:  # pragma: no cover — isfile() guard above makes this unreachable
                continue
            data = src.read()
            target.write_bytes(data)
            written.append(target.name)
    return written
