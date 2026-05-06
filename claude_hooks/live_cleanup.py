#!/usr/bin/env python3
"""
Live feed pruner — runs at SessionStart (Claude + Metis).

Walks ~/.mnemonics/live/*.jsonl and rewrites each file keeping only events
within KEEP_HOURS (default 24). Empty files are deleted.

Idempotent, fast, fail-silent. Never crashes session start.
"""

import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

LIVE_DIR = Path(os.environ.get("MNEMONICS_LIVE_DIR", os.path.expanduser("~/.mnemonics/live")))
KEEP_HOURS = int(os.environ.get("LIVE_KEEP_HOURS", "24"))

def parse_ts(ts: str):
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def prune_file(path: Path, cutoff: datetime) -> tuple[int, int]:
    kept = 0
    dropped = 0
    new_lines = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    dropped += 1
                    continue
                ts = parse_ts(e.get("ts", ""))
                if ts and ts >= cutoff:
                    new_lines.append(line)
                    kept += 1
                else:
                    dropped += 1
    except Exception:
        return (0, 0)

    if kept == 0:
        try:
            path.unlink()
        except Exception:
            pass
        return (0, dropped)

    if dropped == 0:
        return (kept, 0)  # nothing to rewrite

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except Exception:
            pass
    return (kept, dropped)

def main() -> int:
    if not LIVE_DIR.exists():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(hours=KEEP_HOURS)
    total_kept = 0
    total_dropped = 0
    for p in LIVE_DIR.glob("*.jsonl"):
        try:
            k, d = prune_file(p, cutoff)
            total_kept += k
            total_dropped += d
        except Exception:
            continue
    if "--verbose" in sys.argv and (total_kept or total_dropped):
        print(f"live cleanup: kept={total_kept} dropped={total_dropped}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())
