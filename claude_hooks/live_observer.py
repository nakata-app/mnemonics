#!/usr/bin/env python3
"""
Passive peer observer — UserPromptSubmit hook.

When the user types a new message, surface a short summary of what other
sessions in the same project have been doing in the last OBSERVE_MIN minutes.
Stays silent if no peer activity (no token waste).

Output:
  Claude → JSON {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                        "additionalContext": "<text>"}}
  Metis  → plain stdout (hook stdout is auto-injected as system message)
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

LIVE_DIR = Path(os.environ.get("MNEMONICS_LIVE_DIR", os.path.expanduser("~/.mnemonics/live")))
OBSERVE_MIN = int(os.environ.get("LIVE_OBSERVE_MIN", "15"))
MAX_LINES = int(os.environ.get("LIVE_OBSERVE_MAX_LINES", "6"))

def project_root(cwd: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL, timeout=2,
        )
        return out.decode().strip() or cwd
    except Exception:
        return cwd

def project_id(project: str) -> str:
    h = hashlib.sha1(project.encode()).hexdigest()[:12]
    base = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(project)) or "root"
    return f"{base}-{h}"

def feed_path(project: str) -> Path:
    return LIVE_DIR / f"{project_id(project)}.jsonl"

def parse_ts(ts: str):
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def collect_events(feed: Path, my_session: str) -> list[dict]:
    if not feed.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=OBSERVE_MIN)
    out = []
    try:
        with open(feed, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("session") == my_session:
                    continue
                if e.get("closed"):
                    continue
                ts = parse_ts(e.get("ts", ""))
                if not ts or ts < cutoff:
                    continue
                out.append(e)
    except Exception:
        return []
    return out

def summarize(events: list[dict]) -> str | None:
    if not events:
        return None

    # Group by (session, kind) → most-recent + count
    by_peer = defaultdict(lambda: {"latest": None, "edits": set(), "git": [], "decisions": [], "director": []})
    for e in events:
        key = (e["session"], e.get("agent", "?"))
        peer = by_peer[key]
        ts = parse_ts(e.get("ts", "")) or datetime.now(timezone.utc)
        if peer["latest"] is None or ts > peer["latest"]:
            peer["latest"] = ts
        kind = e.get("kind")
        path = e.get("path", "")
        note = e.get("note", "")
        if kind in ("edit", "write") and path:
            peer["edits"].add(os.path.basename(path))
        elif kind == "git":
            peer["git"].append(f"{note} ({path})" if path else note)
        elif kind == "decision":
            peer["decisions"].append(note[:80])
        elif kind in ("plan", "spawn", "complete", "blocked"):
            peer["director"].append(f"{kind}: {note[:80]}")

    if not by_peer:
        return None

    now = datetime.now(timezone.utc)
    lines = []
    for (session, agent), data in sorted(by_peer.items(), key=lambda kv: kv[1]["latest"] or now, reverse=True):
        if len(lines) >= MAX_LINES:
            break
        ago = int((now - data["latest"]).total_seconds() / 60) if data["latest"] else 0
        bits = []
        if data["director"]:
            bits.append("; ".join(data["director"][-2:]))
        if data["edits"]:
            files = sorted(data["edits"])
            preview = ", ".join(files[:3])
            if len(files) > 3:
                preview += f" (+{len(files)-3})"
            bits.append(f"edit: {preview}")
        if data["git"]:
            bits.append("; ".join(data["git"][-2:]))
        if data["decisions"]:
            bits.append(f"karar: {data['decisions'][-1]}")
        if not bits:
            continue
        lines.append(f"  • {session} ({agent}) ~{ago}dk önce: " + " | ".join(bits))

    if not lines:
        return None
    return f"Peer aktivitesi (son {OBSERVE_MIN}dk, aynı proje):\n" + "\n".join(lines)

def metis_payload_from_env() -> dict | None:
    return {
        "cwd": os.environ.get("METIS_CWD") or os.getcwd(),
        "session_id": os.environ.get("METIS_SESSION_ID") or "",
    }

def main() -> int:
    metis_mode = "--metis" in sys.argv
    if metis_mode:
        payload = metis_payload_from_env()
    else:
        raw = sys.stdin.read()
        if not raw.strip():
            return 0
        try:
            payload = json.loads(raw)
        except Exception:
            return 0

    cwd = payload.get("cwd") or os.getcwd()
    my_session = (payload.get("session_id") or "")[:8]
    project = project_root(cwd)
    feed = feed_path(project)

    events = collect_events(feed, my_session)
    text = summarize(events)
    if not text:
        return 0

    if metis_mode:
        print(text)
    else:
        out = {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": text,
            }
        }
        print(json.dumps(out))
    return 0

if __name__ == "__main__":
    sys.exit(main())
