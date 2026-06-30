#!/usr/bin/env python3
"""
Live peer feed reader — conflict warning.

Triggered as PreToolUse hook for file-modifying tools (Edit/Write/MultiEdit/NotebookEdit)
and Bash with git commit/push.

Reads ~/.mnemonics/live/<project>.jsonl, finds events from OTHER sessions in
the last WINDOW_MIN minutes that touched the same file or branch, and emits
a warning via Claude Code's PreToolUse JSON hook output:
  {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                          "additionalContext": "<warn text>"}}

For Metis: just print plain text to stdout (Metis injects hook stdout as system msg).

Always exit 0 (warn-only, never block).
"""

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

LIVE_DIR = Path(os.environ.get("MNEMONICS_LIVE_DIR", os.path.expanduser("~/.mnemonics/live")))
# Per-kind windows: file edits decay fast, git ops linger (rebase risk persists)
WINDOW_EDIT_MIN = int(os.environ.get("LIVE_WINDOW_EDIT_MIN", "10"))
WINDOW_GIT_MIN  = int(os.environ.get("LIVE_WINDOW_GIT_MIN",  "60"))

GIT_OPS = {"commit", "push", "checkout", "merge", "rebase", "reset", "cherry-pick"}
SHELL_SEPS = {"&&", "||", ";", "|", "(", "{", "&"}

def _preceding_cd_dir(tokens: list[str], git_idx: int) -> str | None:
    """Walk back from the `git` token to the start of its shell-separator
    chain and return the directory of a leading `cd <dir>` segment, e.g.
    `cd /x/hermes && git commit ...` -> "/x/hermes". None if that segment
    isn't a bare `cd <dir>`."""
    end = git_idx - 1
    while end >= 0 and tokens[end] in SHELL_SEPS:
        end -= 1
    start = end
    while start >= 0 and tokens[start] not in SHELL_SEPS:
        start -= 1
    start += 1
    if start > end:
        return None
    segment = tokens[start:end + 1]
    if len(segment) == 2 and segment[0] == "cd":
        return segment[1]
    return None

def detect_git_invocation(cmd: str) -> tuple[str, str | None] | None:
    """Return (op, target_dir); mirrors live_writer.py's detector so a
    command's OWN `-C <dir>` / `cd <dir> &&` target (not just the session's
    cwd) is used to pick the branch + feed this command is checked against."""
    if not cmd or "git" not in cmd:
        return None
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        m = re.search(r"(?:^|[\s;&|(){}])git\s+([a-z\-]+)", cmd)
        if m and m.group(1) in GIT_OPS:
            return (m.group(1), None)
        return None
    for i, tok in enumerate(tokens):
        if tok != "git":
            continue
        if i != 0:
            prev = tokens[i - 1]
            if not (prev in SHELL_SEPS or prev.endswith(("&", ";"))):
                continue
        j = i + 1
        git_dir = None
        if j + 1 < len(tokens) and tokens[j] == "-C":
            git_dir = tokens[j + 1]
            j += 2
        if j >= len(tokens) or tokens[j] not in GIT_OPS:
            continue
        if git_dir is None:
            git_dir = _preceding_cd_dir(tokens, i)
        return (tokens[j], git_dir)
    return None

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

def parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def recent_events(feed: Path, my_session: str) -> list[dict]:
    """Return all peer events from the last max(WINDOW_EDIT, WINDOW_GIT) minutes.
    Per-kind cutoff is applied later in conflict_text."""
    if not feed.exists():
        return []
    max_window = max(WINDOW_EDIT_MIN, WINDOW_GIT_MIN)
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max_window)
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
                    continue  # self-loop guard
                if e.get("closed"):
                    continue  # event was sealed by Director.finalize() — stale peer
                ts = parse_ts(e.get("ts", ""))
                if not ts or ts < cutoff:
                    continue
                out.append(e)
    except Exception:
        return []
    return out

def conflict_text(events: list[dict], target_path: str | None, target_branch: str | None) -> str | None:
    now = datetime.now(timezone.utc)
    edit_cutoff = now - timedelta(minutes=WINDOW_EDIT_MIN)
    git_cutoff = now - timedelta(minutes=WINDOW_GIT_MIN)
    hits = []
    for e in events:
        kind = e.get("kind")
        path = e.get("path", "")
        ts = parse_ts(e.get("ts", "")) or now
        if target_path and kind in ("edit", "write") and path == target_path and ts >= edit_cutoff:
            hits.append(f"  • session {e['session']} ({e['agent']}) {e['ts']}: {kind} {path}")
        elif target_branch and kind == "git" and path == target_branch and ts >= git_cutoff:
            hits.append(f"  • session {e['session']} ({e['agent']}) {e['ts']}: {e['note']} on {path}")
    if not hits:
        return None
    return (
        "ÇAKIŞMA UYARISI (peer aktivitesi — file-edit penceresi "
        f"{WINDOW_EDIT_MIN}dk, git penceresi {WINDOW_GIT_MIN}dk):\n"
        + "\n".join(hits)
        + "\nDikkatli ilerle, gerekirse pull/rebase yap veya kullanıcıya bildir."
    )

def metis_payload_from_env() -> dict | None:
    tool_name = os.environ.get("METIS_TOOL_NAME")
    if not tool_name:
        return None
    args_raw = os.environ.get("METIS_TOOL_ARGS", "")
    try:
        tool_input = json.loads(args_raw) if args_raw else {}
    except Exception:
        tool_input = {"command": args_raw} if "Bash" in tool_name else {}
    return {
        "tool_name": tool_name,
        "tool_input": tool_input,
        "cwd": os.environ.get("METIS_CWD") or os.getcwd(),
        "session_id": os.environ.get("METIS_SESSION_ID") or "",
    }

def main() -> int:
    metis_mode = "--metis" in sys.argv

    if metis_mode:
        payload = metis_payload_from_env()
        if not payload:
            return 0
    else:
        raw = sys.stdin.read()
        if not raw.strip():
            return 0
        try:
            payload = json.loads(raw)
        except Exception:
            return 0

    tool_name = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input") or {}
    cwd = payload.get("cwd") or os.getcwd()
    my_session = (payload.get("session_id") or "")[:8]
    project = project_root(cwd)
    feed = feed_path(project)

    target_path = None
    target_branch = None

    if tool_name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        target_path = tool_input.get("file_path") or tool_input.get("notebook_path")
        # Mirror writer logic: derive project from the target file, not session cwd
        if target_path:
            project = project_root(os.path.dirname(target_path) or cwd)
            feed = feed_path(project)
    elif tool_name == "Bash":
        cmd = tool_input.get("command", "")
        invocation = detect_git_invocation(cmd)
        if not invocation:
            return 0
        _op, git_dir = invocation
        # `-C <dir>` / `cd <dir> &&` overrides the session cwd: check this
        # command's OWN target repo for conflicts, not wherever the session
        # itself happens to be sitting.
        target_dir = os.path.normpath(os.path.join(cwd, git_dir)) if git_dir else cwd
        if git_dir:
            project = project_root(target_dir)
            feed = feed_path(project)
        try:
            target_branch = subprocess.check_output(
                ["git", "-C", target_dir, "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL, timeout=2,
            ).decode().strip()
        except Exception:
            target_branch = None
    else:
        return 0

    events = recent_events(feed, my_session)
    text = conflict_text(events, target_path, target_branch)
    if not text:
        return 0

    # Claude Code: JSON output with additionalContext
    if metis_mode:
        print(text)
    else:
        out = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "additionalContext": text,
            }
        }
        print(json.dumps(out))
    return 0

if __name__ == "__main__":
    sys.exit(main())
