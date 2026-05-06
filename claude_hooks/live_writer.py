#!/usr/bin/env python3
"""
Live peer feed writer.

Reads Claude Code / Metis hook JSON from stdin (PostToolUse or PreToolUse for git ops),
classifies the event, and appends a one-line JSON record to
  ~/.mnemonics/live/<project_hash>.jsonl

Schema:
  {"ts": ISO8601, "session": <8char>, "agent": "claude"|"metis",
   "kind": "edit"|"write"|"git"|"decision",
   "path": "...", "note": "...", "project": "/abs/path"}

Triggered by:
  - PostToolUse Edit/Write/MultiEdit/NotebookEdit  → kind=edit/write
  - PreToolUse Bash where command contains git commit|push|checkout|merge|rebase
       → kind=git
  - UserPromptSubmit when user message looks decision-flavored (heuristic)
       → kind=decision
"""

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LIVE_DIR = Path(os.environ.get("MNEMONICS_LIVE_DIR", os.path.expanduser("~/.mnemonics/live")))
LIVE_DIR.mkdir(parents=True, exist_ok=True)

GIT_OPS = {"commit", "push", "checkout", "merge", "rebase", "reset", "cherry-pick"}
SHELL_SEPS = {"&&", "||", ";", "|", "(", "{", "&"}

def detect_git_op(cmd: str) -> str | None:
    """Return the git op name iff the command actually invokes git as a program
    (not when 'git push' appears inside a quoted string like echo "git push")."""
    if not cmd or "git" not in cmd:
        return None
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        # Fallback to a stricter regex for malformed quoting:
        # require start-of-string or shell separator before 'git'
        m = re.search(r"(?:^|[\s;&|(){}])git\s+([a-z\-]+)", cmd)
        if m and m.group(1) in GIT_OPS:
            return m.group(1)
        return None
    for i, tok in enumerate(tokens):
        if tok != "git" or i + 1 >= len(tokens):
            continue
        nxt = tokens[i + 1]
        if nxt not in GIT_OPS:
            continue
        if i == 0:
            return nxt
        prev = tokens[i - 1]
        if prev in SHELL_SEPS or prev.endswith(("&", ";")):
            return nxt
    return None
DECISION_KEYWORDS = (
    "kararım", "şunu yapalım", "şunu kullan", "şunu seç",
    "implement et", "bunu yap", "bunu değiştir", "şuradan devam",
    "let's go with", "decision", "i'll use",
)

def project_root(cwd: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL, timeout=2,
        )
        root = out.decode().strip()
        return root or cwd
    except Exception:
        return cwd

def project_id(project: str) -> str:
    h = hashlib.sha1(project.encode()).hexdigest()[:12]
    base = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(project)) or "root"
    return f"{base}-{h}"

def classify(payload: dict, agent: str) -> dict | None:
    """Return event dict or None to skip."""
    event_name = payload.get("hook_event_name") or payload.get("event") or ""
    tool_name = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input") or {}
    cwd = payload.get("cwd") or os.getcwd()
    session = (payload.get("session_id") or "")[:8]
    project = project_root(cwd)

    base = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session": session or "?",
        "agent": agent,
        "project": project,
    }
    # Director-spawned children inherit DIRECTOR_RUN_ID via env; tag the event
    # so that finalize() can later mark these as "closed" and prevent leakage
    # into subsequent runs as phantom peer activity.
    director_run = os.environ.get("DIRECTOR_RUN_ID", "")
    if director_run:
        base["director_run"] = director_run

    # File-edit tools
    if event_name == "PostToolUse" and tool_name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
        if not path:
            return None
        kind = "write" if tool_name == "Write" else "edit"
        note = ""
        if tool_name == "MultiEdit":
            edits = tool_input.get("edits", [])
            note = f"{len(edits)} edits"
        # The project bucket follows the file's own location, not the session cwd.
        # Otherwise editing ~/.claude/hooks/foo.py from a session cwd'd in
        # peer-test would dump the event into peer-test's feed.
        file_project = project_root(os.path.dirname(path) or cwd)
        return {**base, "project": file_project, "kind": kind, "path": path, "note": note}

    # Git ops via Bash
    if event_name == "PreToolUse" and tool_name == "Bash":
        cmd = tool_input.get("command", "")
        op = detect_git_op(cmd)
        if op:
            try:
                branch = subprocess.check_output(
                    ["git", "-C", cwd, "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL, timeout=2,
                ).decode().strip()
            except Exception:
                branch = "?"
            return {**base, "kind": "git", "path": branch, "note": f"git {op}"}
        return None

    # Decision heuristic (user prompt)
    if event_name == "UserPromptSubmit":
        prompt = payload.get("prompt") or payload.get("user_prompt") or ""
        if not prompt or len(prompt) < 30:
            return None
        low = prompt.lower()
        if any(k in low for k in DECISION_KEYWORDS):
            return {**base, "kind": "decision", "path": "", "note": prompt[:120]}
        return None

    return None

def append_event(event: dict) -> None:
    pid = project_id(event["project"])
    feed = LIVE_DIR / f"{pid}.jsonl"
    line = json.dumps(event, ensure_ascii=False)
    with open(feed, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def metis_payload_from_env() -> dict | None:
    """Metis passes hook context via env vars, not stdin JSON."""
    tool_name = os.environ.get("METIS_TOOL_NAME")
    user_prompt = os.environ.get("METIS_USER_PROMPT")
    event = None
    for arg in sys.argv:
        if arg.startswith("--event="):
            event = arg.split("=", 1)[1]
    if not event:
        return None

    payload: dict = {
        "hook_event_name": event,
        "cwd": os.environ.get("METIS_CWD") or os.getcwd(),
        "session_id": os.environ.get("METIS_SESSION_ID") or "",
    }

    if tool_name:
        payload["tool_name"] = tool_name
        args_raw = os.environ.get("METIS_TOOL_ARGS", "")
        try:
            payload["tool_input"] = json.loads(args_raw) if args_raw else {}
        except Exception:
            payload["tool_input"] = {"command": args_raw} if "Bash" in tool_name else {}

    if user_prompt:
        payload["prompt"] = user_prompt

    return payload

def main() -> int:
    metis_mode = "--metis" in sys.argv
    agent = "metis" if metis_mode else "claude"

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

    event = classify(payload, agent)
    if event:
        try:
            append_event(event)
        except Exception:
            pass
    return 0

if __name__ == "__main__":
    sys.exit(main())
