# claude_hooks, Live peer feed

Companion Claude Code hooks built on top of the mnemonics live event store
(`~/.mnemonics/live/<project_hash>.jsonl`). They make multi-instance Claude
sessions in the same project see each other's edits and surface conflicts
before two instances overwrite the same file.

## Files

| Hook | Event | Purpose |
|---|---|---|
| `live_writer.py` | `PreToolUse` (Bash), `PostToolUse` (Edit/Write/MultiEdit), `UserPromptSubmit` | Append a one-line JSON event for every edit, git op, or new prompt. |
| `live_reader.py` | `PreToolUse` (Bash, Edit/Write/MultiEdit) | Read the store, raise a `ÇAKIŞMA UYARISI` if another session touched the same file (≤10 min) or branch (≤60 min). |
| `live_observer.py` | `UserPromptSubmit` | At the start of each user turn, summarize peer activity in the last N minutes. Stays silent if no peer. |
| `live_cleanup.py` | optional | Trim the JSONL store. |

## Install

Symlink each script from `~/.claude/hooks/` to this directory and register
them in `~/.claude/settings.json`:

```bash
ln -s "$PWD/live_writer.py"   ~/.claude/hooks/live_writer.py
ln -s "$PWD/live_reader.py"   ~/.claude/hooks/live_reader.py
ln -s "$PWD/live_observer.py" ~/.claude/hooks/live_observer.py
ln -s "$PWD/live_cleanup.py"  ~/.claude/hooks/live_cleanup.py
```

settings.json registration:

```json
{
  "hooks": {
    "PreToolUse": [
      {"matcher": "Bash", "hooks": [
        {"type": "command", "command": "~/.claude/hooks/live_writer.py"},
        {"type": "command", "command": "~/.claude/hooks/live_reader.py"}
      ]},
      {"matcher": "Edit|Write|MultiEdit", "hooks": [
        {"type": "command", "command": "~/.claude/hooks/live_reader.py"}
      ]}
    ],
    "PostToolUse": [
      {"matcher": "Write|Edit|MultiEdit", "hooks": [
        {"type": "command", "command": "~/.claude/hooks/live_writer.py"}
      ]}
    ],
    "UserPromptSubmit": [
      {"matcher": "", "hooks": [
        {"type": "command", "command": "~/.claude/hooks/live_writer.py"},
        {"type": "command", "command": "~/.claude/hooks/live_observer.py"}
      ]}
    ]
  }
}
```

## Event store

Each project gets its own JSONL file at
`~/.mnemonics/live/<sha1(cwd)[:12]>.jsonl`. One line per event:

```json
{"ts": "2026-05-06T15:36:16+00:00", "session": "540e6c04",
 "agent": "claude", "kind": "edit", "path": "/abs/path", "note": "...",
 "closed": false}
```

`closed: true` is set by Director.finalize() when a session ends, so stale
events are ignored by the reader.

## Why companion to mnemonics

The hooks reuse the `~/.mnemonics/live/` storage location and ride alongside
the mnemonics MCP server, but they do not depend on the embedding/HNSW core.
They are session-coordination tooling, not memory retrieval, and live in
this folder rather than the main package to keep `pip install mnemonics`
free of Claude-Code-specific behavior.
