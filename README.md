# Mnemonics

`/nɪˈmɒnɪks/`, *ni-MON-iks* (the "m" is silent, like "memories" with an N).

**Agent memory infrastructure. Local-first, tier-aware, MCP-ready.**

Mnemonics is a memory layer for AI agents: ingest conversations or facts, retrieve the relevant ones at inference time, decay old entries the way a brain does. Works as a Python library, a REST server, or an MCP server, plug it into any agent framework in under 10 lines.

```
pip install mnemonics
```

→ [Agent integration guide](AGENTS.md) · [Benchmarks](#benchmarks) · [MCP setup](#mcp-claude-code--cursor--metis)

## Why

Most AI memory tools push your data to a hosted service. Mnemonics doesn't. Your index, your DB, your machine. Every retrieval is fully transparent: you see the raw cosine score, the decay factor, the age, and the tier on every result, nothing is silently demoted behind your back.

Works with LangChain, CrewAI, AutoGen, LlamaIndex, or any framework that can call a Python function or hit an HTTP endpoint. See [AGENTS.md](AGENTS.md) for drop-in examples.

## Install

```bash
pip install mnemonics
```

CLI alias: both `mnemonics` and the shorter `mnem` are available after install.

## Quick start

```bash
# Store something
mnem ingest "The Eiffel Tower is 330 meters tall and located in Paris."

# Retrieve (decay applied by default)
mnem retrieve "how tall is the Eiffel Tower"
#   [0.912] [raw=0.912 decay=1.00 age=0d tier=def] The Eiffel Tower is 330 meters tall ...
```

## Tiers and decay

Every memory has a **tier** that controls how aggressively its score fades over time:

| Tier | Label   | Half-life  | Use for                                   |
|------|---------|------------|-------------------------------------------|
| 0    | pinned  | no decay   | decisions, key facts, things you must keep|
| 1    | default | 90 days    | general notes (the default)               |
| 2    | ambient | 14 days    | low-confidence observations, chit-chat    |

Final score on every retrieval is:

```
score = raw_cosine × exp(-ln(2) × age_days / half_life)
```

Pinned (tier 0) entries always score at full weight. Tier-1 entries lose half their weight after 90 days, tier-2 after 14. Disable decay anytime with `--no-decay` (CLI) or `decay=false` (REST/MCP).

```bash
mnem pin <id>             # tier=0, never decays
mnem tier <id> 2          # tier=2, ambient (fast decay)
mnem retrieve "..." --no-decay   # show raw cosine scores
```

## Maintenance

```bash
# Garbage-collect tier-2 rows not accessed in 30+ days (dry-run by default)
mnem gc --ns default --age-days 30
mnem gc --ns default --age-days 30 --apply   # actually delete

# Sweep tier-1 rows older than 60 days
mnem gc --ns sessions --age-days 60 --tier 1 --apply

# Bulk-delete an entire namespace (dry-run by default)
mnem forget --ns old-project
mnem forget --ns old-project --apply         # actually delete
mnem forget --ns archive --before 2026-01-01 # date-filtered

# Store health check (DB integrity, index vs SQL counts, orphan files)
mnem doctor
mnem doctor --json                           # JSON output for scripting

# Repair orphan vectors without re-encoding
# (Use when doctor shows "orphan vector(s)", loads vectors from old index by ID)
mnem rebuild-index --ns <ns>
```

## Raw + summary

You can attach an optional one-line summary alongside the raw text. The embedding is still computed from the raw chunk, so vector recall is unchanged, but the BM25 mirror indexes both columns, so a keyword can land a hit via the gist even when the raw transcript reads differently.

```bash
mnem ingest "The full session transcript, jargon-heavy, full of code refs." \
    --summary "Zeus HTTP timeout bumped to 300s"

mnem retrieve "timeout" --hybrid
#   [...] Zeus HTTP timeout bumped to 300s
#       └─ raw: The full session transcript, jargon-heavy, full of code refs.
```

Useful when you want to keep the original wording (no summarization loss) but still let keyword search reach the row through a shorter label.

Every retrieval bumps `access_count` and `last_accessed` for the rows it returned, which sets up future reinforcement scoring without any caller action.

## Python API

```python
from mnemonics.store import Store
from mnemonics.ingest import ingest
from mnemonics.retrieve import retrieve

store = Store("~/.mnemonics")

ingest(["Paris is the capital of France.", "Rome is the capital of Italy."], store)

result = retrieve("what is the capital of France", store, top_k=3)
for r in result["results"]:
    print(f"[{r['score']:.3f}] tier={r['tier']} age={r['age_days']:.0f}d  {r['text']}")
```

`store.pin(id)` and `store.set_tier(id, tier)` change a memory's tier directly.

## REST server

```bash
mnem serve --port 7810
```

The server binds to `127.0.0.1` only, no external interface, no telemetry.

| Method | Path | Body |
|--------|------|------|
| POST | `/ingest` | `{"texts": [...], "ns": "default"}` |
| POST | `/retrieve` | `{"query": "...", "top_k": 5, "decay": true}` |
| GET | `/health` | |
| GET | `/namespaces` | |
| GET | `/count?ns=default` | |
| DELETE | `/memory/<id>` | |

## MCP (Claude Code / Cursor / Metis)

```bash
mnem mcp
```

```json
{
  "mcpServers": {
    "mnemonics": {
      "command": "mnem",
      "args": ["mcp"]
    }
  }
}
```

Tools exposed:
- `mnemonics_ingest`
- `mnemonics_retrieve` (decay-aware, supports `decay: false` for raw cosine)
- `mnemonics_forget` (delete by id)
- `mnemonics_pin`
- `mnemonics_tier`
- `mnemonics_gc` (supports `tier: 1` for default-tier rows)
- `mnemonics_stats`
- `mnemonics_health` (DB integrity + index health report as JSON)

## Namespaces

Isolate memories by project, user, or any key:

```bash
mnem ingest "project notes..." --ns work
mnem retrieve "deadlines" --ns work
```

## Architecture

```
texts -> chunk (200w / 40w overlap) -> embed (all-MiniLM-L6-v2)
      -> hnswlib cosine index (per namespace)
      -> SQLite metadata store (id, ns, text, summary, meta, created,
                                tier, last_accessed, access_count)
      -> FTS5 mirror (text + summary), BM25 keyword surface

retrieve -> embed query -> knn search -> tier-aware decay -> ranked results
         -> UPDATE last_accessed, access_count on retrieved rows
         -> --hybrid: fuse vector top-k and BM25 top-k via RRF
```

Storage layout under `~/.mnemonics`:

```
memories.db        SQLite (text, meta, tier, access counters, timestamps)
index_<ns>.bin     hnswlib index for each namespace
```

## Privacy

- The REST server binds to `127.0.0.1` only. There is no `0.0.0.0` flag.
- The mnemonics package contains no outbound HTTP, no telemetry, no analytics.
- **First-run network:** `sentence-transformers` downloads the `all-MiniLM-L6-v2` model (~90 MB) from Hugging Face Hub on the first `ingest` or `retrieve`. The model caches under `~/.cache/huggingface/`. After that first download, you can pin the package fully offline:

  ```bash
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  ```

- **DB encryption-at-rest** is opt-in via SQLCipher (added in 0.3.0). The default install keeps `~/.mnemonics/memories.db` as plaintext, so full-disk encryption (FileVault, LUKS) still matters for the default path. To turn it on:

  ```bash
  # macOS: brew install sqlcipher first (Linux: apt install libsqlcipher-dev)
  pip install 'mnemonics[encrypt]'
  mnemonics encrypt-db          # one-shot migrate ~/.mnemonics/memories.db
  export MNEMONICS_ENCRYPT=1    # add to your shell rc so future runs see it
  ```

  The migration auto-generates a 256-bit key and stores it in the OS keyring (macOS Keychain, Linux SecretService, Windows DPAPI). Pass `--key <hex>` to supply your own, or `--no-keyring` to print the key for manual storage. The pre-encryption DB is preserved as `memories.db.preencrypt-<timestamp>` so the operation is reversible. Losing the key after the original backup is gone loses the DB.

## License

MIT
