# Mnemonics

`/nɪˈmɒnɪks/`, *ni-MON-iks* (the "m" is silent, like "memories" with an N).

**Local-first AI memory with tier-aware decay.**

Mnemonics is a small memory layer that stores text, retrieves it with semantic search, and decays old entries the way a brain does, slowly for the things you use often, faster for ambient noise. No cloud, no telemetry, no daemon required.

## Why

Most AI memory tools push your conversations to a hosted service. Mnemonics doesn't. Your index, your DB, your machine. The library is small enough to read in one sitting, and every retrieval is fully transparent: you see the raw cosine score, the decay factor, the age, and the tier on every result, so nothing is silently demoted behind your back.

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
- `mnemonics_forget`
- `mnemonics_pin`
- `mnemonics_tier`
- `mnemonics_gc`

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
      -> SQLite metadata store (id, ns, text, meta, created,
                                tier, last_accessed, access_count)

retrieve -> embed query -> knn search -> tier-aware decay -> ranked results
         -> UPDATE last_accessed, access_count on retrieved rows
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

- DB encryption-at-rest (SQLCipher) is not currently enabled. The SQLite file at `~/.mnemonics/memories.db` is plaintext on disk; protect it with full-disk encryption (FileVault on macOS) until first-class encryption support lands.

## License

MIT
