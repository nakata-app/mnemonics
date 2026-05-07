# Mnemonics

`/nɪˈmɒnɪks/`, *ni-MON-iks* (the "m" is silent, like "memories" with an N).

**Local-first AI memory.**

Mnemonics is a small, local memory layer that stores text and retrieves it with semantic search. Built on sentence embeddings + HNSW vector search, persisted to SQLite. No cloud, no telemetry, no daemon required for day-to-day use.

## Why

Most AI memory tools push your conversations to a hosted service. Mnemonics doesn't. Your index, your DB, your machine. The library is small enough to read in one sitting.

## Install

```bash
pip install mnemonics
```

## Quick start

```bash
# Store something
mnemonics ingest "The Eiffel Tower is 330 meters tall and located in Paris."

# Retrieve
mnemonics retrieve "how tall is the Eiffel Tower"
#   [0.912] The Eiffel Tower is 330 meters tall and located in Paris.
```

## Python API

```python
from mnemonics.store import Store
from mnemonics.ingest import ingest
from mnemonics.retrieve import retrieve

store = Store("~/.mnemonics")

ingest(["Paris is the capital of France.", "Rome is the capital of Italy."], store)

result = retrieve("what is the capital of France", store, top_k=3)
for r in result["results"]:
    print(f"[{r['score']:.3f}] {r['text']}")
```

## REST server

```bash
mnemonics serve --port 7810
```

| Method | Path | Body |
|--------|------|------|
| POST | `/ingest` | `{"texts": [...], "ns": "default"}` |
| POST | `/retrieve` | `{"query": "...", "top_k": 5}` |
| GET | `/health` | |
| GET | `/namespaces` | |
| GET | `/count?ns=default` | |
| DELETE | `/memory/<id>` | |

The server binds to `127.0.0.1` only. No external interface.

## MCP (Claude Code / Cursor / Metis)

```bash
mnemonics mcp
```

Add to your MCP config:

```json
{
  "mcpServers": {
    "mnemonics": {
      "command": "mnemonics",
      "args": ["mcp"]
    }
  }
}
```

Tools exposed: `mnemonics_ingest`, `mnemonics_retrieve`, `mnemonics_forget`

## Namespaces

Isolate memories by project, user, or any key:

```bash
mnemonics ingest "project notes..." --ns work
mnemonics retrieve "deadlines" --ns work
```

## Architecture

```
texts -> chunk (200w / 40w overlap) -> embed (all-MiniLM-L6-v2)
      -> hnswlib cosine index (per namespace)
      -> SQLite metadata store

retrieve -> embed query -> knn search -> ranked results
```

Storage layout under `~/.mnemonics`:

```
memories.db        SQLite (text, meta, timestamps)
index_default.bin  hnswlib index for "default" namespace
index_<ns>.bin     one index per namespace
```

## License

MIT
