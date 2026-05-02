# Mnemonics

**Your memory doesn't hallucinate.**

Mnemonics is a local-first AI memory layer that stores, retrieves, and *verifies* what it returns. Built on sentence embeddings + HNSW vector search, with optional [halluguard](https://github.com/nakata-app/halluguard) verification to flag results that drift from the indexed corpus.

## Why

Every RAG pipeline has the same silent failure mode: the retriever returns plausible-looking chunks, the LLM fills in the gaps, and nobody notices the fabrication until it matters. Mnemonics surfaces that problem at retrieval time, not after.

## Install

```bash
pip install mnemonics
# with verification support:
pip install "mnemonics[verify]"
```

## Quick start

```bash
# Store something
mnemonics ingest "The Eiffel Tower is 330 meters tall and located in Paris."

# Retrieve with hallucination check
mnemonics retrieve "how tall is the Eiffel Tower"
# trust_score: 1.0  flagged: 0
#   [0.912] The Eiffel Tower is 330 meters tall and located in Paris.
```

## Python API

```python
from mnemonics.store import Store
from mnemonics.ingest import ingest
from mnemonics.retrieve import retrieve

store = Store("~/.mnemonics")

ingest(["Paris is the capital of France.", "Rome is the capital of Italy."], store)

result = retrieve("what is the capital of France", store, top_k=3, verify=True)
for r in result["results"]:
    flag = " FLAGGED" if r["flagged"] else ""
    print(f"[{r['score']:.3f}]{flag} {r['text']}")

print(f"trust_score: {result['trust_score']}")
```

## REST server

```bash
mnemonics serve --port 7810
```

| Method | Path | Body |
|--------|------|------|
| POST | `/ingest` | `{"texts": [...], "ns": "default"}` |
| POST | `/retrieve` | `{"query": "...", "top_k": 5, "verify": true}` |
| GET | `/health` | |
| GET | `/namespaces` | |
| GET | `/count?ns=default` | |
| DELETE | `/memory/<id>` | |

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

retrieve -> embed query -> knn search -> halluguard verify -> ranked results
```

Storage layout under `~/.mnemonics`:

```
memories.db        SQLite (text, meta, timestamps)
index_default.bin  hnswlib index for "default" namespace
index_<ns>.bin     one index per namespace
```

## Verification

When `verify=True`, retrieved chunks are sent to a local halluguard daemon (port 7801) which cross-checks each result against the full retrieved corpus. Results that diverge get flagged and the aggregate `trust_score` drops.

```bash
pip install "mnemonics[verify]"
halluguard serve &
mnemonics retrieve "your query"  # auto-verifies
```

Verification is best-effort: if the daemon is not running, retrieval proceeds normally with `trust_score: 1.0`.

## License

MIT
