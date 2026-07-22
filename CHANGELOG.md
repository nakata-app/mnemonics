# Changelog

All notable changes to mnemonics. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow [SemVer](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Conflict-aware ingest (`dedup.reconcile_ingest`).** The mnemonics answer to
  Mem0's ADD/UPDATE/DELETE/NOOP, done without an LLM in the library and without
  ever hard-deleting a row:
  - **NOOP dedup**, a text ≥ 0.98 cosine to an existing memory is a
    restatement and is skipped instead of stored as a near-duplicate.
  - **Supersede (archive-not-delete)**, `supersede_map` names existing
    memories a new text replaces. The old rows get
    `meta.status='superseded'` + `superseded_by` + `superseded_at`, stay in the
    DB and vector index for audit, but drop out of normal retrieval. The
    contradiction judgment lives at the call site (an agent), not in an
    embedding heuristic, so a still-true memory is never silently lost.
- **`Store.supersede(old_id, new_id)`** records the reversible replacement link.
- **`exclude_superseded` (default `True`) on `Store.search` / `search_bm25`.**
  Retrieval hides superseded rows via `json_extract(meta,'$.status')`; pass
  `False` for audit queries that need the full history.
- **`ingest(..., return_ids=True)`** returns stored row ids (needed to link a
  supersede).
- **`mnemonics_ingest` MCP tool** gains opt-in `reconcile` (NOOP dedup) and
  `supersede` (array of ids, or `{index: [ids]}`) params. Plain append stays
  the default, session-end ingests are unchanged.

## [0.4.0] - 2026-05-26

### Added

- **Auto-resize on ingest.** `Store.add()` now calls `idx.resize_index()`
  before `add_items` when the new batch would exceed the current
  `max_elements` ceiling (default 100K). Prevents `add_items` from silently
  failing on large namespaces without any configuration change.
- **Capacity warning in health_check / doctor.** Each namespace report now
  includes `max_elements`, `usage_pct`, and `capacity_warning` (true when
  ≥ 85% full). `mnemonics doctor` shows "⚠ X% full, rebuild-index
  recommended" so the operator knows before hitting the limit.
- **`mnemonics doctor --fix`** one-shot auto-repair: rebuilds indexes with
  orphan vectors via `Store.repair()`, deletes orphan `.bin` files, reports
  missing vectors (sql > idx) that need manual `rebuild-index`.
- **`mnemonics_repair` MCP tool** exposes `Store.repair()` to AI agents.
- **`mnemonics rebuild-index --ns <ns>`** repairs orphan vectors without
  re-encoding. Reads vectors for current SQL row IDs from the on-disk
  hnswlib index via `get_items()`, writes a clean index, and returns
  `(old_count, new_count)`. Includes a guard for case-insensitive
  filesystems (macOS APFS): raises if the index path collides with another
  namespace to prevent silent overwrites.
- **`mnemonics forget --ns <ns> [--before DATE] [--tier N] [--apply]`** for
  bulk namespace cleanup without raw SQL. Dry-run by default; `--apply` to
  delete. Deletes from SQL and calls `mark_deleted` in hnswlib.
- **`mnemonics doctor [--json]`** health report: DB integrity, WAL size,
  per-namespace SQL vs index counts (orphan vectors, missing vectors), and
  orphan `.bin` files. Exit code 1 when issues found.
- **`mnemonics gc --tier 1`** enables GC for default-tier (tier=1) rows.
  Previously `gc` only targeted tier=2 (ambient). Combine with `--age-days`
  to sweep stale sessions-ns entries without raw SQL.
- **`GET /doctor` REST endpoint** returns the full `health_check()` report as
  JSON (DB integrity, WAL size, per-namespace sql/idx counts, capacity
  warnings, orphan indexes). Matches what `mnemonics doctor --json` prints.
- **`POST /repair` REST endpoint** calls `Store.repair()` and returns a JSON
  summary of what was fixed (orphan vectors rebuilt, orphan .bin files
  removed, missing vectors reported). Matches `mnemonics_repair` MCP tool.
- **Pure BM25 keyword search** across all three interfaces:
  `mnem bm25 <query> [--ns] [--top-k]` (CLI),
  `POST /search-bm25 {"query": "...", "ns": "...", "top_k": N}` (REST),
  `mnemonics_bm25` MCP tool. No vector encoding, instant, exact-token
  matching. Best for dates, IDs, names, or any query where you know the
  exact words. Covers both `text` and `summary` columns via FTS5.
- **`mnem stats` tier breakdown.** Output now shows
  `{ns}: N chunks  (pin=P def=D amb=A)` matching `mnemonics_stats` MCP.
  Previously showed only the total count.
- **`mnem list --tier`, `GET /memories?tier=`, `mnemonics_list tier` filter.**
  Filter browse results to a specific tier without writing SQL.
- **`mnem list` CLI command, `mnemonics_list` MCP tool, `GET /memories` REST
  endpoint.** Browse memories in a namespace newest-first with optional
  `--limit` / `--offset` pagination. Returns id, tier, created timestamp,
  text snippet (200 chars), and summary. Agents can now audit a namespace
  without semantic search.
- \*\*`mnemonics_get` MCP tool, `GET /memory/<id>` REST endpoint, `Store.get()`.
  Fetch a single memory by ID. Returns full text, summary, ns, tier,
  created, last_accessed, access_count. Useful before pinning or deleting.
- **`GET /stats` REST endpoint.** Per-namespace tier breakdown as JSON array.
  Each entry: `{ns, total, pin, def, amb}`. Complements `GET /namespaces`.
- **`POST /forget-ns` REST endpoint.** Bulk-delete a namespace via REST
  (mirrors `mnemonics_forget_ns`). Accepts `ns`, `before`, `tier`,
  `dry_run`. Defaults to dry-run.
- **`mnemonics_forget_ns` MCP tool** bulk-deletes a namespace from AI agents.
  Accepts `ns` (required), `before`, `tier`, `dry_run` (default true).
  Complements `mnemonics_forget` (single-id) and `mnemonics_gc` (age-based).
- **`POST /pin` and `POST /tier` REST endpoints.** Complete REST parity with
  CLI (`mnem pin`, `mnem tier`) and MCP tools. `/pin {"id": N}` pins a
  memory (tier=0). `/tier {"id": N, "tier": 0|1|2}` changes its tier.
  Returns 400 on invalid tier (values outside 0-2).
- **`mnemonics_rebuild_index` MCP tool.** Rebuild the hnswlib index for a
  specific namespace from the SQL source of truth without re-encoding.
  Completes MCP parity with `mnem rebuild-index` (CLI) and
  `POST /rebuild-index` (REST).
- **`mnemonics_stats` tier breakdown.** Output now includes per-namespace
  tier counts: `default: 37 chunks  (pin=0 def=37 amb=0)`. Agents can see
  tier distribution without raw SQL queries.
- **`POST /rebuild-index` REST endpoint.** Accepts `{"ns": "..."}`, returns
  `{ns, old_count, new_count, removed}`. Returns 409 on APFS collision.
- **`POST /gc` and `POST /forget` REST endpoints** enable programmatic
  cleanup without the CLI. Both default to `dry_run: true`. `/gc` accepts
  `ns`, `age_days`, `tier` (1 or 2). `/forget` accepts `ns` (required),
  `before`, `tier`. Matches CLI and MCP tool behavior.

### Fixed

- **`rebuild_ns_index` on empty namespace** no longer creates a stray
  empty `.bin` file (which `health_check()` would flag as an orphan). Added
  an early return when the namespace has no SQL rows.

### Changed

- **Default cross-encoder reranker upgraded to `BAAI/bge-reranker-v2-m3`.**
  Previously `cross-encoder/ms-marco-MiniLM-L-12-v2`. The new default aligns
  with adaptmem's own default (adaptmem/core.py) and with the fine-tuned CE
  lineage (mn-ce-v2) that achieved R@1 = 0.974 on LongMemEval-S (n=500).

  Override at any time via `MNEMONICS_RERANK_MODEL` env var. The model is
  downloaded on first use (~550 MB). If adaptmem is installed, it takes
  priority and may inject a locally fine-tuned CE instead.

## [0.3.0] - 2026-05-17

### Changed

- **Default retrieval method flipped to hybrid (BREAKING, behavior).**
  `retrieve()`, MCP `mnemonics_retrieve`, HTTP `POST /retrieve`, and the
  `mnemonics retrieve` CLI now default to hybrid (vector cosine + BM25
  fused via RRF). Vector-only is reachable via `hybrid=false`
  (library/MCP/HTTP) or `--no-hybrid` (CLI).

  Evidence: 400-chunk gold set sampled from real production memories,
  210 queries across three classes (exact-token, sentence-fragment,
  shuffled-keywords). Hybrid won every class with zero regressions
  (no query lost 2+ rank positions vs vector-only):

  ```
  metric   vector  hybrid    delta
  mrr      0.330   0.676    +0.347
  r@5      0.381   0.943    +0.562
  r@10     0.471   1.000    +0.529
  ndcg@10  0.363   0.757    +0.394
  ```

  Raw per-query results held off-repo (the gold set was sampled from
  private user memory). Methodology reproducible: sample ~400 chunks
  200..2000 chars, build queries in three classes (exact-token,
  sentence-fragment, shuffled-content-words), run
  `mnemonics.eval.run_eval` with `method=vector` and `method=hybrid`.

  To preserve old behavior, pass `hybrid=false` (library/server) or use
  `--no-hybrid` (CLI).

### Added

- **Hybrid retrieval parameters exposed on MCP and HTTP.**
  `mnemonics_retrieve` (MCP) and `POST /retrieve` (HTTP) accept
  `hybrid: bool` and `candidate_k: int` (default `20`). When
  `hybrid=true`, the request fuses the vector cosine top-`candidate_k`
  with the BM25 (SQLite FTS5) top-`candidate_k` via Reciprocal Rank
  Fusion, then applies the existing tier-aware decay + reinforcement
  pass on the top-`top_k`.
- **Raw + summary hybrid storage.** Every row now has an optional
  `summary` column alongside the raw `text`. Embeddings are still
  computed from the raw chunk, but the FTS5 mirror indexes both
  columns, so BM25 can hit a row through its full text or a
  shorter gist. Surfaces:
  - `mnemonics ingest <text> --summary "<one-line gist>"`
  - `POST /ingest` accepts `summaries: [string|null]` parallel to
    `texts`.
  - MCP `mnemonics_ingest` accepts `summaries`. `mnemonics_retrieve`
    output prints the summary on top and the raw chunk below it
    when one exists.
  - Schema is self-healing: pre-`summary` DBs add the column on
    next `Store(...)` open, the FTS5 mirror is dropped and rebuilt
    with the two-column layout, and the existing rows are re-mirrored
    so BM25 keeps working without manual reindex.
- **Opt-in DB encryption-at-rest** via SQLCipher. Set `MNEMONICS_ENCRYPT=1`
  and either `MNEMONICS_DB_KEY=<64-char hex>` or a key in the OS keyring
  (macOS Keychain, Linux SecretService, Windows DPAPI), and `Store` swaps
  stdlib `sqlite3` for `sqlcipher3` transparently. FTS5 search continues
  to work; HNSW vector indexes are untouched and not encrypted (they hold
  no plaintext beyond row IDs).
- **`mnemonics encrypt-db`** one-shot migration command. Snapshots the
  existing plain DB at `memories.db.preencrypt-<timestamp>`, builds an
  encrypted copy via `sqlcipher_export()`, verifies the row count, then
  atomically swaps the file. Aborts if any `mnemonics mcp` process is
  running, unless `--force` is passed.
- **`mnemonics.crypto`** module: `resolve_key()`, `generate_key()`,
  `store_key()`, `clear_key()`, `require_key()` for callers who want to
  manage keys programmatically.
- **`mnemonics[encrypt]`** extra: `pip install 'mnemonics[encrypt]'`
  pulls in `sqlcipher3` and `keyring`. macOS users need
  `brew install sqlcipher` first; Linux users `apt install libsqlcipher-dev`.

### Fixed

- MCP `mnemonics_ingest` now rejects empty `texts`, non-string items, and
  whitespace-only strings with an explicit error. Previously these silently
  returned `Stored 0 chunks.`, mimicking a system bug. The HTTP `/ingest`
  endpoint already had this check; this brings the two paths in line.

## [0.2.1] - 2026-05-08

### Fixed

- `pyproject.toml`: `[project.urls]` table was placed before `dependencies`, causing the TOML parser to read `dependencies` as `project.urls.dependencies` and fail the build. Moved `[project.urls]` to the end of `[project]`.
- `pyproject.toml`: removed `License :: OSI Approved :: MIT License` classifier; modern setuptools rejects it when SPDX `license = "MIT"` is set.
- Bumped `/health` and MCP `serverInfo` version strings to `0.2.1`.

No behavioral changes, packaging-only patch to enable PyPI publish.

## [0.2.0] - 2026-05-08

The "second brain" rewrite. Halluguard is gone, decay is in.

### Added

- **Tier system**: every memory has a tier that controls decay.
  - `0` pinned, never fades.
  - `1` default, 90-day half-life.
  - `2` ambient, 14-day half-life (best for low-confidence noise).
- **Decay scoring**: `score = raw_cosine × exp(-ln(2) × age_days / half_life)`.
  Apply with `--no-decay` (CLI) or `decay=false` (REST/MCP) to see raw cosine.
- **Reinforcement boost**: `boost = min(1 + log(1 + access_count) × 0.1, 2.0)`.
  Frequently retrieved rows climb in rank; the cap prevents runaway.
- **Transparent score breakdown**: every result includes `raw_score`,
  `decay_factor`, `boost`, `age_days`, `tier`. CLI prints the full
  breakdown on every line so nothing is silently demoted.
- **`mnem pin <id>`** and **`mnem tier <id> <0|1|2>`** CLI commands.
- **`mnem gc [--ns NAME] [--age-days N] [--apply]`** for sweeping
  unused ambient memories. Default is dry-run.
- **New MCP tools**: `mnemonics_pin`, `mnemonics_tier`.
- **`mnem`** short alias (same entrypoint as `mnemonics`).
- **Idempotent DB migration** (PRAGMA-driven). Older DBs upgrade
  automatically on next `Store(...)` init; existing rows inherit
  `tier=1` and zero counters.
- **Touch-on-retrieval**: `last_accessed` and `access_count` are
  bumped under the same lock as the search, in a single transactional
  UPDATE, no caller action required.
- **Privacy** section in README: `127.0.0.1` binding, no telemetry,
  HF Hub first-run notice, `TRANSFORMERS_OFFLINE`/`HF_HUB_OFFLINE`
  guidance, and an honest note that DB encryption-at-rest is not yet
  shipped.
- **Test suite**: 96 tests, 85% coverage (retrieve 100%, cli 97%,
  store 81%).

### Removed

- **Halluguard verification layer**. The daemon was silent-failing
  (port 7801 closed in practice), `verify=True` was returning
  `trust_score=1.0` regardless. The whole code path is gone, including
  the `requests` dependency, the `[verify]` and `[server]` extras, the
  `--no-verify` CLI flag, and `trust_score` / `flagged_count` /
  `verified` from the retrieve response.

### Fixed

- **`ingest()` guard against `texts="string"`**. Python iterates
  strings character-by-character, so a single string was creating one
  row per character, 672 rows in production data. Guard wraps a bare
  string in a single-item list before chunking.
- **REST and MCP ingest validation**. `POST /ingest` and
  `mnemonics_ingest` now reject string-instead-of-list with HTTP 400
  / JSON-RPC error rather than silently character-iterating.
- **`decay=False` score precision**. Now sets `score = raw_score`
  exactly instead of leaking float32 noise from the cosine score.
- **DB cleanup**: 672 corrupt single-character records purged, HNSW
  indexes rebuilt for `sessions` (1430 rows), `director` (54),
  `zeus_premortem` (6).

### Changed

- pyproject description rebrand: "Local-first AI memory, semantic
  retrieval over your own SQLite + HNSW index."
- Server bind line carries an explicit comment warning future
  maintainers (and the LLM editing this in two months) not to flip
  the host to `0.0.0.0`.
- README rewrite: pronunciation guide, tier table, decay formula,
  pin/tier/gc commands, MCP tool list, privacy section.

## [0.1.0] - earlier

Initial release. SQLite + HNSW + sentence-transformers, REST + MCP servers, namespaces.
