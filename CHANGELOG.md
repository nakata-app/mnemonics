# Changelog

All notable changes to mnemonics. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow [SemVer](https://semver.org/spec/v2.0.0.html).

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
