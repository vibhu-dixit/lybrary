# Changelog

All notable changes to lybrary are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

---

## [0.3.2] — 2026-10-12

### Added
- An agent-facing setup guide that identifies the published package and verifies a per-repository install.
- Codex and Claude Code plugin marketplaces with a small setup skill for agents asked to enable local codebase memory.

### Changed
- Clarified discovery metadata and homepage wording for repository code search and coding agents.

---

## [0.3.1] — 2026-10-12

### Changed
- Search indexed chunks with SQLite FTS5 first; load the embedding model only when lexical matches are weak.
- Preserve relevant results from the same file and guide agents to follow a named call with a second compact query when needed.
- Upgrade existing indexes to the FTS5 format when the writer opens them; old indexes remain readable before migration.

---

## [0.3.0] — 2026-10-12

### Added
- Blended SQLite text and vector search for conceptual code queries, with exact symbol lookup.
- Compact line-numbered MCP excerpts that show separate relevant source steps.
- Codex and Claude Code setup guides, configurable-index documentation, and MCP Registry metadata.
- A paired Codex benchmark script that records answers, tool calls, and token use.

### Changed
- Kept default MCP responses compact while allowing full source when requested.
- Updated public package metadata and documentation to describe usage without an unverified savings claim.

---

## [0.2.0] — 2026-09-24

### Added
- Documented one-time Codex and Claude Code MCP registration and per-folder `lybrary start` setup.
- Added tests for MCP path handling, daemon status, and incremental indexing.

### Fixed
- Corrected Windows daemon process detection and stop error reporting.
- Opened index stores read-only for status and queries, and closed MCP update stores reliably.
- Rejected MCP update paths outside the repository.
- Fixed watcher handling of ignored files, empty chunks, and file changes during indexing.
- Corrected query scoring and bounded the amount of intermediate search content.

---

## [0.1.0] — 2026-08-07

### Added
- AST-aware chunking (cAST-style) via tree-sitter for Python, JavaScript,
  TypeScript, TSX, Go, Rust, Java, C, and C++.
- Incremental indexing via xxhash content-hash tracking (`file_hashes.json`).
- LanceDB-backed vector store with MiniLM-L6-v2 embeddings
  (sentence-transformers, fully local).
- Background daemon with watchdog file watcher and configurable debounce.
- CLI commands: `init`, `start`, `stop`, `status`, `index`, `query`, `logs`,
  `mcp`, `version`.
- MCP server (`lybrary mcp`) exposing three tools over stdio transport:
  - `memory_query` — semantic search over the indexed code memory.
  - `memory_status` — daemon/index health check.
  - `memory_update` — incremental or full re-index, optionally scoped to
    specific files.
- `py.typed` marker for PEP 561 compliance.

### Fixed
- TSX files now use the tsx tree-sitter grammar instead of being silently
  downgraded to the TypeScript grammar.
- Nested definitions (methods inside classes) are no longer emitted twice.
- Overlapping byte ranges are properly merged before computing leftover
  module-level content.
- Chunk IDs include the byte offset, preventing collisions for same-named
  symbols at the same line.
- `_sanitize_name` strips characters that would corrupt chunk IDs or
  LanceDB filter strings.
- `delete_path` escapes double-quotes in paths to prevent filter injection.
- `search_vector` no longer returns the raw 384-float vector column to callers.
- Full re-index uses `truncate()` (drop + recreate) instead of iterating
  `all_paths()`, eliminating stale row leakage.
- File-move/rename events are handled correctly: the new path is indexed and
  the old path is deleted from the store.
- `_update_file` reads the file exactly once, eliminating the race condition
  between chunk generation and hash storage.
- Windows daemon now spawns a detached subprocess and writes a PID file so
  that `lybrary stop` and `lybrary status` work on Windows.
- Score formula uses proper cosine similarity (`1 - d²/2`) for unit-normalised
  vectors instead of the previous `1/(1+d)` approximation.
- Unused `ThreadPoolExecutor` import removed from `walker.py`.
- Dead `merge_siblings` config option removed from `ChunkConfig`.
- Embedder normalises the device string before using it as an `lru_cache` key.
