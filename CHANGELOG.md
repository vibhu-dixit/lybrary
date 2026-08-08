# Changelog

All notable changes to lybrary are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

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
