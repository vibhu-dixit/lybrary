<div align="center">

# 🧠 lybrary

**Living structure-aware code memory for AI coding agents.**

[![PyPI version](https://img.shields.io/pypi/v/lybrary?color=blue)](https://pypi.org/project/lybrary/)
[![Python](https://img.shields.io/pypi/pyversions/lybrary)](https://pypi.org/project/lybrary/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-86%20passed-brightgreen)](tests/)

```bash
pip install lybrary
```

</div>

---

## The problem

Every time an AI agent starts a session, it re-reads your codebase from scratch.

```
agent: let me grep for auth logic...
agent: reading src/auth/middleware.py...
agent: reading src/auth/jwt.py...
agent: reading src/auth/session.py...
agent: reading src/utils/crypto.py...
↳ 4,000 tokens burned before writing a single line.
```

On a large codebase this happens dozens of times per session. Tokens wasted. Context filled. Same files read over and over.

---

## The solution

lybrary gives your agent a **persistent memory** it can query instead of reading files.

```
agent: memory_query("authentication flow")
↳ 3 chunks returned. 180 tokens. Done.
```

It indexes your repo using real AST boundaries, keeps the index fresh automatically, and exposes it as an **MCP server** that any AI IDE connects to natively.

---

## ✨ Features

| | |
|---|---|
| 🌳 **AST-aware chunking** | tree-sitter parses your code — never splits a function in half |
| ⚡ **Background daemon** | watches for file changes, re-indexes only what changed |
| 🔍 **Semantic search** | vector search with token-budget packing |
| 🔌 **MCP server** | works with Kiro, Cursor, Claude Desktop, Windsurf out of the box |
| 📦 **Fully local** | no cloud, no API keys, embeddings run on your machine |
| 🐍 **Pure pip install** | Python 3.11–3.14, no PyTorch, no compilation needed |

**Supported languages:** Python · JavaScript · TypeScript · TSX · Go · Rust · Java · C · C++

---

## 🚀 Quick start

```bash
pip install lybrary

cd /path/to/your/repo
lybrary init
lybrary start        # builds index + starts background daemon
lybrary query "authentication flow"
```

After `lybrary start`, the daemon keeps running even after you close the terminal. File changes are picked up automatically — only affected chunks are re-indexed.

---

## 🔌 MCP integration

Add to your MCP config (Kiro, Cursor, Claude Desktop, Windsurf):

```json
{
  "mcpServers": {
    "lybrary": {
      "command": "lybrary",
      "args": ["mcp"]
    }
  }
}
```

Your agent now has three tools:

| Tool | What it does |
|---|---|
| `memory_query` | Semantic search — returns ranked chunks with full source, file path, and line numbers |
| `memory_status` | Reports daemon state, chunk count, and tracked files |
| `memory_update` | Triggers incremental or full re-index, optionally scoped to specific files |

> **Agents should call `memory_query` before reading any files.**
> This replaces multi-file reads with a single targeted query — cutting token usage by 80–90% on large codebases.

---

## 🖥️ CLI reference

| Command | Description |
|---|---|
| `lybrary init` | Create `.lybrary/` and default config |
| `lybrary start` | Index (if needed) + start persistent daemon |
| `lybrary stop` | Stop the daemon |
| `lybrary status` | Show running state, chunk count, tracked files |
| `lybrary index` | Force (re)index |
| `lybrary query` | Semantic search over the memory |
| `lybrary logs` | View / follow daemon log |
| `lybrary mcp` | Start MCP server (stdio transport) |

---

## 🏗️ How chunking works

```
your file
    │
    ▼
tree-sitter parser
    │
    ▼
AST definition nodes          ← functions, classes, methods, interfaces
    │
    ├── class Foo  ──────────► chunk: entire class body
    │     ├── def bar ───────► chunk: method bar (its own chunk too)
    │     └── def baz ───────► chunk: method baz (its own chunk too)
    │
    └── module-level ────────► chunk: imports, constants, top-level statements
```

Each chunk gets a context header and is embedded with **MiniLM-L6-v2 via ONNX Runtime** — fast, local, no GPU needed.

---

## 🗂️ Architecture

```
.lybrary/
├── config.toml          # model, chunk size, ignore patterns
├── index.db             # SQLite: chunks + float32 vector blobs
├── file_hashes.json     # content-hash map for incremental updates
├── daemon.pid
└── daemon.log
```

- **Indexer** — tree-sitter → AST chunks → fastembed / ONNX Runtime embeddings
- **Store** — SQLite + numpy (cosine similarity via batched dot product, no external vector DB)
- **Daemon** — watchdog file watcher + debounce + incremental re-chunk/embed
- **MCP** — FastMCP server over stdio

---

## 🗺️ Roadmap

- [x] AST chunker (multi-language, cAST-style)
- [x] Incremental indexing via content hashes
- [x] Background daemon + file watcher (Windows + Unix)
- [x] CLI (`init` / `start` / `stop` / `status` / `index` / `query` / `logs` / `mcp`)
- [x] MCP server (`memory_query`, `memory_status`, `memory_update`)
- [ ] Call/import graph expansion
- [ ] Hierarchical file/package summaries
- [ ] Cross-session decision memory
- [ ] systemd / launchd user service helper

---

## 🤝 Contributing

Issues and PRs welcome. Run the test suite with:

```bash
pip install -e ".[dev]"
pytest
```

---

## 📄 License

MIT
