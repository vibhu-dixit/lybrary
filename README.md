# Lybrary — local codebase memory for coding agents

Lybrary indexes a repository by code structure and gives Codex, Claude Code, and other MCP clients a searchable map of its functions, classes, and files. It runs locally, watches for changes, and lets an agent find relevant source lines before opening whole files.

<!-- mcp-name: io.github.vibhu-dixit/lybrary -->

[PyPI](https://pypi.org/project/lybrary/) · [Documentation](https://vibhu-dixit.github.io/lybrary/) · [Agent setup guide](https://vibhu-dixit.github.io/lybrary/agent-setup.html) · [Source](https://github.com/vibhu-dixit/lybrary) · [Issues](https://github.com/vibhu-dixit/lybrary/issues)

If you want a coding agent to set this up, ask it to **find and install a local MCP code search tool for the repository** and point it to the [agent setup guide](https://vibhu-dixit.github.io/lybrary/agent-setup.html). The guide identifies the PyPI package, source repository, install command, and checks that confirm the agent is using this repo's index.

## Start in any repository

Install once, then run `lybrary start` in each folder you want indexed:

```bash
pip install lybrary
cd /path/to/your/repo
lybrary start
lybrary status
```

`start` creates `.lybrary/` if needed, builds the index, and runs a file-watching daemon in the background. It keeps running after the terminal closes. Run `lybrary stop` in that repository to stop it. The first indexing run may download the local embedding model; later runs use the cache.

Each repository has its own index and `.lybrary/config.toml`. The index stays on your machine; no API key or hosted database is required.

## Connect an agent

Register the MCP server once on your machine. Start your agent in an indexed repository, then ask it to call `memory_status` to confirm the root and chunk count.

### Codex

Codex users can install Lybrary's setup skill from the repository's plugin marketplace:

```bash
codex plugin marketplace add vibhu-dixit/lybrary
codex plugin add lybrary@lybrary
```

The plugin helps with setup; register the local MCP server separately:

```bash
codex mcp add lybrary -- lybrary mcp
codex mcp list
```

Add this to that repository's `AGENTS.md` if you want Codex to use the memory consistently:

```markdown
When .lybrary/index.db exists, use Lybrary memory_query before broad code searches or reading multiple files. Inspect the returned source paths and lines before editing. If the index is absent or the tool fails, use normal search.
```

### Claude Code

Claude Code users can install the repository's plugin, which registers the MCP server and includes a setup skill:

```bash
claude plugin marketplace add vibhu-dixit/lybrary
claude plugin install lybrary@lybrary
```

The Python package must also be installed with `pip install lybrary`. Or register the MCP server directly without the plugin:

```bash
claude mcp add --scope user --transport stdio lybrary -- lybrary mcp
claude mcp list
```

In Claude Code, `/mcp` shows connection status. Put the same instruction in the repository's `CLAUDE.md`. Registration makes the tool available; the instruction helps the agent choose it when useful.

For another stdio MCP client, configure command `lybrary` with argument `mcp`. See the [setup guides](https://vibhu-dixit.github.io/lybrary/).

## What the agent gets

`memory_query` checks exact symbols and an SQLite FTS5 index first. Strong lexical matches avoid loading the embedding model; semantic search remains a fallback when wording does not match the code. Its default response gives ranked paths, symbols, line ranges, and compact numbered excerpts. If a trace crosses a named call missing from the first result, query that symbol once. Set `include_content=true` when a full chunk is needed.

`memory_status` reports the active repository and index health. `memory_update` triggers incremental, full, or path-scoped re-indexing.

This can avoid repeated broad searches and large file reads, especially when the agent needs to locate a feature across sessions. Actual token use depends on the task, the agent's tool choices, and how much source it still needs to inspect. The [benchmark guide](https://vibhu-dixit.github.io/lybrary/token-usage.html) explains how to compare complete runs rather than infer savings from the size of one response.

## Make it fit your repo

Edit `.lybrary/config.toml` to choose the embedding model and device, chunk sizes, indexed languages, excluded paths, and file-watcher debounce. Add Git-style ignore patterns to `.lybrary/ignore`. Re-run `lybrary index --full` after changing settings that affect stored chunks or embeddings.

Supported parsers: Python, JavaScript, TypeScript, TSX, Go, Rust, Java, C, and C++. The indexer uses tree-sitter boundaries and refreshes changed files while the daemon is running.

## CLI

| Command | Purpose |
| --- | --- |
| `lybrary start` | Initialize, index, and watch the current repository |
| `lybrary stop` | Stop its background daemon |
| `lybrary status` | Show daemon and index status |
| `lybrary query "authentication flow"` | Search from the terminal |
| `lybrary index --full` | Rebuild the index |
| `lybrary logs` | Show daemon logs |
| `lybrary mcp` | Run the stdio MCP server |

If an agent starts above the indexed repository, pass that repository's absolute path as `repo_root` to the MCP tool, or start the agent inside it.

## Measure the effect

From a source checkout, compare the same read-only question with and without Lybrary:

```bash
python benchmark.py /path/to/indexed/repo "Trace how interview invitations are sent"
```

The script records both answers, tool calls, and total, cached, and uncached input tokens under `.lybrary/benchmarks/`. Check answer correctness first; one question does not establish typical savings.

## Contribute

Issues and pull requests are welcome. For a local checkout:

```bash
pip install -e ".[dev]"
pytest
```

Lybrary is MIT licensed. See the [changelog](CHANGELOG.md) for releases.
