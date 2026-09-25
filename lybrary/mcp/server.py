"""MCP server exposing lybrary tools to AI agents.

Exposes three tools:
  - memory_query    : semantic search over the indexed code memory
  - memory_status   : daemon/index health check
  - memory_update   : trigger incremental (or full) re-index

Run via:  lybrary mcp  (uses stdio transport — works with any MCP host)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, field_validator

# NumPy's first import can block inside an MCP request on Windows; load it at startup.
from lybrary.indexer.store import IndexStore

# ---------------------------------------------------------------------------
# Server init
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "lybrary_mcp",
    instructions=(
        "lybrary gives AI agents low-token access to a codebase's structure. "
        "Always call memory_query before reading raw files. "
        "Returned chunks include file path, line numbers, and the full symbol body."
    ),
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_root(repo_root: str | None) -> Path:
    """Return the repository root Path, falling back to find_repo_root()."""
    if repo_root:
        p = Path(repo_root).expanduser().resolve()
        if not p.exists():
            raise ValueError(f"repo_root does not exist: {p}")
        return p
    from lybrary.config import find_repo_root
    return find_repo_root()


def _format_chunk(r: dict[str, Any], rank: int) -> dict[str, Any]:
    """Return a clean, agent-friendly dict for a single search result."""
    return {
        "rank": rank,
        "score": round(r.get("score", 0.0), 4),
        "path": r.get("path"),
        "symbol": r.get("symbol"),
        "kind": r.get("kind"),
        "start_line": r.get("start_line"),
        "end_line": r.get("end_line"),
        "context_header": r.get("context_header"),
        "content": r.get("content", ""),
    }


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------


class QueryInput(BaseModel):
    """Input for memory_query."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    query: str = Field(
        ...,
        description="Natural-language or keyword description of what you want to find. "
        "E.g. 'authentication middleware', 'database connection pool', 'retry logic'.",
        min_length=1,
        max_length=500,
    )
    limit: int = Field(
        default=8,
        description="Maximum number of chunks to return (1–20).",
        ge=1,
        le=20,
    )
    max_tokens: int = Field(
        default=6000,
        description="Approximate token budget for returned content. "
        "Results are packed until this budget is reached.",
        ge=100,
        le=32000,
    )
    repo_root: str | None = Field(
        default=None,
        description="Absolute path to the repository root. "
        "Omit to auto-detect from the current working directory.",
    )

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class StatusInput(BaseModel):
    """Input for memory_status."""

    model_config = ConfigDict(str_strip_whitespace=True)

    repo_root: str | None = Field(
        default=None,
        description="Absolute path to the repository root. Omit to auto-detect.",
    )


class UpdateInput(BaseModel):
    """Input for memory_update."""

    model_config = ConfigDict(str_strip_whitespace=True)

    repo_root: str | None = Field(
        default=None,
        description="Absolute path to the repository root. Omit to auto-detect.",
    )
    full: bool = Field(
        default=False,
        description="When true, drop and rebuild the entire index. "
        "When false (default), only re-index changed files.",
    )
    paths: list[str] | None = Field(
        default=None,
        description="Optional list of specific relative file paths to re-index. "
        "When provided, only these files are updated regardless of 'full'.",
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool(
    name="memory_query",
    annotations={
        "title": "Search Code Memory",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def memory_query(params: QueryInput) -> str:
    """Semantic search over the lybrary code memory index.

    Returns the most relevant code chunks for the given query, packed within
    the token budget. Each chunk contains the full function/class body, its
    file path, line numbers, and a context header.

    ALWAYS call this before reading raw files. It drastically reduces token
    consumption by surfacing only the relevant code units.

    Args:
        params (QueryInput):
            - query (str): What to search for (e.g. "JWT token validation")
            - limit (int): Max chunks to return, 1–20 (default 8)
            - max_tokens (int): Token budget for content (default 6000)
            - repo_root (str | None): Repo path, or None to auto-detect

    Returns:
        str: JSON with schema:
        {
            "query": str,
            "results_count": int,
            "chunks": [
                {
                    "rank": int,
                    "score": float,       # cosine similarity 0–1
                    "path": str,          # relative file path
                    "symbol": str,        # function/class name
                    "kind": str,          # e.g. "function_definition"
                    "start_line": int,
                    "end_line": int,
                    "context_header": str,
                    "content": str        # full source of the symbol
                }
            ]
        }

        On error: {"error": "<message>"}

    Examples:
        - "Find authentication logic" → query="authentication login"
        - "Where is the database connection set up?" → query="database connection init"
        - "Show me all retry/backoff code" → query="retry backoff exponential"
    """
    try:
        root = _resolve_root(params.repo_root)
        from lybrary.query import search

        raw = search(root, params.query, limit=params.limit, max_tokens=params.max_tokens)
        chunks = [_format_chunk(r, i + 1) for i, r in enumerate(raw)]
        return json.dumps(
            {"query": params.query, "results_count": len(chunks), "chunks": chunks},
            indent=2,
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="memory_status",
    annotations={
        "title": "Check Memory Index Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def memory_status(params: StatusInput) -> str:
    """Check the health and statistics of the lybrary memory index.

    Returns the daemon running state, total chunk count, number of tracked
    files, and the data directory path.

    Use this to verify the index is populated before querying, or to
    diagnose why queries return no results.

    Args:
        params (StatusInput):
            - repo_root (str | None): Repo path, or None to auto-detect

    Returns:
        str: JSON with schema:
        {
            "running": bool,         # true if background daemon is active
            "pid": int | null,       # daemon process ID
            "chunks": int,           # total indexed chunks
            "tracked_files": int,    # files with stored hashes
            "root": str,             # resolved repo root path
            "data_dir": str,         # .lybrary/ directory path
            "log": str | null        # path to daemon log file
        }

        On error: {"error": "<message>"}
    """
    try:
        root = _resolve_root(params.repo_root)
        from lybrary.daemon import daemon_status

        info = daemon_status(root)
        return json.dumps(info, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(
    name="memory_update",
    annotations={
        "title": "Update Code Memory Index",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def memory_update(params: UpdateInput) -> str:
    """Trigger a re-index of the repository (or specific files).

    The background daemon handles file changes automatically. Use this tool
    when you want to force an immediate update after bulk edits, or when the
    daemon is not running.

    Args:
        params (UpdateInput):
            - repo_root (str | None): Repo path, or None to auto-detect
            - full (bool): True = full re-index; False = incremental (default)
            - paths (list[str] | None): Specific relative paths to re-index.
              When set, only these files are processed.

    Returns:
        str: JSON with schema:
        {
            "status": "ok",
            "files_scanned": int,
            "chunks_total": int,
            "files_updated": int
        }

        On error: {"error": "<message>"}
    """
    store = None
    try:
        root = _resolve_root(params.repo_root)

        if params.paths:
            for rel in params.paths:
                try:
                    (root / rel).resolve().relative_to(root.resolve())
                except ValueError as e:
                    raise ValueError(f"path outside repository: {rel}") from e

        from lybrary.config import LybraryConfig
        config = LybraryConfig.load(root)
        store = IndexStore(root)

        if params.paths:
            # Targeted update: re-index only the listed files
            from lybrary.indexer.ast_chunker import chunk_file
            from lybrary.indexer.embedder import Embedder

            embedder = Embedder(config.embed.model, config.embed.device)
            updated = 0
            for rel in params.paths:
                path = root / rel
                if not path.exists():
                    store.delete_path(rel)
                    hashes = store.load_hashes()
                    hashes.pop(rel, None)
                    store.save_hashes(hashes)
                    continue
                chunks = chunk_file(
                    path,
                    root,
                    soft_max_tokens=config.chunk.soft_max_tokens,
                    min_chunk_tokens=config.chunk.min_chunk_tokens,
                )
                if chunks:
                    texts = [c.embedding_text() for c in chunks]
                    vectors = embedder.embed(texts, batch_size=config.embed.batch_size)
                    store.upsert_chunks(chunks, vectors)
                    from lybrary.utils import content_hash
                    hashes = store.load_hashes()
                    hashes[rel] = content_hash(path.read_bytes())
                    store.save_hashes(hashes)
                    updated += 1

            return json.dumps(
                {
                    "status": "ok",
                    "files_scanned": len(params.paths),
                    "chunks_total": store.count(),
                    "files_updated": updated,
                }
            )

        # Full or incremental repository scan
        from lybrary.indexer.walker import index_repository

        stats = index_repository(root, store, config, full=params.full)
        return json.dumps(
            {
                "status": "ok",
                "files_scanned": stats.get("files", 0),
                "chunks_total": stats.get("chunks", 0),
                "files_updated": stats.get("updated", 0),
            }
        )

    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        if store is not None:
            store.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Run the MCP server over stdio (default transport for IDE integrations)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
