"""MCP server exposing lybrary tools to AI agents.

Exposes three tools:
  - memory_query    : lexical-first search over the indexed code memory
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
        "Use memory_query to locate code, then inspect the relevant source lines. "
        "If a trace crosses a named call missing from the first result, query that "
        "symbol once. Request full content only when compact excerpts are insufficient."
    ),
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_root(repo_root: str | None) -> Path:
    """Return the repository root Path, falling back to find_repo_root()."""
    if repo_root:
        p = Path(repo_root).expanduser().resolve()
    else:
        from lybrary.config import find_repo_root
        p = find_repo_root()
    if not p.is_dir():
        raise ValueError(f"repo_root is not a directory: {p}")
    if not (p / ".lybrary" / "index.db").exists():
        nearby = [child for child in p.iterdir()
                  if child.is_dir() and (child / ".lybrary" / "index.db").exists()]
        if nearby:
            names = ", ".join(str(child) for child in nearby[:3])
            raise ValueError(
                f"No Lybrary index in {p}. Indexed child: {names}. "
                "Set repo_root to that folder."
            )
    return p


def _excerpt(r: dict[str, Any], term_weights: dict[str, float]) -> str:
    """Show short source spans covering distinct query terms."""
    lines = r.get("content", "").splitlines()
    if not lines:
        return ""

    def score(i: int, covered: set[str]) -> float:
        line = lines[i].strip().lower()
        matched = sum(
            weight for term, weight in term_weights.items()
            if term not in covered and term in line
        )
        if not matched:
            return 0.0
        if line.startswith("//"):
            return matched * 0.2
        if "console.log(" in line:
            return matched * 0.02
        return matched + (0.3 if "await " in line else 0.03 if "=" in line else 0)

    anchors: list[int] = []
    covered: set[str] = set()
    for _ in range(3):
        candidates = [i for i in range(len(lines)) if all(abs(i - a) > 3 for a in anchors)]
        if not candidates:
            break
        best = max(candidates, key=lambda i: score(i, covered))
        if score(best, covered) <= 0:
            break
        anchors.append(best)
        covered.update(term for term in term_weights if term in lines[best].lower())
    if not anchors:
        anchors = [0]

    selected = sorted({
        i for a in anchors
        for i in range(a, min(a + (6 if "await " in lines[a] else 4), len(lines)))
        if lines[i].strip() and ("console.log(" not in lines[i] or i in anchors)
    })
    excerpt: list[str] = []
    previous = None
    for i in selected:
        if previous is not None and i > previous + 1:
            excerpt.append("...")
        excerpt.append(f"{r.get('start_line', 1) + i}: {lines[i].rstrip()[:140]}")
        previous = i
    return "\n".join(excerpt)


def _format_chunk(
    r: dict[str, Any], rank: int, include_content: bool = False,
    term_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return a clean, agent-friendly dict for a single search result."""
    result = {
        "rank": rank,
        "score": round(r.get("score", 0.0), 4),
        "path": r.get("path"),
        "symbol": r.get("symbol"),
        "kind": r.get("kind"),
        "start_line": r.get("start_line"),
        "end_line": r.get("end_line"),
    }
    if include_content:
        result["context_header"] = r.get("context_header")
        result["content"] = r.get("content", "")
    else:
        result["excerpt"] = _excerpt(r, term_weights or {})
    return result


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
        default=5,
        description="Maximum number of compact locations to return (1–8). Start with 5.",
        ge=1,
        le=8,
    )
    max_tokens: int = Field(
        default=2000,
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
    include_content: bool = Field(
        default=False,
        description="Include full source bodies. Defaults to compact line-numbered excerpts.",
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
    """Locate code with exact symbols, lexical search, then semantic fallback.

    Returns compact locations with line-numbered source excerpts by default.
    Follow a named call with another compact query if the first result leaves a gap.
    Set include_content for full function/class bodies when needed.

    Args:
        params (QueryInput):
            - query (str): What to search for (e.g. "JWT token validation")
            - limit (int): Max chunks to return, 1–8 (default 5)
            - max_tokens (int): Token budget when include_content is true (default 2000)
            - repo_root (str | None): Repo path, or None to auto-detect
            - include_content (bool): Include full source (default false)

    Returns:
        str: JSON with schema:
        {
            "query": str,
            "results_count": int,
            "chunks": [
                {
                    "rank": int,
                    "score": float,       # relevance 0–1
                    "path": str,          # relative file path
                    "symbol": str,        # function/class name
                    "kind": str,          # e.g. "function_definition"
                    "start_line": int,
                    "end_line": int,
                    "excerpt": str       # compact mode; relevant numbered source lines
                    # full mode instead adds context_header and content
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
        from lybrary.query import query_terms, search

        raw = search(
            root, params.query, limit=params.limit,
            max_tokens=params.max_tokens if params.include_content else None,
        )
        terms = query_terms(params.query)
        term_weights = {
            term: 1 / max(1, sum(term in r.get("content", "").lower() for r in raw)) ** 2
            for term in terms
        }
        chunks = [
            _format_chunk(r, i + 1, params.include_content, term_weights)
            for i, r in enumerate(raw)
        ]
        return json.dumps(
            {"query": params.query, "results_count": len(chunks), "chunks": chunks},
            separators=(",", ":"),
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
