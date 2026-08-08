"""lybrary CLI — living code memory for AI agents."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from lybrary import __version__
from lybrary.config import LybraryConfig, find_repo_root, get_lybrary_dir

app = typer.Typer(
    name="lybrary",
    help="Living structure-aware code memory for AI coding agents.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def _root(path: Optional[Path] = None) -> Path:
    return find_repo_root(path)


@app.command()
def init(
    path: Optional[Path] = typer.Argument(None, help="Repository path (default: cwd)"),
) -> None:
    """Initialize lybrary in a repository."""
    root = _root(path)
    ly_dir = get_lybrary_dir(root)
    ly_dir.mkdir(parents=True, exist_ok=True)

    config = LybraryConfig()
    config.save(root)

    # Create a basic ignore file if missing
    ignore = ly_dir / "ignore"
    if not ignore.exists():
        ignore.write_text(
            "\n".join(
                [
                    "# Extra ignore patterns (gitignore syntax)",
                    "*.min.js",
                    "*.map",
                    "dist/",
                    "build/",
                    "",
                ]
            )
        )

    console.print(
        Panel.fit(
            f"[bold green]lybrary initialized[/bold green]\n\n"
            f"Root: [cyan]{root}[/cyan]\n"
            f"Data: [cyan]{ly_dir}[/cyan]\n\n"
            f"Next: [bold]lybrary start[/bold]",
            title="lybrary",
        )
    )


@app.command()
def start(
    path: Optional[Path] = typer.Argument(None, help="Repository path"),
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground"),
) -> None:
    """Start the lybrary daemon (indexes if needed, then watches for changes)."""
    root = _root(path)
    ly_dir = get_lybrary_dir(root)

    if not ly_dir.exists():
        console.print("[yellow]No .lybrary/ found. Running init first...[/yellow]")
        init(path)

    from lybrary.daemon import start_daemon

    start_daemon(root, foreground=foreground)


@app.command()
def stop(
    path: Optional[Path] = typer.Argument(None, help="Repository path"),
) -> None:
    """Stop the running lybrary daemon."""
    root = _root(path)
    from lybrary.daemon import stop_daemon

    stop_daemon(root)


@app.command()
def status(
    path: Optional[Path] = typer.Argument(None, help="Repository path"),
) -> None:
    """Show daemon and index status."""
    root = _root(path)
    from lybrary.daemon import daemon_status

    info = daemon_status(root)

    table = Table(title="lybrary status", show_header=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")

    for k, v in info.items():
        table.add_row(k, str(v))

    console.print(table)


@app.command()
def index(
    path: Optional[Path] = typer.Argument(None, help="Repository path"),
    full: bool = typer.Option(False, "--full", help="Force full re-index"),
) -> None:
    """Index (or re-index) the repository."""
    root = _root(path)
    ly_dir = get_lybrary_dir(root)
    if not ly_dir.exists():
        init(path)

    from lybrary.indexer.store import IndexStore
    from lybrary.indexer.walker import index_repository

    config = LybraryConfig.load(root)
    store = IndexStore(root)

    console.print(f"[bold]Indexing[/bold] [cyan]{root}[/cyan] ...")
    stats = index_repository(root, store, config, full=full)

    console.print(
        Panel.fit(
            f"[green]Index complete[/green]\n\n"
            f"Files: {stats.get('files', 0)}\n"
            f"Chunks: {stats.get('chunks', 0)}\n"
            f"Added/updated: {stats.get('updated', 0)}",
            title="lybrary index",
        )
    )


@app.command()
def query(
    q: str = typer.Argument(..., help="Search query"),
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Repository path"),
    limit: int = typer.Option(8, "--limit", "-n", help="Max results"),
    max_tokens: int = typer.Option(6000, "--max-tokens", help="Token budget for context"),
) -> None:
    """Search the code memory."""
    root = _root(path)
    from lybrary.query import search

    results = search(root, q, limit=limit, max_tokens=max_tokens)

    if not results:
        console.print("[yellow]No results.[/yellow]")
        return

    for i, r in enumerate(results, 1):
        console.print(
            Panel(
                f"[dim]{r.get('path')} :: {r.get('symbol', '')}[/dim]\n\n"
                f"{r.get('content', '')[:800]}{'...' if len(r.get('content', '')) > 800 else ''}",
                title=f"[{i}] {r.get('score', 0):.3f}",
                border_style="blue",
            )
        )


@app.command()
def logs(
    path: Optional[Path] = typer.Argument(None, help="Repository path"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
) -> None:
    """Show daemon logs."""
    root = _root(path)
    log_file = get_lybrary_dir(root) / "daemon.log"
    if not log_file.exists():
        console.print("[yellow]No log file yet.[/yellow]")
        raise typer.Exit(1)

    if follow:
        import time

        with open(log_file) as f:
            f.seek(0, 2)
            while True:
                line = f.readline()
                if line:
                    console.print(line.rstrip())
                else:
                    time.sleep(0.3)
    else:
        console.print(log_file.read_text()[-8000:])


@app.command()
def mcp(
    repo_root: Optional[Path] = typer.Option(
        None, "--root", "-r", help="Repository root (default: auto-detect)"
    ),
) -> None:
    """Start the MCP server (stdio transport) for AI IDE integration.

    Configure your IDE / MCP host to run:  lybrary mcp

    Kiro / Cursor / Claude Desktop example config:
      {
        "mcpServers": {
          "lybrary": {
            "command": "lybrary",
            "args": ["mcp"]
          }
        }
      }
    """
    try:
        from lybrary.mcp.server import mcp as _mcp_server  # noqa: F401
    except ImportError:
        console.print(
            "[red]MCP dependencies not installed.[/red]\n"
            "Run: [bold]pip install 'lybrary[mcp]'[/bold]"
        )
        raise typer.Exit(1)

    if repo_root:
        import os
        os.chdir(repo_root)

    from lybrary.mcp.server import run
    run()


@app.command()
def version() -> None:
    """Show version."""
    console.print(f"lybrary {__version__}")


if __name__ == "__main__":
    app()
