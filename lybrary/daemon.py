"""Background daemon: watches files and keeps the index fresh."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from lybrary.config import LybraryConfig, get_lybrary_dir

console = Console()


def _pid_file(root: Path) -> Path:
    return get_lybrary_dir(root) / "daemon.pid"


def _log_file(root: Path) -> Path:
    return get_lybrary_dir(root) / "daemon.log"


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def daemon_status(root: Path) -> dict[str, Any]:
    ly_dir = get_lybrary_dir(root)
    pid_path = _pid_file(root)
    info: dict[str, Any] = {
        "root": str(root),
        "data_dir": str(ly_dir),
        "running": False,
        "pid": None,
    }

    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            info["pid"] = pid
            info["running"] = _is_running(pid)
        except Exception:
            pass

    # Index stats
    try:
        from lybrary.indexer.store import IndexStore

        store = IndexStore(root)
        info["chunks"] = store.count()
        hashes = store.load_hashes()
        info["tracked_files"] = len(hashes)
    except Exception as e:
        info["index_error"] = str(e)

    log = _log_file(root)
    if log.exists():
        info["log"] = str(log)

    return info


def stop_daemon(root: Path) -> None:
    pid_path = _pid_file(root)
    if not pid_path.exists():
        console.print("[yellow]No daemon pid file found.[/yellow]")
        return

    try:
        pid = int(pid_path.read_text().strip())
    except Exception:
        console.print("[red]Invalid pid file.[/red]")
        pid_path.unlink(missing_ok=True)
        return

    if not _is_running(pid):
        console.print("[yellow]Daemon not running (stale pid file).[/yellow]")
        pid_path.unlink(missing_ok=True)
        return

    if os.name == "nt":
        # Windows: os.kill with signal.SIGTERM is not reliable for subprocesses;
        # use taskkill to terminate the whole process tree.
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                check=False,
                capture_output=True,
            )
        except Exception:
            pass
    else:
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            time.sleep(0.2)
            if not _is_running(pid):
                break
        else:
            os.kill(pid, signal.SIGKILL)

    pid_path.unlink(missing_ok=True)
    console.print(f"[green]Daemon stopped[/green] (pid {pid})")


def start_daemon(root: Path, foreground: bool = False) -> None:
    status = daemon_status(root)
    if status["running"]:
        console.print(f"[green]Daemon already running[/green] (pid {status['pid']})")
        return

    # Ensure index exists
    from lybrary.indexer.store import IndexStore
    from lybrary.indexer.walker import index_repository

    config = LybraryConfig.load(root)
    store = IndexStore(root)
    if store.count() == 0:
        console.print("[bold]First run — building index...[/bold]")
        stats = index_repository(root, store, config, full=True)
        console.print(f"[green]Indexed {stats.get('chunks', 0)} chunks[/green]")

    if foreground:
        console.print("[bold]Running daemon in foreground (Ctrl+C to stop)[/bold]")
        _run_watcher(root, config)
        return

    if os.name == "nt":
        # Fix #5: Windows — spawn a detached subprocess and write its PID so
        # that `status` and `stop` work correctly.
        _start_daemon_windows(root, config)
        return

    # Unix — double-fork daemonisation
    pid = os.fork()
    if pid > 0:
        # Parent: write PID and return immediately.
        _pid_file(root).write_text(str(pid))
        console.print(f"[green]Daemon started[/green] (pid {pid})")
        console.print(f"Logs: {_log_file(root)}")
        console.print("Agents can now use lybrary memory.")
        return

    # Child — become daemon
    os.setsid()
    # Redirect stdio to log
    log = open(_log_file(root), "a", buffering=1)
    sys.stdout = log  # type: ignore
    sys.stderr = log  # type: ignore
    # Write PID from child so the file reflects the actual daemon process.
    _pid_file(root).write_text(str(os.getpid()))
    try:
        _run_watcher(root, config)
    except Exception as e:
        print(f"Daemon crashed: {e}", flush=True)
        raise


def _start_daemon_windows(root: Path, config: LybraryConfig) -> None:
    """Fix #5: Launch a detached background process on Windows and write its PID."""
    log_path = _log_file(root)
    log_file = open(log_path, "a")  # noqa: WPS515 – kept open until proc exits

    # Re-invoke lybrary with the internal _watch subcommand so the child
    # process starts the watcher directly without re-entering start_daemon.
    proc = subprocess.Popen(
        [sys.executable, "-m", "lybrary._daemon_worker", str(root)],
        stdout=log_file,
        stderr=log_file,
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        creationflags=0x00000008 | 0x00000200,
        close_fds=True,
    )

    pid = proc.pid
    _pid_file(root).write_text(str(pid))
    console.print(f"[green]Daemon started[/green] (pid {pid})")
    console.print(f"Logs: {log_path}")
    console.print("Agents can now use lybrary memory.")


def _run_watcher(root: Path, config: LybraryConfig) -> None:
    from lybrary.watcher import start_watching

    print(f"lybrary daemon watching {root}", flush=True)
    start_watching(root, config)
