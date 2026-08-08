"""Entry point for the Windows background daemon subprocess.

Invoked as:  python -m lybrary._daemon_worker <root>

Writes its own PID to the pid file (overwriting the one set by the parent
just before spawn, in case they differ), then starts the file watcher.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m lybrary._daemon_worker <repo_root>", file=sys.stderr)
        sys.exit(1)

    root = Path(sys.argv[1]).resolve()

    from lybrary.config import LybraryConfig, get_lybrary_dir

    pid_file = get_lybrary_dir(root) / "daemon.pid"
    # Overwrite with our actual PID (the parent wrote its own PID first).
    pid_file.write_text(str(os.getpid()))

    config = LybraryConfig.load(root)

    print(f"lybrary daemon watching {root}", flush=True)

    from lybrary.watcher import start_watching

    start_watching(root, config)


if __name__ == "__main__":
    main()
