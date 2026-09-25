import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from lybrary.daemon import _is_running, stop_daemon


def test_current_process_is_running():
    assert _is_running(os.getpid())


@pytest.mark.skipif(os.name != "nt", reason="Windows detached process behavior")
def test_detached_child_is_running():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        creationflags=0x00000008 | 0x00000200,
    )
    try:
        assert _is_running(child.pid)
    finally:
        child.terminate()
        child.wait(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Windows taskkill behavior")
def test_failed_stop_keeps_pid_file(tmp_path):
    pid_file = tmp_path / ".lybrary" / "daemon.pid"
    pid_file.parent.mkdir()
    pid_file.write_text("12345")
    with (
        patch("lybrary.daemon._is_running", return_value=True),
        patch("lybrary.daemon.subprocess.run", return_value=subprocess.CompletedProcess([], 1)),
    ):
        stop_daemon(tmp_path)
    assert pid_file.exists()
