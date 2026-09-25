"""Tests for lybrary.watcher.LybraryHandler — no real filesystem observer needed."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from lybrary.config import LybraryConfig
from lybrary.utils import content_hash
from lybrary.watcher import LybraryHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(tmp_path: Path) -> LybraryHandler:
    config = LybraryConfig()
    config.daemon.debounce_ms = 0  # instant debounce for tests
    store = MagicMock()
    store.load_hashes.return_value = {}
    embedder = MagicMock()
    embedder.embed.return_value = [[0.0] * 384]
    return LybraryHandler(tmp_path, config, store, embedder)


def _event(src_path: str, is_directory: bool = False):
    evt = MagicMock()
    evt.src_path = src_path
    evt.is_directory = is_directory
    # Not a move event by default
    from watchdog.events import FileMovedEvent
    evt.__class__ = type("FileModifiedEvent", (), {})
    return evt


def _move_event(src_path: str, dest_path: str):
    from watchdog.events import FileMovedEvent
    evt = MagicMock(spec=FileMovedEvent)
    evt.src_path = src_path
    evt.dest_path = dest_path
    evt.is_directory = False
    return evt


# ---------------------------------------------------------------------------
# Queuing behaviour
# ---------------------------------------------------------------------------


def test_on_any_event_queues_source_file(tmp_path):
    handler = _make_handler(tmp_path)
    f = tmp_path / "src.py"
    f.write_text("x = 1")
    evt = _event(str(f))
    handler.on_any_event(evt)
    assert len(handler._pending) == 1


def test_on_any_event_ignores_directories(tmp_path):
    handler = _make_handler(tmp_path)
    evt = _event(str(tmp_path / "some_dir"), is_directory=True)
    handler.on_any_event(evt)
    assert handler._pending == {}


def test_on_any_event_ignores_unknown_extension(tmp_path):
    handler = _make_handler(tmp_path)
    f = tmp_path / "README.md"
    f.write_text("# hi")
    evt = _event(str(f))
    handler.on_any_event(evt)
    assert handler._pending == {}


def test_on_any_event_ignores_lybrary_dir(tmp_path):
    handler = _make_handler(tmp_path)
    lybrary_file = tmp_path / ".lybrary" / "file_hashes.json"
    evt = _event(str(lybrary_file))
    handler.on_any_event(evt)
    assert handler._pending == {}


# ---------------------------------------------------------------------------
# Move / rename events (Bug #7)
# ---------------------------------------------------------------------------


def test_move_event_enqueues_dest_path(tmp_path):
    handler = _make_handler(tmp_path)
    dest = tmp_path / "renamed.py"
    dest.write_text("def foo(): pass")
    evt = _move_event(
        src_path=str(tmp_path / "original.py"),
        dest_path=str(dest),
    )
    handler.on_any_event(evt)
    # dest path should be pending
    pending_files = list(handler._pending.keys())
    assert any("renamed.py" in p for p in pending_files)


def test_move_event_marks_old_path_for_deletion(tmp_path):
    handler = _make_handler(tmp_path)
    dest = tmp_path / "renamed.py"
    dest.write_text("def foo(): pass")
    evt = _move_event(
        src_path=str(tmp_path / "original.py"),
        dest_path=str(dest),
    )
    handler.on_any_event(evt)
    assert any("original.py" in p for p in handler._moved_from)


def test_process_pending_deletes_moved_from_paths(tmp_path):
    handler = _make_handler(tmp_path)
    handler._moved_from.add("old_path.py")

    handler.process_pending()

    handler.store.delete_path.assert_called_with("old_path.py")
    assert handler._moved_from == set()


# ---------------------------------------------------------------------------
# process_pending debounce
# ---------------------------------------------------------------------------


def test_process_pending_respects_debounce(tmp_path):
    config = LybraryConfig()
    config.daemon.debounce_ms = 10_000  # 10 seconds — nothing should fire
    store = MagicMock()
    store.load_hashes.return_value = {}
    embedder = MagicMock()
    handler = LybraryHandler(tmp_path, config, store, embedder)

    f = tmp_path / "src.py"
    f.write_text("x = 1")
    evt = _event(str(f))
    handler.on_any_event(evt)
    handler.process_pending()

    # _update_file should NOT have been called — debounce not elapsed
    store.upsert_chunks.assert_not_called()


# ---------------------------------------------------------------------------
# _update_file — deleted file
# ---------------------------------------------------------------------------


def test_update_file_deletes_missing_path(tmp_path):
    handler = _make_handler(tmp_path)
    rel = "src/gone.py"
    # File does not exist — _update_file should call delete_path
    handler._update_file(rel)
    handler.store.delete_path.assert_called_once_with(rel)


# ---------------------------------------------------------------------------
# _update_file — existing file (Bug #8: single read)
# ---------------------------------------------------------------------------


def test_update_file_reads_and_indexes(tmp_path):
    handler = _make_handler(tmp_path)

    src = tmp_path / "module.py"
    src.write_text("def hello(): pass\n")

    fake_chunk = MagicMock()
    fake_chunk.embedding_text.return_value = "def hello(): pass"

    with (
        patch("lybrary.watcher.chunk_source", return_value=[fake_chunk]) as mock_chunk,
        patch("lybrary.watcher.content_hash", return_value="deadbeef") as mock_hash,
    ):
        handler._update_file("module.py")

    # chunk_file and upsert_chunks should have been called
    mock_chunk.assert_called_once()
    handler.store.upsert_chunks.assert_called_once()
    # Hash should have been saved
    handler.store.save_hashes.assert_called()


def test_update_file_no_chunks_skips_upsert(tmp_path):
    handler = _make_handler(tmp_path)
    src = tmp_path / "empty.py"
    src.write_text("")

    with patch("lybrary.watcher.chunk_source", return_value=[]):
        handler._update_file("empty.py")

    handler.store.upsert_chunks.assert_not_called()


def test_watcher_ignores_gitignored_source(tmp_path):
    (tmp_path / ".gitignore").write_text("generated/\n")
    handler = _make_handler(tmp_path)
    path = tmp_path / "generated" / "code.py"
    path.parent.mkdir()
    path.write_text("def hidden(): pass")
    handler.on_any_event(_event(str(path)))
    assert handler._pending == {}


def test_update_uses_same_snapshot_for_chunks_and_hash(tmp_path):
    handler = _make_handler(tmp_path)
    path = tmp_path / "module.py"
    path.write_text("def original(): pass\n")
    with patch("lybrary.watcher.chunk_source", return_value=[MagicMock()]) as chunk:
        handler._update_file("module.py")
    assert chunk.call_args.args[0] == path.read_bytes().decode("utf-8")
    assert handler.store.save_hashes.call_args.args[0]["module.py"] == content_hash(
        path.read_bytes()
    )
