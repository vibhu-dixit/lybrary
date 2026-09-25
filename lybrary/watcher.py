"""File system watcher for incremental updates."""

from __future__ import annotations

import time
from pathlib import Path

from watchdog.events import (
    FileMovedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from lybrary.config import LybraryConfig
from lybrary.indexer.ast_chunker import chunk_source, detect_language
from lybrary.indexer.embedder import Embedder
from lybrary.indexer.store import IndexStore
from lybrary.indexer.walker import _build_spec
from lybrary.utils import content_hash, relative_to_root


class LybraryHandler(FileSystemEventHandler):
    def __init__(self, root: Path, config: LybraryConfig, store: IndexStore, embedder: Embedder):
        super().__init__()
        self.root = root
        self.config = config
        self.store = store
        self.embedder = embedder
        self._pending: dict[str, float] = {}
        # Fix #7: track old paths from move events so we can delete them
        self._moved_from: set[str] = set()
        self.debounce = config.daemon.debounce_ms / 1000.0
        self._ignore = _build_spec(root, config)

    def _enqueue(self, path: Path) -> None:
        """Add a source file to the pending update queue."""
        if detect_language(path) is None:
            return
        # Ignore changes inside our own data directory
        try:
            if ".lybrary" in path.parts:
                return
        except Exception:
            return
        rel = relative_to_root(path, self.root)
        if self._ignore.match_file(rel.replace("\\", "/")):
            return
        self._pending[rel] = time.time()

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return

        # Fix #7: FileMovedEvent carries both src_path (old) and dest_path (new).
        # Enqueue the new path for re-indexing and mark the old path for deletion.
        if isinstance(event, FileMovedEvent):
            old_path = Path(event.src_path)
            new_path = Path(event.dest_path)
            # Queue the old relative path for deletion
            try:
                if ".lybrary" not in old_path.parts:
                    old_rel = relative_to_root(old_path, self.root)
                    self._moved_from.add(old_rel)
            except Exception:
                pass
            # Queue the new path for indexing
            self._enqueue(new_path)
            return

        self._enqueue(Path(event.src_path))

    def process_pending(self) -> None:
        now = time.time()
        ready = [rel for rel, ts in self._pending.items() if now - ts >= self.debounce]

        # Fix #7: remove stale chunks for files that were moved/renamed
        for old_rel in list(self._moved_from):
            print(f"[watch] removing moved/renamed path {old_rel}", flush=True)
            self.store.delete_path(old_rel)
            hashes = self.store.load_hashes()
            hashes.pop(old_rel, None)
            self.store.save_hashes(hashes)
        self._moved_from.clear()

        for rel in ready:
            del self._pending[rel]
            self._update_file(rel)

    def _update_file(self, rel: str) -> None:
        path = self.root / rel
        print(f"[watch] updating {rel}", flush=True)
        if not path.exists():
            self.store.delete_path(rel)
            hashes = self.store.load_hashes()
            hashes.pop(rel, None)
            self.store.save_hashes(hashes)
            return

        try:
            # Fix #8: read the file bytes exactly once to guarantee the chunks
            # and the stored hash are derived from the same snapshot.
            raw_bytes = path.read_bytes()
            source = raw_bytes.decode("utf-8", errors="replace")

            chunks = chunk_source(
                source,
                rel,
                detect_language(path),
                soft_max_tokens=self.config.chunk.soft_max_tokens,
                min_chunk_tokens=self.config.chunk.min_chunk_tokens,
            )
            if not chunks:
                self.store.delete_path(rel)
                hashes = self.store.load_hashes()
                hashes[rel] = content_hash(raw_bytes)
                self.store.save_hashes(hashes)
                return
            texts = [c.embedding_text() for c in chunks]
            vectors = self.embedder.embed(texts, batch_size=self.config.embed.batch_size)
            self.store.upsert_chunks(chunks, vectors)

            # Fix #8: hash is computed from the same bytes we just chunked
            hashes = self.store.load_hashes()
            hashes[rel] = content_hash(raw_bytes)
            self.store.save_hashes(hashes)
            print(f"[watch] {rel} -> {len(chunks)} chunks", flush=True)
        except Exception as e:
            print(f"[watch] error on {rel}: {e}", flush=True)


def start_watching(root: Path, config: LybraryConfig) -> None:
    store = IndexStore(root)
    embedder = Embedder(config.embed.model, config.embed.device)
    handler = LybraryHandler(root, config, store, embedder)
    observer = Observer()
    observer.schedule(handler, str(root), recursive=True)
    observer.start()
    print("File watcher started", flush=True)
    try:
        while True:
            time.sleep(0.25)
            handler.process_pending()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
