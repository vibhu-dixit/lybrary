"""Repository walker and indexing orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pathspec import PathSpec
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from lybrary.config import LybraryConfig
from lybrary.indexer.ast_chunker import chunk_file, detect_language
from lybrary.indexer.embedder import Embedder
from lybrary.indexer.store import IndexStore
from lybrary.utils import content_hash, relative_to_root


def _build_spec(root: Path, config: LybraryConfig) -> PathSpec:
    patterns = list(config.index.exclude)
    # Also respect .gitignore if present
    gitignore = root / ".gitignore"
    if gitignore.exists():
        patterns.extend(gitignore.read_text().splitlines())
    ignore_file = root / ".lybrary" / "ignore"
    if ignore_file.exists():
        patterns.extend(ignore_file.read_text().splitlines())
    return PathSpec.from_lines("gitwildmatch", patterns)


def iter_source_files(root: Path, config: LybraryConfig) -> list[Path]:
    spec = _build_spec(root, config)
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = relative_to_root(p, root)
        if spec.match_file(rel):
            continue
        if detect_language(p) is None:
            continue
        files.append(p)
    return files


def index_repository(
    root: Path,
    store: IndexStore,
    config: LybraryConfig,
    full: bool = False,
) -> dict[str, Any]:
    """Index the whole repository (full or incremental)."""
    files = iter_source_files(root, config)
    current_hashes: dict[str, str] = {}
    for f in files:
        rel = relative_to_root(f, root)
        current_hashes[rel] = content_hash(f.read_bytes())

    if full:
        changed = list(current_hashes.keys())
        deleted = []
        # Fix #9: drop and recreate the table so no stale rows survive,
        # even if all_paths() would have missed inconsistent data.
        store.truncate()
    else:
        changed, deleted = store.get_changed_files(current_hashes)

    for p in deleted:
        store.delete_path(p)

    if not changed:
        store.save_hashes(current_hashes)
        return {"files": len(files), "chunks": store.count(), "updated": 0}

    embedder = Embedder(config.embed.model, config.embed.device)
    all_new_chunks = []
    updated = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Chunking & embedding", total=len(changed))

        for rel in changed:
            path = root / rel
            if not path.exists():
                progress.advance(task)
                continue
            chunks = chunk_file(
                path,
                root,
                soft_max_tokens=config.chunk.soft_max_tokens,
                min_chunk_tokens=config.chunk.min_chunk_tokens,
            )
            if not chunks:
                progress.advance(task)
                continue

            texts = [c.embedding_text() for c in chunks]
            vectors = embedder.embed(texts, batch_size=config.embed.batch_size)
            store.upsert_chunks(chunks, vectors)
            all_new_chunks.extend(chunks)
            updated += 1
            progress.advance(task)

    store.save_hashes(current_hashes)
    return {
        "files": len(files),
        "chunks": store.count(),
        "updated": updated,
        "new_chunks": len(all_new_chunks),
    }
