"""Shared utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path

import xxhash


def content_hash(data: bytes | str) -> str:
    """Fast content hash for change detection."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return xxhash.xxh64(data).hexdigest()


def file_hash(path: Path) -> str:
    """Hash file contents."""
    return content_hash(path.read_bytes())


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
