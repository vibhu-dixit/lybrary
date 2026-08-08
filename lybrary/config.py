"""Configuration management for lybrary."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class EmbedConfig(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32


class ChunkConfig(BaseModel):
    soft_max_tokens: int = 1200
    min_chunk_tokens: int = 50
    # merge_siblings was removed — the chunker always emits containers (classes,
    # impls) as one chunk and their nested definitions as separate chunks.


class IndexConfig(BaseModel):
    exclude: list[str] = Field(
        default_factory=lambda: [
            ".git",
            ".lybrary",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            "dist",
            "build",
            "*.min.js",
            "*.map",
            "*.lock",
        ]
    )
    languages: list[str] = Field(
        default_factory=lambda: [
            "python",
            "javascript",
            "typescript",
            "go",
            "rust",
            "java",
            "c",
            "cpp",
        ]
    )


class DaemonConfig(BaseModel):
    debounce_ms: int = 500
    host: str = "127.0.0.1"
    port: int = 7420  # "LIBR" on phone keypad-ish


class LybraryConfig(BaseSettings):
    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    chunk: ChunkConfig = Field(default_factory=ChunkConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)

    @classmethod
    def load(cls, root: Path) -> "LybraryConfig":
        config_path = root / ".lybrary" / "config.toml"
        if config_path.exists():
            with open(config_path, "rb") as f:
                data = tomllib.load(f)
            return cls.model_validate(data)
        return cls()

    def save(self, root: Path) -> None:
        config_dir = root / ".lybrary"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.toml"
        data = self.model_dump()
        with open(config_path, "wb") as f:
            tomli_w.dump(data, f)


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward looking for .git or .lybrary."""
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists() or (parent / ".lybrary").exists():
            return parent
    return current


def get_lybrary_dir(root: Path | None = None) -> Path:
    root = root or find_repo_root()
    return root / ".lybrary"
