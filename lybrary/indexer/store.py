"""Persistent storage for chunks, embeddings, and file hashes.

Backend: SQLite (stdlib) + numpy for vector search.
No lancedb or pyarrow required — works on any Python 3.11+ installation.

Schema (single table 'chunks'):
    id              TEXT PRIMARY KEY
    path            TEXT NOT NULL
    symbol          TEXT
    kind            TEXT
    start_line      INTEGER
    end_line        INTEGER
    content         TEXT
    language        TEXT
    context_header  TEXT
    embedding_text  TEXT
    vector          BLOB   -- numpy float32 array serialised with np.save

Vector search is cosine similarity via numpy dot product on unit-normalised
vectors. For typical codebases (<50k chunks) this is fast enough; it scans
linearly in ~milliseconds at that scale.
"""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from lybrary.config import get_lybrary_dir
from lybrary.indexer.ast_chunker import CodeChunk
from lybrary.utils import content_hash, ensure_dir

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id             TEXT PRIMARY KEY,
    path           TEXT NOT NULL,
    symbol         TEXT,
    kind           TEXT,
    start_line     INTEGER,
    end_line       INTEGER,
    content        TEXT,
    language       TEXT,
    context_header TEXT,
    embedding_text TEXT,
    vector         BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks (path);
"""


def _vec_to_blob(vec: list[float]) -> bytes:
    """Serialise a float list to compact binary (numpy float32)."""
    arr = np.array(vec, dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    """Deserialise binary blob back to a numpy float32 array."""
    return np.load(io.BytesIO(blob))


class IndexStore:
    """SQLite-backed store for code chunks + embeddings."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.ly_dir = get_lybrary_dir(self.root)
        ensure_dir(self.ly_dir)
        self.db_path = self.ly_dir / "index.db"
        self.hash_path = self.ly_dir / "file_hashes.json"
        self._conn = self._connect()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL mode: faster writes, safe concurrent reads from watcher
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        conn.commit()
        return conn

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # File hash tracking
    # ------------------------------------------------------------------

    def load_hashes(self) -> dict[str, str]:
        if self.hash_path.exists():
            return json.loads(self.hash_path.read_text(encoding="utf-8"))
        return {}

    def save_hashes(self, hashes: dict[str, str]) -> None:
        self.hash_path.write_text(
            json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8"
        )

    def get_changed_files(self, current: dict[str, str]) -> tuple[list[str], list[str]]:
        """Return (changed_or_new, deleted) relative paths."""
        old = self.load_hashes()
        changed = [p for p, h in current.items() if old.get(p) != h]
        deleted = [p for p in old if p not in current]
        return changed, deleted

    # ------------------------------------------------------------------
    # Chunk CRUD
    # ------------------------------------------------------------------

    def delete_path(self, rel_path: str) -> None:
        """Remove all chunks for *rel_path* (parameterised — no injection risk)."""
        self._conn.execute("DELETE FROM chunks WHERE path = ?", (rel_path,))
        self._conn.commit()

    def truncate(self) -> None:
        """Drop all chunks for a true full re-index."""
        self._conn.execute("DELETE FROM chunks")
        self._conn.commit()

    def upsert_chunks(self, chunks: list[CodeChunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        # Delete stale rows for each affected path first
        paths = {c.path for c in chunks}
        self._conn.executemany(
            "DELETE FROM chunks WHERE path = ?", [(p,) for p in paths]
        )
        self._conn.executemany(
            """
            INSERT INTO chunks
                (id, path, symbol, kind, start_line, end_line,
                 content, language, context_header, embedding_text, vector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    c.id,
                    c.path,
                    c.symbol,
                    c.kind,
                    c.start_line,
                    c.end_line,
                    c.content,
                    c.language,
                    c.context_header,
                    c.embedding_text(),
                    _vec_to_blob(vec),
                )
                for c, vec in zip(chunks, vectors)
            ],
        )
        self._conn.commit()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def search_vector(self, query_vector: list[float], limit: int = 10) -> list[dict[str, Any]]:
        """Cosine-similarity search via numpy.

        Vectors are unit-normalised at embed time so cosine similarity
        reduces to a dot product: sim = query · chunk_vec.
        Returns rows sorted by descending similarity, with a '_distance'
        key (L2 distance in [0,2]) for compatibility with the query layer.
        """
        rows = self._conn.execute(
            """
            SELECT id, path, symbol, kind, start_line, end_line,
                   content, language, context_header, embedding_text, vector
            FROM chunks
            """
        ).fetchall()

        if not rows:
            return []

        q = np.array(query_vector, dtype=np.float32)
        # Stack all stored vectors into a matrix for a single batched dot product
        vecs = np.stack([_blob_to_vec(r["vector"]) for r in rows])  # (N, D)
        scores = vecs @ q  # cosine similarity for unit vectors, shape (N,)

        # Pick top-k indices
        k = min(limit, len(rows))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]  # sorted desc

        results = []
        for i in top_idx:
            r = rows[i]
            sim = float(scores[i])
            # Convert cosine similarity back to L2 distance for the query layer:
            # sim = 1 - d²/2  →  d = sqrt(2*(1-sim)), clamped to [0, 2]
            d2 = max(0.0, 2.0 * (1.0 - sim))
            results.append(
                {
                    "id": r["id"],
                    "path": r["path"],
                    "symbol": r["symbol"],
                    "kind": r["kind"],
                    "start_line": r["start_line"],
                    "end_line": r["end_line"],
                    "content": r["content"],
                    "language": r["language"],
                    "context_header": r["context_header"],
                    "embedding_text": r["embedding_text"],
                    "_distance": d2,
                }
            )
        return results

    def all_paths(self) -> set[str]:
        rows = self._conn.execute("SELECT DISTINCT path FROM chunks").fetchall()
        return {r["path"] for r in rows}
