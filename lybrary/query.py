"""Hybrid search over the code memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lybrary.config import LybraryConfig
from lybrary.indexer.embedder import Embedder
from lybrary.indexer.store import IndexStore


def _distance_to_score(distance: float) -> float:
    """Fix #11: convert LanceDB L2 distance to a [0, 1] relevance score.

    The embedder normalises vectors to unit length, so L2 distance is in
    [0, 2] (0 = identical, 2 = opposite).  We convert to cosine similarity
    with:  similarity = 1 - (distance² / 2)

    This gives a proper [−1, 1] range (clipped to [0, 1] for display) that
    is directly interpretable as cosine similarity, rather than the previous
    formula which compressed all variation above distance=1 into near-zero.
    """
    # cosine_sim = 1 - L2² / 2  (valid when vectors are unit-normalised)
    score = 1.0 - (distance ** 2) / 2.0
    return max(0.0, min(1.0, score))


def search(
    root: Path,
    query: str,
    limit: int = 8,
    max_tokens: int = 6000,
) -> list[dict[str, Any]]:
    """Semantic search with simple token-budget packing.

    Returns results sorted by descending cosine similarity score.
    Each result dict contains: id, path, symbol, kind, start_line,
    end_line, content, context_header, score.
    """
    config = LybraryConfig.load(root)
    store = IndexStore(root)
    embedder = Embedder(config.embed.model, config.embed.device)

    qvec = embedder.embed_query(query)
    # Fetch extra candidates so the token-budget filter still has choices.
    raw = store.search_vector(qvec, limit=limit * 2)

    results: list[dict[str, Any]] = []
    used = 0
    for row in raw:
        content = row.get("content", "")
        approx = max(1, len(content) // 4)
        if used + approx > max_tokens and results:
            break
        results.append(
            {
                "id": row.get("id"),
                "path": row.get("path"),
                "symbol": row.get("symbol"),
                "kind": row.get("kind"),
                "start_line": row.get("start_line"),
                "end_line": row.get("end_line"),
                "content": content,
                "context_header": row.get("context_header"),
                # Fix #11: use proper cosine-similarity score
                "score": _distance_to_score(row.get("_distance", 2.0)),
            }
        )
        used += approx
        if len(results) >= limit:
            break

    return results
