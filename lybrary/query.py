"""Hybrid search over the code memory."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from lybrary.config import LybraryConfig
from lybrary.indexer.embedder import Embedder
from lybrary.indexer.store import IndexStore

_STOPWORDS = {
    "and", "are", "code", "does", "file", "find", "for", "from", "how",
    "becoming", "into", "line", "lines", "prevents", "reaches", "that",
    "the", "this", "trace", "what", "where", "which", "with",
}


def query_terms(query: str) -> list[str]:
    """Keep distinctive words for lexical search and source excerpts."""
    split_names = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", query)
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", split_names.lower())
    return list(dict.fromkeys(w for w in words if len(w) >= 3 and w not in _STOPWORDS))[:12]


def _lexical_coverage(row: dict[str, Any], terms: list[str]) -> int:
    text = " ".join(str(row.get(key) or "") for key in ("path", "symbol", "content")).lower()
    return sum(term in text for term in terms)


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
    max_tokens: int | None = 6000,
) -> list[dict[str, Any]]:
    """Exact symbol or lexical-first search with semantic fallback.

    Returns results sorted by descending relevance score.
    Each result dict contains: id, path, symbol, kind, start_line,
    end_line, content, context_header, score.
    """
    quoted = re.search(r"`([A-Za-z_$][\w$]*)`", query)
    identifier = quoted.group(1) if quoted else None
    if identifier is None and re.fullmatch(r"[A-Za-z_$][\w$]*", query.strip()):
        identifier = query.strip()
    store = IndexStore(root, read_only=True)
    try:
        raw = store.search_symbol(identifier, limit=limit * 2) if identifier else []
        text_match = bool(raw)
        lexical_only = False
        terms = query_terms(query)
        if not raw:
            lexical = store.search_text(terms, limit=limit * 3)
            required = min(3, (len(terms) + 1) // 2)
            if lexical and (len(terms) == 1 or any(
                _lexical_coverage(row, terms) >= required for row in lexical[:5]
            )):
                raw = lexical
                lexical_only = True
            else:
                config = LybraryConfig.load(root)
                embedder = Embedder(config.embed.model, config.embed.device)
                semantic = store.search_vector(embedder.embed_query(query), limit=limit * 3)
                if lexical:
                    combined: dict[str, dict[str, Any]] = {}
                    for rows, weight in ((semantic, 1.0), (lexical, 1.2)):
                        for rank, row in enumerate(rows, 1):
                            key = row["id"]
                            if key not in combined:
                                combined[key] = dict(row)
                                combined[key]["_hybrid_score"] = 0.0
                            combined[key]["_hybrid_score"] += weight / rank / 2.2
                    raw = sorted(
                        combined.values(), key=lambda row: row["_hybrid_score"], reverse=True
                    )
                else:
                    raw = semantic
    finally:
        store.close()

    results: list[dict[str, Any]] = []
    used = 0
    per_path: dict[str, int] = {}
    for row in raw:
        path = row.get("path", "")
        if not text_match and per_path.get(path, 0) >= (3 if lexical_only else 2):
            continue
        content = row.get("content", "")
        approx = max(1, len(content) // 4)
        if max_tokens is not None and used + approx > max_tokens and results:
            continue
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
                "score": (
                    1.0 if text_match else (
                        max(1, _lexical_coverage(row, terms)) / max(1, len(terms))
                        if lexical_only else row.get(
                        "_hybrid_score", _distance_to_score(row.get("_distance", 2.0))
                        )
                    )
                ),
            }
        )
        used += approx
        per_path[path] = per_path.get(path, 0) + 1
        if len(results) >= limit:
            break

    return results
