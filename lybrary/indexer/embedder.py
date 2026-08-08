"""Embedding generation using fastembed (ONNX Runtime backend).

fastembed runs the same MiniLM-L6-v2 model as sentence-transformers but via
ONNX Runtime, with no torch/scipy/transformers dependency. Install footprint
is ~50MB vs ~700MB, and CPU inference is 2-3x faster.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np


@lru_cache(maxsize=8)
def _get_model(model_name: str):
    """Load and cache a fastembed TextEmbedding model by name."""
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=model_name)


class Embedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",  # kept for config compatibility; fastembed auto-selects
    ):
        self.model_name = model_name
        self.device = device.strip().lower()

    @property
    def model(self):
        return _get_model(self.model_name)

    def embed(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        # fastembed.embed() is a generator of numpy arrays (already unit-normalised)
        embeddings = list(self.model.embed(list(texts), batch_size=batch_size))
        return [e.tolist() for e in embeddings]

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]
