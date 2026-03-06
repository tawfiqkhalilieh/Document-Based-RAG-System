"""
Semantic search over book pages using BGE-M3 embeddings.
"""
from typing import Any, Dict, List

import numpy as np

from embedding import ensure_index_loaded, ensure_model_loaded


def search(
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Semantic search over all pages. Returns list of dicts with
    keys: score (float), book (str), page (str).
    """
    if not query.strip():
        return []

    embeddings, meta = ensure_index_loaded()
    model = ensure_model_loaded()

    q_outputs = model.encode(
        [query.strip()],
        batch_size=1,
        max_length=512,
    )
    q_vec = np.asarray(q_outputs["dense_vecs"][0], dtype="float32")
    q_norm = np.linalg.norm(q_vec)
    if q_norm < 1e-6:
        return []
    q_vec = q_vec / q_norm

    scores = embeddings @ q_vec
    top_k = max(1, min(top_k, scores.shape[0]))
    top_indices = np.argsort(-scores)[:top_k]

    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        item = dict(meta[idx])
        item["score"] = float(scores[idx])
        results.append(item)
    return results
