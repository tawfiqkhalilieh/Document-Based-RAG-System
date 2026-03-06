"""
BGE-M3 embedding model and dense index build/load.
"""
import json
from typing import Any, Dict, List, Tuple

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from books import flatten_books, load_books
from config import INDEX_PATH, META_PATH, get_device


def _load_bge_model(device: str | None = None) -> BGEM3FlagModel:
    """Load BAAI/bge-m3. Uses FP16 on GPU when available."""
    dev = device or get_device()
    use_fp16 = dev == "cuda"
    return BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=use_fp16,
        device=dev,
    )


def build_index(
    batch_size: int = 16,
    max_length: int = 512,
    device: str | None = None,
) -> None:
    """
    Build (or rebuild) dense embedding index for all pages in books.json.
    Saves L2-normalized embeddings to INDEX_PATH and metadata to META_PATH.
    """
    books = load_books()
    texts, meta = flatten_books(books)
    if not texts:
        raise ValueError("No non-empty page texts found in books.json")

    model = _load_bge_model(device)
    all_vecs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        outputs = model.encode(
            batch,
            batch_size=len(batch),
            max_length=max_length,
        )
        dense_vecs = np.asarray(outputs["dense_vecs"], dtype="float32")
        all_vecs.append(dense_vecs)

    embeddings = np.vstack(all_vecs).astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-6, None)
    embeddings = embeddings / norms

    np.savez_compressed(INDEX_PATH, embeddings=embeddings)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_index() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Load embeddings and metadata from disk."""
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Missing index files. Run `python rag.py build` from the project root first."
        )
    data = np.load(INDEX_PATH)
    embeddings = np.asarray(data["embeddings"], dtype="float32")
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if len(meta) != embeddings.shape[0]:
        raise ValueError(
            "Index/meta size mismatch: embeddings and metadata lengths differ"
        )
    return embeddings, meta


# Module-level caches
_MODEL_CACHE: BGEM3FlagModel | None = None
_EMBEDDINGS_CACHE: np.ndarray | None = None
_META_CACHE: List[Dict[str, Any]] | None = None


def ensure_index_loaded() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Return in-memory index and meta, loading from disk if needed."""
    global _EMBEDDINGS_CACHE, _META_CACHE
    if _EMBEDDINGS_CACHE is None or _META_CACHE is None:
        emb, m = load_index()
        _EMBEDDINGS_CACHE = emb
        _META_CACHE = m
    return _EMBEDDINGS_CACHE, _META_CACHE  # type: ignore[return-value]


def ensure_model_loaded(device: str | None = None) -> BGEM3FlagModel:
    """Return BGE model, loading once and caching."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = _load_bge_model(device)
    return _MODEL_CACHE
