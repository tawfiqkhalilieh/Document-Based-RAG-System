"""
Project configuration: paths and device selection.
"""
import os
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent
BOOKS_DIR = ROOT_DIR / "books"
BOOKS_JSON_PATH = BOOKS_DIR / "books.json"
INDEX_PATH = BOOKS_DIR / "books_bge_m3_index.npz"
META_PATH = BOOKS_DIR / "books_bge_m3_meta.json"

# Qwen model; override with a local path if you have cloned Qwen2.5/Qwen3.
DEFAULT_QWEN_PATH = "Qwen/Qwen3-4B-Instruct-2507"


def get_device(prefer: str | None = None) -> str:
    """
    Choose compute device: 'cpu' or 'cuda'.

    Priority:
    1) `prefer` if given and valid (and cuda only if available).
    2) Environment variable QWEN_DEVICE = 'cpu' or 'cuda'.
    3) CUDA if available, else CPU.
    """
    candidates: list[str] = []
    if prefer and prefer.strip().lower() in ("cpu", "cuda"):
        candidates.append(prefer.strip().lower())
    env_val = os.getenv("QWEN_DEVICE", "").strip().lower()
    if env_val in ("cpu", "cuda"):
        candidates.append(env_val)
    candidates.append("cuda" if torch.cuda.is_available() else "cpu")

    for d in candidates:
        if d == "cuda" and not torch.cuda.is_available():
            continue
        if d in ("cpu", "cuda"):
            return d
    return "cpu"
