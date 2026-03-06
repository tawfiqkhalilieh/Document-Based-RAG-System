"""
Load and flatten book data from books.json.
"""
import json
from typing import Any, Dict, List, Tuple

from config import BOOKS_JSON_PATH


def load_books() -> Dict[str, Dict[str, str]]:
    """Load the books JSON (e.g. produced by OCR)."""
    if not BOOKS_JSON_PATH.exists():
        raise FileNotFoundError(f"books.json not found at {BOOKS_JSON_PATH}")
    with open(BOOKS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            "books.json must contain a top-level object mapping book -> {page: text}"
        )
    return data


def flatten_books(books: Dict[str, Dict[str, str]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Turn {book: {page: text}} into parallel lists of texts and metadata."""
    texts: List[str] = []
    meta: List[Dict[str, Any]] = []
    for book_name, pages in books.items():
        if not isinstance(pages, dict):
            continue
        for page, text in pages.items():
            if not isinstance(text, str):
                continue
            cleaned = text.strip()
            if not cleaned:
                continue
            texts.append(cleaned)
            meta.append({"book": book_name, "page": str(page)})
    return texts, meta
