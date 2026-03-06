"""
Retrieval for QA: run search and attach page text to hits.
"""
from typing import Any, Dict, List

from books import load_books
from search import search


def build_context_snippets(
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Run semantic search and return snippets with book, page, score, and text."""
    books = load_books()
    hits = search(query, top_k=top_k)
    snippets: List[Dict[str, Any]] = []
    for hit in hits:
        book = hit["book"]
        page = hit["page"]
        text = ""
        if book in books and page in books[book]:
            text = books[book][page]
        snippets.append(
            {
                "book": book,
                "page": page,
                "score": hit["score"],
                "text": text,
            }
        )
    return snippets
