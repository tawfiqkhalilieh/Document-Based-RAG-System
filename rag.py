"""
RAG API and CLI entry point.

Public API (for imports):
  - load_books, search, build_index

CLI: python rag.py build | python rag.py search "query" [--top_k N]
"""
from books import load_books
from embedding import build_index
from search import search

__all__ = ["load_books", "search", "build_index"]

if __name__ == "__main__":
    from cli import main_rag
    main_rag()
