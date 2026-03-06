"""
CLI entry points: QA chat and RAG build/search.
"""
import argparse

from config import INDEX_PATH, META_PATH
from embedding import build_index
from qa_pipeline import answer_query
from search import search


def cli_chat() -> None:
    """Interactive QA loop in the terminal."""
    print("Classical Arabic Philosophy QA (Qwen + BGE-M3)")
    print("Type your question, or 'exit' to quit.\n")

    while True:
        try:
            q = input("سؤالك> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        result = answer_query(q)
        print("\nالإجابة:\n")
        print(result["answer"])
        if result["snippets"]:
            print("\nالمراجع المستخدمة:")
            for s in result["snippets"]:
                print(f"- {s['book']} (صفحة {s['page']}) [score={s['score']:.3f}]")
        print("\n" + "=" * 80 + "\n")


def cli_build_index() -> None:
    """Build BGE-M3 index from books.json."""
    print("Building BGE-M3 index from books.json ...")
    build_index()
    print(f"Saved index to {INDEX_PATH.name} and metadata to {META_PATH.name}")


def cli_search(query: str, top_k: int = 5) -> None:
    """Print semantic search results for a query."""
    print(f"Query: {query}")
    results = search(query, top_k=top_k)
    if not results:
        print("No results.")
        return
    for r in results:
        print(f"- [{r['score']:.3f}] {r['book']} (page {r['page']})")


def main_rag() -> None:
    """RAG CLI: build index or run search."""
    parser = argparse.ArgumentParser(
        description="BGE-M3 search over books.json pages"
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("build", help="Build embeddings index from books.json")

    search_parser = subparsers.add_parser("search", help="Run a semantic search query")
    search_parser.add_argument("query", type=str, help="Search query text")
    search_parser.add_argument(
        "--top_k", type=int, default=5, help="Number of results to return"
    )

    args = parser.parse_args()

    if args.command == "build":
        cli_build_index()
    elif args.command == "search":
        cli_search(args.query, top_k=args.top_k)
    else:
        parser.print_help()
