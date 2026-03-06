import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path


class _HTMLTextExtractor(HTMLParser):
    """
    Minimal HTML-to-text converter using the standard library.
    """

    _IGNORED_TAGS = {"style", "script", "head"}

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._ignore_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._IGNORED_TAGS:
            self._ignore_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._IGNORED_TAGS and self._ignore_depth > 0:
            self._ignore_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignore_depth > 0:
            return
        if data and data.strip():
            self._chunks.append(data)

    def get_text(self) -> str:
        # Join with spaces and normalize whitespace
        raw = " ".join(self._chunks)
        # Decode entities like &nbsp; &amp; etc.
        return " ".join(unescape(raw).split())


def html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.get_text()


def convert_digits(text: str) -> str:
    """Convert Arabic-Indic digits to English digits."""
    ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
    ENGLISH_DIGITS = "0123456789"
    table = str.maketrans(ARABIC_INDIC_DIGITS, ENGLISH_DIGITS)
    return text.translate(table)


def file_to_book_key(path: Path, htm_dir: Path) -> str:
    """
    Derive the book key. If the file is in a subdirectory, 
    include the parent folder name to distinguish volumes.
    """
    rel_path = path.relative_to(htm_dir)
    if len(rel_path.parts) > 1:
        # e.g. "درء تعارض العقل والنقل/001.htm" -> "درء تعارض العقل والنقل - 001"
        return f"{rel_path.parent.name} - {path.stem}"
    return path.stem


def extract_pages(html_content: str) -> dict[str, str]:
    """
    Extract pages from Shamela-style HTM content.
    Uses <div class='PageText'> as the delimiter and <span class='PageNumber'> for labeling.
    """
    pages: dict[str, str] = {}
    
    # Split by the PageText div. 
    # Shamela HTML uses <div class='PageText'> or <div class="PageText">
    parts = re.split(r"<div\s+class=['\"]PageText['\"]>", html_content, flags=re.IGNORECASE)
    
    # The first part (parts[0]) is everything before the first PageText div (usually headers/styles).
    # We iterate through the rest.
    page_counter = 1
    for part in parts[1:]:
        # Try to find a page number marker: <span class='PageNumber'>(ص: ١)</span>
        pg_match = re.search(r"<span\s+class=['\"]PageNumber['\"]>\s*\((.*?)\)\s*</span>", part, flags=re.IGNORECASE)
        
        pg_key = ""
        if pg_match:
            pg_label = pg_match.group(1)
            # Look for digits (Arabic or English) in the label
            num_match = re.search(r"(\d+|[٠-٩]+)", pg_label)
            if num_match:
                pg_key = convert_digits(num_match.group(1))
        
        # Fallback to sequential counter if no number found or if it's empty
        if not pg_key:
            pg_key = str(page_counter)
            page_counter += 1
        
        # Ensure the key is unique within this book (handle duplicate markers if any)
        original_pg_key = pg_key
        suffix = 1
        while pg_key in pages:
            pg_key = f"{original_pg_key}_{suffix}"
            suffix += 1

        text = html_to_text(part)
        if text.strip():
            pages[pg_key] = text
            
    return pages


def load_books_json(books_json_path: Path) -> dict:
    if not books_json_path.exists():
        return {}
    with books_json_path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_books_json(books_json_path: Path, data: dict) -> None:
    # Use UTF-8 and ensure Arabic keys/text are preserved
    with books_json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_htm_files_to_books(
    base_dir: Path | None = None,
    htm_subdir: str = "htm-files",
    books_json_name: str = "books.json",
) -> None:
    """
    Read all .htm files under books/htm-files recursively and add them to books.json.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent

    htm_dir = base_dir / htm_subdir
    books_json_path = base_dir / books_json_name

    if not htm_dir.exists():
        print(f"Error: HTML directory not found at {htm_dir}")
        return

    data = load_books_json(books_json_path)

    # Find all .htm / .html files recursively
    htm_files = sorted(list(htm_dir.rglob("*.htm")) + list(htm_dir.rglob("*.html")))

    print(f"Found {len(htm_files)} HTML files. Processing...")

    for htm_path in htm_files:
        key = file_to_book_key(htm_path, htm_dir)
        
        # Log progress for every 10 files
        print(f"Processing: {key}")

        with htm_path.open("r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        pages = extract_pages(html_content)

        if pages:
            data[key] = pages

    save_books_json(books_json_path, data)
    print(f"Successfully updated {books_json_path} with {len(htm_files)} volumes.")


if __name__ == "__main__":
    add_htm_files_to_books()
