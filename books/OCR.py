import gradio as gr
from transformers import AutoModel, AutoTokenizer
import torch
import os
import sys
import tempfile
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageOps
import fitz
import re
import numpy as np
import base64
from io import StringIO, BytesIO
from rich.progress import Progress
from io import BytesIO
from PIL import Image
import pymupdf
import logging
import transformers
import warnings
from pathlib import Path
import re


MODEL_NAME = 'deepseek-ai/DeepSeek-OCR-2'

# Use CUDA if available; otherwise CPU (bfloat16 not well supported on CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

if DEVICE == "cpu":
    print("⚠️ CUDA not available, running on CPU (this will be slower)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModel.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True, use_safetensors=True
).eval()
model = model.to(DEVICE)

BASE_SIZE = 1024
IMAGE_SIZE = 768
CROP_MODE = True

TASK_PROMPTS = {
    "📋 Markdown": {"prompt": "<image>\n<|grounding|>Convert the document to markdown.", "has_grounding": True},
    "📝 Free OCR": {"prompt": "<image>\nFree OCR.", "has_grounding": False},
    "📍 Locate": {"prompt": "<image>\nLocate <|ref|>text<|/ref|> in the image.", "has_grounding": True},
    "🔍 Describe": {"prompt": "<image>\nDescribe this image in detail.", "has_grounding": False},
    "✏️ Custom": {"prompt": "", "has_grounding": False}
}

def normalize_arabic(text):

    text = re.sub(r'[ًٌٍَُِّْـ]', '', text)  # remove tashkeel
    text = text.replace("ى","ي")
    text = text.replace("ة","ه")

    text = re.sub(r'\s+', ' ', text)

    return text

# Cross-platform font for bounding-box labels (avoids OSError on Windows)
def _get_label_font(size=15):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arialbd.ttf"),
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arial.ttf"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def extract_grounding_references(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    return re.findall(pattern, text, re.DOTALL)


def draw_bounding_boxes(image, refs, extract_images=False):
    img_w, img_h = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    font = _get_label_font(15)
    crops = []

    color_map = {}
    np.random.seed(42)

    for ref in refs:
        label = ref[1]
        if label not in color_map:
            color_map[label] = (np.random.randint(50, 255), np.random.randint(
                50, 255), np.random.randint(50, 255))

        color = color_map[label]
        coords_str = re.sub(r'[^\d,.\[\]\s\-]', '', ref[2])
        try:
            coords = eval(coords_str)
        except:
            continue
        color_a = color + (60,)

        for box in coords:
            x1, y1, x2, y2 = int(
                box[0]/999*img_w), int(box[1]/999*img_h), int(box[2]/999*img_w), int(box[3]/999*img_h)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            if extract_images and label == 'image':
                crops.append(image.crop((x1, y1, x2, y2)))

            width = 5 if label == 'title' else 3
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            draw2.rectangle([x1, y1, x2, y2], fill=color_a)

            text_bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            ty = max(0, y1 - 20)
            draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=color)
            draw.text((x1 + 2, ty + 2), label, font=font, fill=(255, 255, 255))

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw, crops


def clean_output(text, include_images=False):
    if not text:
        return ""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    img_num = 0

    for match in matches:
        if '<|ref|>image<|/ref|>' in match[0]:
            if include_images:
                text = text.replace(
                    match[0], f'\n\n**[Figure {img_num + 1}]**\n\n', 1)
                img_num += 1
            else:
                text = text.replace(match[0], '', 1)
        else:
            text = re.sub(
                rf'(?m)^[^\n]*{re.escape(match[0])}[^\n]*\n?', '', text)

    text = text.replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

    return text.strip()


def embed_images(markdown, crops):
    if not crops:
        return markdown
    for i, img in enumerate(crops):
        buf = BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        markdown = markdown.replace(
            f'**[Figure {i + 1}]**', f'\n\n![Figure {i + 1}](data:image/png;base64,{b64})\n\n', 1)
    return markdown


def process_image(image, task, custom_prompt):
    if image is None:
        return "Error: Upload an image", "", "", None, []
    if task in ["✏️ Custom", "📍 Locate"] and not custom_prompt.strip():
        return "Please enter a prompt", "", "", None, []

    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    image = ImageOps.exif_transpose(image)

    if task == "✏️ Custom":
        prompt = f"<image>\n{custom_prompt.strip()}"
        has_grounding = '<|grounding|>' in custom_prompt
    elif task == "📍 Locate":
        prompt = f"<image>\nLocate <|ref|>{custom_prompt.strip()}<|/ref|> in the image."
        has_grounding = True
    else:
        prompt = TASK_PROMPTS[task]["prompt"]
        has_grounding = TASK_PROMPTS[task]["has_grounding"]

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    image.save(tmp.name, 'JPEG', quality=95)
    tmp.close()
    out_dir = tempfile.mkdtemp()

    stdout = sys.stdout
    sys.stdout = StringIO()

    model.infer(
        tokenizer=tokenizer,
        prompt=prompt,
        image_file=tmp.name,
        output_path=out_dir,
        base_size=BASE_SIZE,
        image_size=IMAGE_SIZE,
        crop_mode=CROP_MODE,
        save_results=False,
    )

    debug_filters = ['PATCHES', '====', 'BASE:',
                     'directly resize', 'NO PATCHES', 'torch.Size', '%|']
    result = '\n'.join([l for l in sys.stdout.getvalue().split('\n')
                        if l.strip() and not any(s in l for s in debug_filters)]).strip()
    sys.stdout = stdout

    os.unlink(tmp.name)
    shutil.rmtree(out_dir, ignore_errors=True)

    if not result:
        return "No text detected", "", "", None, []

    cleaned = clean_output(result, False)
    markdown = clean_output(result, True)

    img_out = None
    crops = []

    if has_grounding and '<|ref|>' in result:
        refs = extract_grounding_references(result)
        if refs:
            img_out, crops = draw_bounding_boxes(image, refs, True)

    markdown = embed_images(markdown, crops)

    return cleaned, markdown, result, img_out, crops


import json

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

BOOKS_JSON_PATH = Path(__file__).resolve().parent / "books.json"
# Set your PDF path here or pass as first command-line argument
PDF_PATH = "pdfs/aa.pdf"


def pdf_page_to_pil(page, dpi=150):
    """Render a pymupdf page to a PIL Image for OCR."""
    mat = pymupdf.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    return Image.open(BytesIO(img_bytes))


def ocr_pdf_to_pages(pdf_path: str) -> dict[str, str]:
    """Run OCR on each page of the PDF. Returns a dict: page_number (str) -> page text."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = pymupdf.open(pdf_path)
    pages = {}
    with Progress() as progress:
        task = progress.add_task("OCR pages", total=len(doc))
        for page_index in range(len(doc)):
            page = doc[page_index]
            pil_img = pdf_page_to_pil(page)
            cleaned, markdown, raw, img_out, crops = process_image(
                pil_img, task="📋 Markdown", custom_prompt=""
            )
            pages[str(page_index + 1)] = cleaned
            progress.update(task, advance=1)
    doc.close()
    return pages


def load_books() -> dict:
    """Load books.json; return empty dict if missing or invalid."""
    if not BOOKS_JSON_PATH.exists():
        return {}
    try:
        with open(BOOKS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_books(books: dict) -> None:
    """Write books dict to books.json."""
    BOOKS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BOOKS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(books, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    for PDF_PATH in Path("pdfs").glob("*.pdf"):
        book_key = Path(PDF_PATH).stem
        print(book_key)
        pages = ocr_pdf_to_pages(PDF_PATH)
        books = load_books()
        books[book_key] = pages
        save_books(books)
        print(f"Saved {len(pages)} pages to books.json under key '{book_key}'.")