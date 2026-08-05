"""
OCR + Text Extraction Agent
Owner: Bikkini Vasanth Kumar (Day-1)

Job: detect whether a file is scanned or digital, extract text
(OCR for scanned/images, direct read for digital), and clean it.
Handles hybrid PDFs page-by-page, plus DOCX and TXT files.

NOTE: OCR needs Tesseract installed on the system:
    Ubuntu:  sudo apt install tesseract-ocr
    Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
Python libs: pip install pytesseract pillow pymupdf python-docx
If those libraries are missing, the functions fall back gracefully.
"""

import os
import re

# Optional imports - the code still runs if they are not installed yet
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pytesseract
    from PIL import Image
    import io
except ImportError:
    pytesseract = None

try:
    import docx  # python-docx for .docx files
except ImportError:
    docx = None


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tiff", ".bmp")


def clean_text(text: str) -> str:
    """Remove OCR / parsing noise -> neat readable text."""
    text = re.sub(r"-\n", "", text)          # join broken words
    text = re.sub(r"\n{2,}", "\n", text)     # remove extra blank lines
    text = re.sub(r"[ \t]{2,}", " ", text)   # remove extra spaces
    text = re.sub(r"[•♦]+", "", text)        # remove junk symbols
    return text.strip()


def ocr_image(path: str) -> str:
    """Read text from an image file using OCR."""
    if not pytesseract:
        return "[OCR not available - install pytesseract & pillow]"
    img = Image.open(path)
    return pytesseract.image_to_string(img)


def parse_docx(path: str) -> str:
    """Read text from a .docx file using python-docx."""
    if not docx:
        return "[python-docx not available - run: pip install python-docx]"
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)


def extract_from_pdf(path: str) -> str:
    """
    Handle digital, scanned, and HYBRID PDFs.
    Checks each page: has text -> parse it, no text -> OCR it.
    """
    if not fitz:
        return "[PyMuPDF not available - install pymupdf]"

    doc = fitz.open(path)
    full_text = ""

    for i, page in enumerate(doc):
        text = page.get_text().strip()

        if len(text) > 50:
            # digital page -> use text directly
            full_text += f"\n{text}"
        else:
            # scanned page -> render to image and OCR it
            if pytesseract:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes()))
                full_text += f"\n{pytesseract.image_to_string(img)}"
            else:
                full_text += "\n[Scanned page - OCR not available]"

    return full_text


def extract_text(saved_path: str) -> str:
    """
    Main entry point. Detects the file type and returns CLEAN text.
    - image        -> OCR
    - pdf          -> per-page (digital + scanned handled)
    - docx         -> python-docx
    - txt          -> read directly
    """
    ext = os.path.splitext(saved_path)[1].lower()

    if ext in IMAGE_EXTS:
        raw = ocr_image(saved_path)          # image => always OCR
    elif ext == ".pdf":
        raw = extract_from_pdf(saved_path)   # pdf => detect per page
    elif ext == ".docx":
        raw = parse_docx(saved_path)         # docx => python-docx
    elif ext == ".txt":
        with open(saved_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()                   # txt => read directly
    else:
        raw = ""                             # unknown type

    return clean_text(raw)