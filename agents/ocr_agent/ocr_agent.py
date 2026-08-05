import os
import re

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


def extract_from_pdf(path):

    """
    Handle:

    1. Text PDFs
    2. Scanned PDFs
    3. Hybrid PDFs

    Logic:

    Text exists?
        YES -> Direct extraction
        NO  -> OCR
    """

    # Verify PyMuPDF
    if not fitz:
        return (
            "[PyMuPDF not available "
            "- install pymupdf]"
        )

    # Open PDF
    doc = fitz.open(path)

    # Store complete extracted text
    full_text = ""

    # Loop through all pages
    for page in doc:

        # Extract page text
        text = page.get_text().strip()

        # -------------------------------------
        # DIGITAL PAGE
        # -------------------------------------
        if len(text) > 50:

            # Directly use extracted text
            full_text += "\n" + text

        # -------------------------------------
        # SCANNED PAGE
        # -------------------------------------
        else:

            # OCR available?
            if pytesseract:

                # Convert page into image
                image = page.get_pixmap(dpi=300)

                # Convert Pixmap -> PIL Image
                img = Image.frombytes(
                    "RGB",
                    (image.width, image.height),
                    image.samples
                )

                # OCR extraction
                ocr_text = (
                    pytesseract.image_to_string(
                        img
                    )
                )

                # Append OCR result
                full_text += "\n" + ocr_text

            else:

                # OCR library missing
                full_text += (
                    "\n[Scanned page - OCR not available]"
                )

    # Close PDF
    doc.close()

    # Return extracted content
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