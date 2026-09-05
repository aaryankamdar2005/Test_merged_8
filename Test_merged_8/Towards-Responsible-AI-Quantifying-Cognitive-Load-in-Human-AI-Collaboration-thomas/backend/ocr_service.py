"""Local OCR fallback for scanned PDF pages.

Normal PDFs are handled by pypdf. This module is used only for pages where
normal text extraction produced no usable text, which keeps ingestion fast
for born-digital documents and supports scanned assessment material.
"""

from __future__ import annotations

import io
import os
import shutil
from typing import Any


class OCRUnavailableError(RuntimeError):
    """Raised when OCR was requested but its local dependencies are missing."""


def ocr_runtime_status() -> dict[str, Any]:
    """Report whether the Python wrapper and Tesseract executable are usable."""
    try:
        import fitz  # noqa: F401
        import pytesseract
        from PIL import Image  # noqa: F401
    except ImportError:
        return {"enabled": True, "ready": False, "reason": "python_dependencies_missing"}
    configured = os.getenv("TESSERACT_CMD", "").strip()
    executable = configured or shutil.which("tesseract")
    if not executable:
        return {"enabled": True, "ready": False, "reason": "tesseract_executable_missing"}
    if configured:
        pytesseract.pytesseract.tesseract_cmd = configured
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return {"enabled": True, "ready": False, "reason": "tesseract_not_runnable"}
    return {"enabled": True, "ready": True, "reason": "ready"}


def extract_page_text_with_ocr(page: Any) -> str:
    """Render one PDF page and extract text with local Tesseract OCR."""
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise OCRUnavailableError(
            "OCR dependencies are not installed. Install PyMuPDF, Pillow, "
            "and pytesseract."
        ) from exc

    tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        dpi = max(120, min(int(os.getenv("OCR_DPI", "200")), 400))
        language = os.getenv("OCR_LANGUAGE", "eng").strip() or "eng"
        scale = dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        return pytesseract.image_to_string(image, lang=language).strip()
    except Exception as exc:
        raise OCRUnavailableError(f"Local OCR failed: {exc}") from exc


def extract_pdf_pages(payload: bytes) -> tuple[list[str], list[int]]:
    """Extract normal text and OCR fallback text from a PDF payload.

    Returns (page_text, ocr_page_numbers). OCR is enabled by default and can
    be disabled with OCR_ENABLED=false when the deployment has no Tesseract.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise OCRUnavailableError("pypdf is required for PDF ingestion") from exc

    reader = PdfReader(io.BytesIO(payload))
    pages: list[str] = []
    ocr_pages: list[int] = []
    ocr_enabled = os.getenv("OCR_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
    for page_number, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if text or not ocr_enabled:
            pages.append(text)
            continue
        text = extract_page_text_with_ocr(page)
        pages.append(text)
        if text:
            ocr_pages.append(page_number)
    return pages, ocr_pages
