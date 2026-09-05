from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import ocr_service


def test_ocr_can_be_disabled_for_text_extraction(monkeypatch) -> None:
    monkeypatch.setenv("OCR_ENABLED", "false")
    fake_page = type("Page", (), {"extract_text": lambda self: "embedded text"})()
    fake_reader = type("Reader", (), {"pages": [fake_page]})
    with patch("pypdf.PdfReader", return_value=fake_reader):
        pages, ocr_pages = ocr_service.extract_pdf_pages(b"%PDF-test")
    assert pages == ["embedded text"]
    assert ocr_pages == []


def test_ocr_fallback_is_used_only_for_empty_pages(monkeypatch) -> None:
    monkeypatch.setenv("OCR_ENABLED", "true")
    text_page = type("Page", (), {"extract_text": lambda self: "normal text"})()
    image_page = type("Page", (), {"extract_text": lambda self: ""})()
    fake_reader = type("Reader", (), {"pages": [text_page, image_page]})
    with patch("pypdf.PdfReader", return_value=fake_reader), patch(
        "ocr_service.extract_page_text_with_ocr", return_value="ocr text"
    ) as ocr:
        pages, ocr_pages = ocr_service.extract_pdf_pages(b"%PDF-test")
    assert pages == ["normal text", "ocr text"]
    assert ocr_pages == [2]
    ocr.assert_called_once_with(image_page)


def test_ocr_runtime_status_is_explicit_when_tesseract_is_missing(monkeypatch) -> None:
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(ocr_service.shutil, "which", lambda _: None)
    status = ocr_service.ocr_runtime_status()
    assert status["ready"] is False
    assert status["reason"] == "tesseract_executable_missing"
