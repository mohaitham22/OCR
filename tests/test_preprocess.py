"""Triage tests. Every fixture is built in memory: a PDF that reads as a scan is
the failure mode under test, so the bytes have to be constructed, not trusted.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.config import settings
from app.preprocess import MIN_CHARS_PER_PAGE, load_document

try:
    import pymupdf as fitz
except ImportError:  # PyMuPDF below 1.24 only ships the `fitz` name.
    import fitz


def _page_image(width: int = 800, height: int = 1000) -> np.ndarray:
    """A page that looks scanned: ink on paper, no character codes anywhere."""
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    for offset in range(120, height - 120, 90):
        cv2.line(image, (80, offset), (width - 80, offset), (30, 30, 30), 6)
    cv2.rectangle(image, (60, 60), (width - 60, height - 60), (0, 0, 0), 3)
    return image


def _text_pdf(lines: list[str], pages: int = 1) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_text((72, 120), lines, fontsize=12)
    return doc.tobytes()


def _scan_pdf(pages: int = 1) -> bytes:
    ok, encoded = cv2.imencode(".png", _page_image())
    assert ok
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page()
        page.insert_image(page.rect, stream=encoded.tobytes())
    return doc.tobytes()


# --- The two directions triage must not confuse ---------------------------


def test_pdf_with_text_layer_is_read_not_ocred() -> None:
    data = _text_pdf(["INVOICE INV-2024-0198", "Vendor: Nile Trading Company", "Total: SAR 4,312.50"])

    loaded = load_document(data, "invoice.pdf")

    assert loaded.has_text_layer is True
    assert "INV-2024-0198" in loaded.embedded_text
    assert "Nile Trading Company" in loaded.embedded_text
    assert loaded.page_count == 1


def test_image_only_pdf_falls_through_to_ocr_with_a_rendered_page() -> None:
    loaded = load_document(_scan_pdf(), "scan.pdf")

    assert loaded.has_text_layer is False
    assert loaded.embedded_text == ""
    assert loaded.page_count == 1

    page = loaded.pages[0]
    assert page.ndim == 3 and page.shape[2] == 3
    assert page.dtype == np.uint8
    assert page.std() > 0, "rendered page is blank"


def test_text_layer_pdf_is_also_rendered_for_the_vision_engine() -> None:
    loaded = load_document(_text_pdf(["Rendered as well as read"] * 4), "invoice.pdf")

    assert loaded.pages[0].std() > 0


# --- Where the threshold sits ---------------------------------------------


def test_pdf_of_blank_lines_is_not_a_text_layer() -> None:
    loaded = load_document(_text_pdf(["   ", "\t", "", "     "]), "blank.pdf")

    assert loaded.has_text_layer is False
    assert loaded.embedded_text == ""


def test_stray_stamp_text_is_not_a_text_layer() -> None:
    loaded = load_document(_text_pdf(["Page 1 of 1", "COPY"]), "stamped.pdf")

    assert loaded.has_text_layer is False
    assert loaded.embedded_text == ""


def test_threshold_counts_non_whitespace_characters() -> None:
    just_under = _text_pdf(["x " * (MIN_CHARS_PER_PAGE - 1)])
    just_over = _text_pdf(["x " * MIN_CHARS_PER_PAGE])

    assert load_document(just_under, "under.pdf").has_text_layer is False
    assert load_document(just_over, "over.pdf").has_text_layer is True


def test_threshold_is_per_page_not_per_document() -> None:
    body = ["y " * MIN_CHARS_PER_PAGE]

    doc = fitz.open()
    doc.new_page().insert_text((72, 120), body, fontsize=12)
    doc.new_page()  # a second page carrying nothing halves the average
    mixed = doc.tobytes()

    assert load_document(_text_pdf(body, pages=2), "both.pdf").has_text_layer is True
    assert load_document(mixed, "half.pdf").has_text_layer is False


# --- Rendering and the page cap -------------------------------------------


def test_pages_are_rendered_at_the_configured_dpi() -> None:
    loaded = load_document(_scan_pdf(), "scan.pdf")

    expected_height = 842 / 72 * settings.pdf_dpi  # PyMuPDF's default page is A4
    assert loaded.pages[0].shape[0] == pytest.approx(expected_height, abs=2)


def test_long_documents_are_capped_and_report_the_true_length() -> None:
    loaded = load_document(_scan_pdf(pages=5), "long.pdf", max_pages=2)

    assert loaded.page_count == 2
    assert loaded.source_page_count == 5
    assert loaded.truncated is True


def test_short_documents_are_not_reported_as_truncated() -> None:
    loaded = load_document(_scan_pdf(pages=2), "short.pdf", max_pages=5)

    assert loaded.page_count == 2
    assert loaded.source_page_count == 2
    assert loaded.truncated is False


# --- Images ---------------------------------------------------------------


def test_image_upload_becomes_a_single_bgr_page() -> None:
    ok, encoded = cv2.imencode(".png", _page_image())
    assert ok

    loaded = load_document(encoded.tobytes(), "receipt.png")

    assert loaded.is_pdf is False
    assert loaded.has_text_layer is False
    assert loaded.embedded_text == ""
    assert loaded.page_count == 1
    assert loaded.pages[0].shape[2] == 3


def test_multi_page_tiff_keeps_every_page() -> None:
    ok, encoded = cv2.imencodemulti(".tiff", [_page_image(), _page_image()])
    if not ok:  # pragma: no cover - depends on the OpenCV build's TIFF support
        pytest.skip("this OpenCV build cannot encode multi-page TIFF")

    loaded = load_document(encoded.tobytes(), "scan.tiff")

    assert loaded.page_count == 2


def test_extension_does_not_override_the_bytes() -> None:
    loaded = load_document(_text_pdf(["Filenames lie; magic bytes do not."] * 2), "upload.png")

    assert loaded.is_pdf is True
    assert loaded.has_text_layer is True


# --- Failure modes --------------------------------------------------------


def test_undecodable_image_names_the_accepted_formats() -> None:
    with pytest.raises(ValueError) as excinfo:
        load_document(b"this is not an image", "notes.png")

    message = str(excinfo.value)
    assert "notes.png" in message
    assert "pdf" in message and "png" in message and "tiff" in message


def test_unreadable_pdf_raises_value_error() -> None:
    with pytest.raises(ValueError, match="broken.pdf"):
        load_document(b"%PDF-1.7 truncated before anything useful", "broken.pdf")


def test_empty_upload_raises_value_error() -> None:
    with pytest.raises(ValueError, match="nothing.pdf"):
        load_document(b"", "nothing.pdf")
