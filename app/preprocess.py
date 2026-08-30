"""Loading and triage: deciding whether a file has to be read by OCR at all.

Triage is the most expensive decision in the pipeline to get wrong, and it is
wrong in two directions. Treat a scan as a text PDF and we hand the engines an
empty string and return nothing. Treat a digital PDF as a scan and we pay for
OCR on text the file already carries character-perfect. Hence the deliberately
blunt rule in `_has_text_layer`: a page has to carry real characters before we
trust its text layer, because a stamp, a page number or a watermark drawn as
text is not a transcription of the document.

Pages are always rendered, text layer or not: the vision engine reads images
regardless of what the PDF claims to contain, and rasterising costs CPU rather
than inference budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES: tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
)

MIN_CHARS_PER_PAGE = 40
"""Non-whitespace characters a page must average before its text layer is used."""

# The cap belongs in app.config.settings, which carries no max_pages field yet;
# until it does, this fallback is what stops a 400-page scan from becoming 400
# OCR calls. getattr on settings, never os.getenv: config stays the only source.
_MAX_PAGES_FALLBACK = 20


@dataclass(slots=True)
class LoadedDocument:
    """A document reduced to what every engine needs: page images and any exact text."""

    filename: str
    pages: list[np.ndarray]
    embedded_text: str
    has_text_layer: bool
    is_pdf: bool
    source_page_count: int

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def truncated(self) -> bool:
        return self.source_page_count > len(self.pages)


def load_document(data: bytes, filename: str, *, max_pages: int | None = None) -> LoadedDocument:
    if not data:
        raise ValueError(f"{filename!r} is empty")

    limit = max_pages if max_pages is not None else getattr(settings, "max_pages", _MAX_PAGES_FALLBACK)
    limit = max(1, int(limit))

    is_pdf = _is_pdf(data, filename)
    if is_pdf:
        pages, page_texts, source_page_count = _load_pdf(data, filename, limit)
    else:
        pages = _load_image(data, filename, limit)
        page_texts = []
        source_page_count = len(pages)

    has_text_layer = _has_text_layer(page_texts)
    # An unusable text layer is reported as no text layer at all, so that
    # `embedded_text` is non-empty exactly when it can be trusted.
    embedded_text = "\n\n".join(text.strip() for text in page_texts).strip() if has_text_layer else ""

    logger.info(
        "loaded %s: %d page(s)%s, text_layer=%s, embedded_chars=%d",
        filename,
        len(pages),
        f" of {source_page_count}" if source_page_count > len(pages) else "",
        has_text_layer,
        len(embedded_text),
    )
    return LoadedDocument(
        filename=filename,
        pages=pages,
        embedded_text=embedded_text,
        has_text_layer=has_text_layer,
        is_pdf=is_pdf,
        source_page_count=source_page_count,
    )


def _is_pdf(data: bytes, filename: str) -> bool:
    # Magic bytes first: uploads arrive with whatever extension the user typed.
    return data[:5].startswith(b"%PDF") or filename.lower().endswith(".pdf")


def _load_pdf(data: bytes, filename: str, max_pages: int) -> tuple[list[np.ndarray], list[str], int]:
    try:
        import pymupdf as fitz
    except ImportError:  # PyMuPDF below 1.24 only ships the `fitz` name.
        import fitz

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - any failure here means "not a readable PDF"
        raise ValueError(f"could not open {filename!r} as a PDF: {exc}") from exc

    with doc:
        if doc.needs_pass and not doc.authenticate(""):
            raise ValueError(f"{filename!r} is password protected")

        source_page_count = doc.page_count
        if source_page_count > max_pages:
            logger.warning(
                "%s has %d pages; reading the first %d", filename, source_page_count, max_pages
            )

        pages: list[np.ndarray] = []
        page_texts: list[str] = []
        for index in range(min(source_page_count, max_pages)):
            page = doc[index]
            page_texts.append(page.get_text("text"))
            pages.append(_pixmap_to_bgr(page.get_pixmap(dpi=settings.pdf_dpi, alpha=False)))

    return pages, page_texts, source_page_count


def _pixmap_to_bgr(pix) -> np.ndarray:
    buffer = np.frombuffer(pix.samples, dtype=np.uint8)
    # Rows are padded to `stride`; slicing the padding off keeps odd widths honest.
    rows = buffer.reshape(pix.height, pix.stride)[:, : pix.width * pix.n]
    rgb = rows.reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        return cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _load_image(data: bytes, filename: str, max_pages: int) -> list[np.ndarray]:
    buffer = np.frombuffer(data, dtype=np.uint8)

    if filename.lower().endswith((".tif", ".tiff")):
        # Multi-page TIFFs are a normal scanner output; imdecode would silently
        # return only the first page, which is the failure mode triage exists to avoid.
        ok, frames = cv2.imdecodemulti(buffer, cv2.IMREAD_COLOR)
        if ok and frames:
            return [np.ascontiguousarray(frame) for frame in frames[:max_pages]]

    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        accepted = ", ".join(suffix.lstrip(".") for suffix in IMAGE_SUFFIXES)
        raise ValueError(f"could not decode {filename!r}; accepted formats are pdf, {accepted}")
    return [image]


def _has_text_layer(page_texts: list[str]) -> bool:
    """True when the file carries enough real text that OCR would only re-read it."""
    if not page_texts:
        return False
    characters = sum(len("".join(text.split())) for text in page_texts)
    return characters >= MIN_CHARS_PER_PAGE * len(page_texts)
