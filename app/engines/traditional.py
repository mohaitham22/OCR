"""The traditional backend: a recogniser reads pixels, rules read the recogniser.

The two stages are deliberately separable. Stage one turns a page into
`TextBlock`s -- text, a confidence and a box each -- and stage two turns those
into a document. Keeping them apart buys two things a single call cannot: the
review UI gets coordinates and per-line confidences to highlight with, and stage
two can be swapped for a model or LayoutLM without a line of stage one changing.
The seam between them is a string plus boxes, not a private handshake.

Stage two is keyword and pattern matching, not a model call: this engine makes
no network request and needs no API key, which is the whole point of calling it
"traditional" rather than a second wrapper around a vision model wearing an OCR
label. The trade is accuracy on anything that does not print the handful of
English and Arabic labels stage two looks for in the place it expects them --
`app.engines.vlm` exists for the documents that need it.

Neither recogniser is imported at module level. `paddleocr` pulls in
`paddlepaddle` and `pytesseract` needs a binary on PATH, and the app has to boot
on a machine with neither, so each is imported inside the method that uses it
and a missing one is a `RuntimeError` naming the command that fixes it.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date as date_type
from typing import Any, ClassVar

import cv2
import numpy as np

from app.config import settings
from app.engines.base import Extractor, coerce_fields
from app.schemas import ExtractionResult, TextBlock, schema_for

logger = logging.getLogger(__name__)

BACKENDS: dict[str, str] = {"paddle": "PaddleOCR", "tesseract": "Tesseract"}
DEFAULT_BACKEND = "paddle"

DEFAULT_LINE_TOLERANCE = 12.0
"""Pixels two blocks' vertical centres may differ by and still be one line."""

# Tesseract wants ISO 639-2; settings.languages speaks the two-letter codes the
# rest of the app uses. Anything already three letters passes through, so an
# unusual traineddata name can be configured without editing this map.
_TESSERACT_LANGS: dict[str, str] = {
    "ar": "ara",
    "en": "eng",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "tr": "tur",
    "ur": "urd",
    "fa": "fas",
}

_ARABIC = re.compile(
    "[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]"
)
_LATIN = re.compile("[A-Za-z]")


# --- Reading order -------------------------------------------------------


def reading_order_text(
    blocks: Sequence[TextBlock],
    line_tolerance: float = DEFAULT_LINE_TOLERANCE,
) -> str:
    """The blocks as running text, in the order a person would read them.

    Both recognisers emit detection order, which is neither reading order nor
    stable: a total detected before the line above it arrives first, and the
    text model downstream is then asked to reconstruct a document whose rows
    have been shuffled. Grouping by vertical proximity and sorting each group by
    its left edge undoes that.
    """
    if not blocks:
        return ""

    pages: dict[int, list[TextBlock]] = {}
    for block in blocks:
        pages.setdefault(block.page or 1, []).append(block)

    rendered = (_page_text(pages[page], line_tolerance) for page in sorted(pages))
    return "\n\n".join(text for text in rendered if text)


@dataclass(slots=True)
class _Line:
    """Blocks sharing a row, and the running mean of their vertical centres."""

    centre: float
    blocks: list[TextBlock] = field(default_factory=list)

    def add(self, block: TextBlock, centre: float) -> None:
        self.centre = (self.centre * len(self.blocks) + centre) / (len(self.blocks) + 1)
        self.blocks.append(block)


def _page_text(blocks: Sequence[TextBlock], line_tolerance: float) -> str:
    placed = [block for block in blocks if _centre(block) is not None]
    loose = [block for block in blocks if _centre(block) is None]

    lines: list[_Line] = []
    for block in sorted(placed, key=lambda b: (_centre(b) or 0.0, _left(b))):
        centre = _centre(block) or 0.0
        if lines and abs(centre - lines[-1].centre) <= line_tolerance:
            lines[-1].add(block, centre)
        else:
            lines.append(_Line(centre=centre, blocks=[block]))

    # Sorting a line's boxes left to right, Arabic included, is deliberate. The
    # recogniser has already emitted each box's characters in logical order --
    # an Arabic box reads right to left on the page and arrives correct -- so
    # the only thing left to order is the boxes themselves. Reversing them for
    # Arabic would break every mixed line, which is most lines here: a bill row
    # whose label is Arabic and whose amount is Latin digits would come back
    # with the amount and the label swapped, and a run of one script inside a
    # line of the other would land at the wrong end. Leave this alone.
    ordered = [" ".join(_words(line)) for line in lines]
    # A block with no box cannot be placed among the ones that have boxes, so it
    # keeps its own arrival order at the end of its page rather than being
    # dropped or guessed into a row.
    ordered.extend(text for text in (block.text.strip() for block in loose) if text)
    return "\n".join(text for text in ordered if text)


def _words(line: _Line) -> Iterator[str]:
    for block in sorted(line.blocks, key=_left):
        text = block.text.strip()
        if text:
            yield text


def _centre(block: TextBlock) -> float | None:
    box = block.bbox
    return (float(box[1]) + float(box[3])) / 2.0 if box and len(box) >= 4 else None


def _left(block: TextBlock) -> float:
    box = block.bbox
    return float(box[0]) if box and len(box) >= 4 else 0.0


# --- Rule-based structuring (stage two, no model call) --------------------
# `_structure` is the whole of stage two: keyword and pattern matching over the
# reading-order text, independent per field. It covers only the fields
# `app.validate` actually requires plus a currency and a date, because that is
# the line between "a genuinely useful offline fallback" and "a hand-rolled
# document parser" -- line items, addresses and payment details stay null here
# the same way an unreadable field stays null on the model-based path, just for
# a different reason: nothing anchored a pattern on them, not that the page did
# not carry them.

_ARABIC_DIGIT_MAP = str.maketrans(
    {
        **{chr(0x0660 + i): str(i) for i in range(10)},
        **{chr(0x06F0 + i): str(i) for i in range(10)},
        "٫": ".",
        "٬": "",
    }
)
_AMOUNT_TOKEN = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Checked first, in order: an unambiguous "this is the document's total" label.
_TOTAL_LABELS = (
    "grand total",
    "amount due",
    "total due",
    "balance due",
    "الإجمالي",
    "المبلغ الإجمالي",
)
# Checked only if nothing above matched, and only on a line that does not also
# read as a subtotal -- "total" alone is a substring of "subtotal", so the bare
# label has to be paired with an exclusion or it finds the wrong row.
_TOTAL_LABELS_LOOSE = ("total", "المجموع")
_SUBTOTAL_MARKERS = ("subtotal", "sub total", "sub-total", "المجموع الفرعي")

_INVOICE_NUMBER_LABELS = (
    "invoice no",
    "invoice number",
    "invoice #",
    "inv no",
    "inv#",
    "فاتورة رقم",
    "رقم الفاتورة",
)

_CURRENCY_CODE = re.compile(
    r"\b(SAR|AED|EGP|USD|EUR|GBP|KWD|QAR|BHD|OMR|JOD|SDG|SYP|IQD|LYD|MAD|TND|DZD|LBP)\b"
)
_CURRENCY_SYMBOLS: dict[str, str] = {"$": "USD", "€": "EUR", "£": "GBP", "﷼": "SAR"}

# Year-first is checked before day-first, since a 4-digit year on either side of
# a separator is unambiguous where DD/MM/YYYY and MM/DD/YYYY are not. The
# system prompt's own rule -- an ambiguous all-numeric date reads day-month-year
# -- is what _DATE_DMY encodes for the fallback.
_DATE_ISO = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")
_DATE_DMY = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b")


def _normalise_digits(text: str) -> str:
    return text.translate(_ARABIC_DIGIT_MAP)


def _amount_on_line(line: str) -> str | None:
    matches = _AMOUNT_TOKEN.findall(_normalise_digits(line))
    return matches[-1] if matches else None


def _find_amount(
    text: str,
    exact_labels: Sequence[str],
    loose_labels: Sequence[str] = (),
    exclude: Sequence[str] = (),
) -> str | None:
    """The amount beside the most convincing label, read from the bottom up.

    A grand total conventionally sits below its line items and its subtotal,
    so the last matching line in the document is likelier to be the one that
    means the whole document rather than one row of it.
    """
    lines = list(reversed(text.splitlines()))
    for raw_line in lines:
        folded = raw_line.casefold()
        if any(label in folded for label in exact_labels):
            amount = _amount_on_line(raw_line)
            if amount:
                return amount
    for raw_line in lines:
        folded = raw_line.casefold()
        if any(marker in folded for marker in exclude):
            continue
        if any(label in folded for label in loose_labels):
            amount = _amount_on_line(raw_line)
            if amount:
                return amount
    return None


def _find_label_value(text: str, labels: Sequence[str]) -> str | None:
    """Whatever follows a label on its own line, after a colon, dash or run of spaces."""
    for raw_line in text.splitlines():
        folded = raw_line.casefold()
        for label in labels:
            index = folded.find(label)
            if index == -1:
                continue
            remainder = raw_line[index + len(label) :].strip(" \t:#-–—.")
            if remainder:
                return remainder
    return None


def _valid_date(year: int, month: int, day: int) -> str | None:
    try:
        return date_type(year, month, day).isoformat()
    except ValueError:
        return None


def _find_date(text: str) -> str | None:
    normalised = _normalise_digits(text)
    iso = _DATE_ISO.search(normalised)
    if iso:
        year, month, day = (int(part) for part in iso.groups())
        found = _valid_date(year, month, day)
        if found:
            return found
    dmy = _DATE_DMY.search(normalised)
    if dmy:
        day_s, month_s, year_s = dmy.groups()
        year = int(year_s) if len(year_s) == 4 else 2000 + int(year_s) if int(year_s) < 70 else 1900 + int(year_s)
        found = _valid_date(year, int(month_s), int(day_s))
        if found:
            return found
    return None


def _find_currency(text: str) -> str | None:
    match = _CURRENCY_CODE.search(text.upper())
    if match:
        return match.group(1)
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code
    return None


def _first_named_line(text: str) -> tuple[str | None, str | None]:
    """The first non-blank line, as (latin_name, arabic_name).

    Receipts and invoices both put the issuing party's name at the top of the
    page, well before any label a pattern could anchor on -- a shop or a
    letterhead is printed, not labelled "merchant name:".
    """
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        return (None, line) if _script(line) == "ar" else (line, None)
    return None, None


def _structure(doc_type: str, source_text: str) -> dict[str, Any]:
    """Field values pulled from `source_text` by keyword and pattern matching alone.

    No model call and no whole-document understanding: each field is found (or
    not) independently. Good enough for the fields `app.validate` requires;
    everything else is left null rather than guessed, same rule the model-based
    engines are held to.
    """
    if doc_type == "receipt":
        name, name_ar = _first_named_line(source_text)
        return {
            "merchant_name": name,
            "merchant_name_ar": name_ar,
            "date": _find_date(source_text),
            "currency": _find_currency(source_text),
            "total": _find_amount(source_text, _TOTAL_LABELS, _TOTAL_LABELS_LOOSE, _SUBTOTAL_MARKERS),
        }
    if doc_type == "invoice":
        name, name_ar = _first_named_line(source_text)
        return {
            "invoice_number": _find_label_value(source_text, _INVOICE_NUMBER_LABELS),
            "vendor_name": name,
            "vendor_name_ar": name_ar,
            "issue_date": _find_date(source_text),
            "currency": _find_currency(source_text),
            "total": _find_amount(source_text, _TOTAL_LABELS, _TOTAL_LABELS_LOOSE, _SUBTOTAL_MARKERS),
        }
    # "document": no required field and no reliable keyword to anchor a title or
    # an issuer on, so the one thing worth returning is the transcription
    # itself -- a lossless copy, not a guess.
    return {"full_text": source_text}


# --- The engine ----------------------------------------------------------


class TraditionalExtractor(Extractor):
    """OCR, then a text model.

    `key` and `label` are set per instance because the backend is part of the
    identity: a compare-mode run of Paddle against Tesseract is two entries in
    the registry, not one. The class-level values exist so the base's
    `__init_subclass__` check passes at import and so an unparameterised
    `traditional` still names something.
    """

    key: ClassVar[str] = "traditional"
    label: ClassVar[str] = "Traditional OCR"
    gives_boxes: ClassVar[bool] = True

    def __init__(self, backend: str | None = None) -> None:
        chosen = str(backend or settings.ocr_backend or DEFAULT_BACKEND).strip().lower()
        if chosen not in BACKENDS:
            raise ValueError(f"unknown OCR backend {backend!r}; expected one of {sorted(BACKENDS)}")
        self.backend = chosen
        self.key = f"traditional:{chosen}"
        self.label = f"Traditional OCR ({BACKENDS[chosen]})"

    def extract(
        self,
        pages: Sequence[np.ndarray],
        doc_type: str,
        embedded_text: str | None = None,
    ) -> ExtractionResult:
        schema_for(doc_type)  # a bad doc_type is a caller bug; fail before the clock starts

        with self._timed() as elapsed:
            blocks: list[TextBlock] = []
            used_text_layer = bool(embedded_text and embedded_text.strip())

            if used_text_layer:
                # The PDF's own text is exact. Rendering it to pixels and reading
                # the pixels back can only lose characters, never gain one, so
                # the recogniser is not run at all.
                source_text = str(embedded_text).strip()
            else:
                try:
                    blocks = self._recognise(pages)
                except Exception:
                    # A backend that will not load or will not read is this
                    # engine failing, and per the `Extractor.extract` contract
                    # that comes back as an empty result: in compare mode the
                    # other engine still has an answer to give.
                    logger.exception("%s failed to read %d page(s)", self.key, len(pages))
                    return self._empty(doc_type, pages, elapsed())
                source_text = reading_order_text(blocks)

            logger.info(
                "%s read %s: %d block(s), %d chars, text_layer=%s",
                self.key,
                doc_type,
                len(blocks),
                len(source_text),
                used_text_layer,
            )

            if not source_text:
                logger.warning("%s found no text on %d page(s)", self.key, len(pages))
                return self._empty(doc_type, pages, elapsed(), blocks=blocks)

            try:
                raw = _structure(doc_type, source_text)
            except Exception:
                logger.exception("%s could not structure %s", self.key, doc_type)
                return self._empty(doc_type, pages, elapsed(), blocks=blocks, raw_text=source_text)

            document, issues = coerce_fields(doc_type, raw)
            for issue in issues:
                # `ExtractionResult` has nowhere to carry these yet, so logging
                # is all an engine can do with them. See CLAUDE.md.
                logger.warning("%s: %s", self.key, issue.message)

            return ExtractionResult(
                doc_type=doc_type,
                document=document,
                engine=self.key,
                text_blocks=blocks,
                raw_text=source_text,
                # No recogniser ran on the text-layer path, so there is no OCR
                # confidence to report -- and 1.0 would be a claim about the
                # extraction, not about the transcription.
                confidence=None if used_text_layer else _overall_confidence(blocks),
                page_count=len(pages),
                duration_ms=elapsed(),
            )

    # --- Recognition -----------------------------------------------------

    def _recognise(self, pages: Sequence[np.ndarray]) -> list[TextBlock]:
        read = self._read_paddle if self.backend == "paddle" else self._read_tesseract
        floor = float(getattr(settings, "ocr_min_confidence", 0.0) or 0.0)

        blocks: list[TextBlock] = []
        dropped = 0
        for number, page in enumerate(pages, start=1):
            for block in read(page, number):
                if not block.text.strip():
                    continue
                if block.confidence is not None and block.confidence < floor:
                    dropped += 1
                    continue
                blocks.append(block)

        if dropped:
            logger.info("%s dropped %d block(s) below confidence %.2f", self.key, dropped, floor)
        return blocks

    def _read_paddle(self, page: np.ndarray, number: int) -> Iterator[TextBlock]:
        reader = _paddle_reader(_paddle_lang())
        raw = reader.predict(page) if hasattr(reader, "predict") else reader.ocr(page)
        for text, score, poly in _paddle_lines(raw):
            yield _block(text, score, _bbox(poly), number)

    def _read_tesseract(self, page: np.ndarray, number: int) -> Iterator[TextBlock]:
        pytesseract = _require_pytesseract()
        # pytesseract hands a bare array to PIL, which reads it as RGB; OpenCV
        # pages are BGR, so the channels would arrive swapped.
        rgb = cv2.cvtColor(page, cv2.COLOR_BGR2RGB) if page.ndim == 3 else page
        try:
            data = pytesseract.image_to_data(
                rgb,
                lang=_tesseract_lang(),
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "tesseract is not on PATH; install it and set TESSERACT_CMD to its "
                "executable, or run with OCR_BACKEND=paddle"
            ) from exc

        for text, score, box in _tesseract_lines(data):
            yield _block(text, score, box, number)

    # --- Results ---------------------------------------------------------

    def _empty(
        self,
        doc_type: str,
        pages: Sequence[np.ndarray],
        duration_ms: float,
        *,
        blocks: list[TextBlock] | None = None,
        raw_text: str | None = None,
    ) -> ExtractionResult:
        """A failure, in the shape every caller already knows how to read."""
        return ExtractionResult(
            doc_type=doc_type,
            document=None,
            engine=self.key,
            text_blocks=blocks or [],
            raw_text=raw_text,
            confidence=_overall_confidence(blocks or []),
            page_count=len(pages),
            duration_ms=duration_ms,
        )


# --- PaddleOCR -----------------------------------------------------------
# One reader per language, because construction loads detection, recognition
# and angle models from disk and costs seconds; the request path must not pay
# that twice. The lock is what stops two Streamlit or FastAPI threads building
# the same reader at once and throwing one of them away.

_PADDLE_READERS: dict[str, Any] = {}
_PADDLE_LOCK = threading.Lock()


def _paddle_lang() -> str:
    """Paddle takes one language, and its Arabic model already reads Latin digits."""
    return "arabic" if "ar" in settings.languages else "en"


def _paddle_reader(lang: str) -> Any:
    reader = _PADDLE_READERS.get(lang)
    if reader is not None:
        return reader

    with _PADDLE_LOCK:
        if lang in _PADDLE_READERS:
            return _PADDLE_READERS[lang]
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "paddleocr is not installed; run `pip install paddleocr paddlepaddle` "
                "or run with OCR_BACKEND=tesseract"
            ) from exc

        logger.info("loading PaddleOCR models for lang=%s", lang)
        _PADDLE_READERS[lang] = _construct_paddle(PaddleOCR, lang)
        return _PADDLE_READERS[lang]


def _construct_paddle(cls: Any, lang: str) -> Any:
    """Build a reader across PaddleOCR's incompatible constructor generations.

    2.x wants `use_angle_cls` and `show_log`; 3.x removed both and rejects an
    unknown keyword outright. Only that rejection is caught -- a constructor
    that fails because the models will not download is a real failure and has to
    surface as one.
    """
    attempts = (
        {"lang": lang, "use_angle_cls": True, "show_log": False},
        {"lang": lang, "use_angle_cls": True},
        {"lang": lang},
    )
    last: Exception | None = None
    for kwargs in attempts:
        try:
            return cls(**kwargs)
        except (TypeError, ValueError) as exc:
            last = exc
            logger.debug("PaddleOCR rejected %s: %s", sorted(kwargs), exc)
    raise RuntimeError(f"could not construct PaddleOCR for lang={lang!r}: {last}")


def _paddle_lines(raw: Any) -> Iterator[tuple[str, float | None, Any]]:
    """Text, score and polygon out of whichever result shape this Paddle returns.

    3.x returns one dict per image with parallel `rec_*` lists; 2.x returns one
    list per image of `[polygon, (text, score)]`, and older builds return that
    list unwrapped. All three reach here, and none of them is worth making the
    caller think about.
    """
    for result in _pages_of(raw):
        if isinstance(result, dict):
            texts = result.get("rec_texts") or []
            scores = result.get("rec_scores") or []
            polys = result.get("rec_polys")
            if polys is None:
                polys = result.get("dt_polys") or []
            for index, text in enumerate(texts):
                yield str(text), _score(_at(scores, index)), _at(polys, index)
            continue
        for entry in result or ():
            parsed = _paddle_entry(entry)
            if parsed is not None:
                yield parsed


def _pages_of(raw: Any) -> Iterator[Any]:
    items = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    if items and _paddle_entry(items[0]) is not None:
        yield items  # an unwrapped single page: the items are its own lines
        return
    for item in items:
        if item is not None:
            yield item


def _paddle_entry(entry: Any) -> tuple[str, float | None, Any] | None:
    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        return None
    poly, payload = entry[0], entry[1]
    if isinstance(payload, (list, tuple)) and payload and isinstance(payload[0], str):
        return payload[0], _score(payload[1] if len(payload) > 1 else None), poly
    if isinstance(payload, str):
        return payload, _score(entry[2] if len(entry) > 2 else None), poly
    return None


# --- Tesseract -----------------------------------------------------------


def _tesseract_lang() -> str:
    codes = [_TESSERACT_LANGS.get(code, code) for code in settings.languages if code]
    return "+".join(dict.fromkeys(codes)) or "eng"


def _require_pytesseract() -> Any:
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract is not installed; run `pip install pytesseract` and install the "
            "tesseract binary, or run with OCR_BACKEND=paddle"
        ) from exc

    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = str(settings.tesseract_cmd)
    return pytesseract


def _tesseract_lines(data: dict[str, Any]) -> Iterator[tuple[str, float | None, list[float]]]:
    """Words regrouped into lines, so both backends emit the same unit.

    Tesseract reports one row per word and Paddle one per line. Merging here
    keeps `reading_order_text` and the review UI from having to care which
    backend ran. Words are joined in Tesseract's own word order rather than by
    position, because within a line that order is already the logical one --
    which is what an Arabic line needs.
    """
    rows = zip(
        data.get("block_num", []),
        data.get("par_num", []),
        data.get("line_num", []),
        data.get("word_num", []),
        data.get("text", []),
        data.get("conf", []),
        data.get("left", []),
        data.get("top", []),
        data.get("width", []),
        data.get("height", []),
    )

    lines: dict[tuple[int, int, int], list[tuple[int, str, float | None, list[float]]]] = {}
    for block, par, line, word, text, conf, left, top, width, height in rows:
        if not str(text).strip():
            continue
        score = _score(conf)
        if score is not None and score < 0:
            continue  # tesseract's -1: a box it found no text in
        box = [float(left), float(top), float(left) + float(width), float(top) + float(height)]
        lines.setdefault((int(block), int(par), int(line)), []).append(
            (int(word), str(text).strip(), score, box)
        )

    for key in sorted(lines):
        words = sorted(lines[key])
        scores = [word[2] for word in words if word[2] is not None]
        yield (
            " ".join(word[1] for word in words),
            sum(scores) / len(scores) if scores else None,
            _union([word[3] for word in words]),
        )


def _union(boxes: Sequence[Sequence[float]]) -> list[float]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


# --- Shared helpers ------------------------------------------------------


def _block(text: str, score: float | None, bbox: list[float] | None, page: int) -> TextBlock:
    return TextBlock(text=text, confidence=score, bbox=bbox, page=page, language=_script(text))


def _script(text: str) -> str | None:
    arabic = bool(_ARABIC.search(text))
    latin = bool(_LATIN.search(text))
    if arabic and latin:
        return "mixed"
    if arabic:
        return "ar"
    return "en" if latin else None


def _score(value: Any) -> float | None:
    """Confidences arrive as 0-1 from Paddle and as 0-100 from Tesseract."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number / 100.0 if number > 1.0 else number


def _bbox(poly: Any) -> list[float] | None:
    """A quadrilateral -- or an already-flat box -- as [x0, y0, x1, y1]."""
    if poly is None:
        return None
    try:
        points = np.asarray(poly, dtype=float)
    except (TypeError, ValueError):
        return None
    if points.ndim == 1 and points.size == 4:
        x0, y0, x1, y1 = points.tolist()
        return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 2:
        return None
    xs, ys = points[:, 0], points[:, 1]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _at(values: Any, index: int) -> Any:
    try:
        return values[index]
    except (IndexError, KeyError, TypeError):
        return None


def _overall_confidence(blocks: Sequence[TextBlock]) -> float | None:
    """Mean block confidence, weighted by how much text each block carries.

    An unweighted mean lets one misread character count as much as the line
    carrying the total, which is backwards: the long blocks are the document.
    """
    weighted = [
        (block.confidence, len(block.text.strip()))
        for block in blocks
        if block.confidence is not None and block.text.strip()
    ]
    if not weighted:
        return None
    total = sum(weight for _, weight in weighted)
    return round(sum(score * weight for score, weight in weighted) / total, 4)
