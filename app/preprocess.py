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


# ---------------------------------------------------------------------------
# Geometry and lighting
#
# Recognisers lose more accuracy to skew than to resolution: a receipt shot 8
# degrees off axis is worse input than the same receipt scanned at half the
# DPI. So the order below is deliberate — find the page, straighten it, then
# fix the light — and every step is allowed to decline. A correction applied to
# a page that did not need it is a net loss, which is why each function returns
# its input unchanged rather than guessing.
# ---------------------------------------------------------------------------

_WORK_SIDE = 800
"""Longest edge of the downscaled copy used to *measure* skew and page edges.

Contours and angles are scale-invariant; running the search at full resolution
buys nothing and costs seconds on a 300-DPI page.
"""

_FULL_FRAME_RATIO = 0.98
"""Above this share of the frame the "page" is the frame: a flatbed scan."""

_MIN_BACKGROUND_CONTRAST = 20.0
"""Grey levels between page and surroundings before a quad counts as a page edge."""

_MIN_SKEW_DEGREES = 0.3
"""Below this the rotation costs more in resampling blur than it recovers."""

_MIN_INK_PIXELS = 200
"""Fewer ink pixels than this and minAreaRect is fitting noise."""

_MAX_INK_COVERAGE = 0.60
"""Above this the ink is a dark photo or an inverted scan, not text."""


def crop_to_document(image: np.ndarray, min_area_ratio: float = 0.25) -> np.ndarray:
    """Warp the page out of the scene it was photographed in, or pass it through.

    A phone photo puts the document at an angle inside a desk; a flatbed scan
    is already the whole frame, and cropping that is all downside. Anything
    short of a convincing page — one convex four-gon covering at least
    `min_area_ratio` of the frame and less than all of it — is left alone.
    """
    height, width = image.shape[:2]
    if height < 40 or width < 40:
        return image

    scale = min(1.0, _WORK_SIDE / float(max(height, width)))
    gray = _to_gray(_scaled(image, scale))

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    # Dilate before contouring: a page edge crossing a low-contrast patch of
    # background arrives in fragments that findContours would never join.
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest, 0.02 * cv2.arcLength(largest, True), True)
    # Convexity is the cheap guard against a dilated block of text passing for
    # a page: a text blob approximates to four points often enough to matter.
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        logger.debug("crop: largest contour is not a convex quad; leaving the frame alone")
        return image

    area_ratio = abs(cv2.contourArea(approx)) / float(gray.shape[0] * gray.shape[1])
    if area_ratio < min_area_ratio:
        logger.debug("crop: quad covers %.2f of the frame, below %.2f", area_ratio, min_area_ratio)
        return image
    if area_ratio > _FULL_FRAME_RATIO:
        logger.debug("crop: quad is the whole frame; a warp here would only resample it")
        return image

    # Last guard, and the one that saves flatbed scans: a photographed page has
    # a *scene* around it — desk, hand, shadow — while a scan has more of the
    # same paper. If what surrounds the quad is the same tone as what is inside
    # it, this is a border printed on the document, not the edge of the
    # document, and cropping to it would throw away letterhead and footer.
    quad_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(quad_mask, approx.reshape(4, 2).astype(np.int32), 255)
    outside = gray[quad_mask == 0]
    if outside.size:
        contrast = abs(float(np.median(gray[quad_mask == 255])) - float(np.median(outside)))
        if contrast < _MIN_BACKGROUND_CONTRAST:
            logger.debug("crop: background matches the page (%.0f levels); this is a printed border", contrast)
            return image

    quad = approx.reshape(4, 2).astype(np.float32) / scale
    cropped = _four_point_transform(image, quad)
    logger.debug("crop: %dx%d -> %dx%d", width, height, cropped.shape[1], cropped.shape[0])
    return cropped


def deskew(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Rotate the page so its lines of text run horizontally.

    The estimate is a minAreaRect over ink pixels, which is only text as long
    as the tilt is small. Past `max_angle` the box is being set by a border, a
    logo or a table rule, so the safe reading of a large angle is "not skew".
    """
    angle = _ink_angle(image)
    if angle is None:
        return image
    if abs(angle) < _MIN_SKEW_DEGREES:
        return image
    if abs(angle) > max_angle:
        logger.debug("deskew: %.2f deg exceeds %.1f; treating it as content, not skew", angle, max_angle)
        return image

    logger.debug("deskew: rotating by %.2f deg", angle)
    return _rotate(image, angle)


def normalise_lighting(image: np.ndarray) -> np.ndarray:
    """Flatten shadow and gradient so ink reads the same across the page."""
    gray = _to_gray(image)

    # Dividing by a heavy median blur removes the paper — the blur *is* the
    # local paper colour — so a shadow across half a phone photo stops looking
    # like grey ink to the recogniser.
    background = cv2.medianBlur(gray, _background_ksize(gray.shape))
    flat = cv2.divide(gray, background, scale=255)

    flat = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(flat)
    flat = cv2.fastNlMeansDenoising(flat, None, h=7, templateWindowSize=7, searchWindowSize=21)

    # Deliberately NOT binarised, and nobody should "optimise" a threshold back
    # in. Otsu and adaptive thresholding eat thin strokes, and Arabic is thin
    # strokes: the joins between letters, and the dots that are the only
    # difference between several letters, are a pixel or two wide. Grey levels
    # let the recogniser decide; a threshold decides for it, once, wrongly.
    return cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR) if image.ndim == 3 else flat


def preprocess_page(
    image: np.ndarray,
    *,
    crop: bool | None = None,
    straighten: bool | None = None,
    lighting: bool | None = None,
    max_side: int | None = None,
) -> np.ndarray:
    """Run the correction chain, each step defaulting to its settings flag.

    The keywords are named for what they do rather than after the functions
    they call, so that `deskew` the module function stays reachable in here.
    """
    crop = settings.auto_crop if crop is None else crop
    straighten = settings.deskew if straighten is None else straighten
    lighting = settings.fix_lighting if lighting is None else lighting
    max_side = settings.max_image_px if max_side is None else max_side

    result = image
    # Crop first: a skew estimate taken over desk, hand and background measures
    # the background. Downscale only after cropping, so the cap lands on the
    # page rather than on the scene it was sitting in.
    if crop:
        result = crop_to_document(result)
    if straighten:
        result = deskew(result)
    result = _limit_side(result, max_side)
    if lighting:
        result = normalise_lighting(result)
    return result


def encode_jpeg(image: np.ndarray, *, max_side: int = 1800, quality: int = 90) -> bytes:
    """Wire format for the vision engine: smaller than the working page on purpose."""
    return _encode(_limit_side(image, max_side), ".jpg", [cv2.IMWRITE_JPEG_QUALITY, int(quality)])


def encode_png(image: np.ndarray) -> bytes:
    """Lossless, for anything a human or a diff will look at."""
    return _encode(image, ".png", [cv2.IMWRITE_PNG_COMPRESSION, 6])


# --- helpers ---------------------------------------------------------------


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _scaled(image: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _limit_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return image
    scale = max_side / float(longest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _background_ksize(shape: tuple[int, ...]) -> int:
    # Roughly 5% of the short side: wide enough that no glyph survives the
    # blur, which is the point — what survives has to be the paper.
    ksize = max(3, int(min(shape[0], shape[1]) * 0.05)) | 1
    return min(ksize, 101)


def _paper_colour(image: np.ndarray) -> tuple[float, ...]:
    """The colour of blank page, taken as the median of the frame's own edge pixels."""
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    if image.ndim == 3:
        return tuple(float(value) for value in np.median(border, axis=0))
    return (float(np.median(border)),)


def _order_corners(quad: np.ndarray) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    total = quad.sum(axis=1)
    diff = np.diff(quad, axis=1).ravel()
    ordered[0] = quad[np.argmin(total)]
    ordered[2] = quad[np.argmax(total)]
    ordered[1] = quad[np.argmin(diff)]
    ordered[3] = quad[np.argmax(diff)]
    return ordered


def _four_point_transform(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    ordered = _order_corners(quad)
    top_left, top_right, bottom_right, bottom_left = ordered

    # The perspective is unknown, so take the longer of each opposite pair: the
    # near edge of a tilted page is the one that kept its scale.
    width = int(round(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left))))
    height = int(round(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left))))
    if width < 2 or height < 2:
        return image

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_paper_colour(image),
    )


def _ink_angle(image: np.ndarray) -> float | None:
    """Skew in degrees from the minimum-area box around the ink, or None."""
    height, width = image.shape[:2]
    gray = _to_gray(_scaled(image, min(1.0, _WORK_SIDE / float(max(height, width)))))

    # Ink is dark on paper, so the inverse threshold puts the strokes in the
    # foreground; Otsu picks the cut, so a grey scan and a white one both work.
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Close glyphs into words and words into lines: the box should be measuring
    # the text block, not the accidental orientation of one letter.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))

    points = cv2.findNonZero(mask)
    if points is None or len(points) < _MIN_INK_PIXELS:
        logger.debug("deskew: too little ink to measure skew")
        return None
    if len(points) / float(mask.size) > _MAX_INK_COVERAGE:
        logger.debug("deskew: page reads as mostly ink; refusing to guess an angle")
        return None

    angle = float(cv2.minAreaRect(points)[-1])
    # minAreaRect reports (0, 90] on OpenCV >= 4.5 and [-90, 0) before it; both
    # fold to the same (-45, 45] tilt of the same box.
    if angle > 45.0:
        angle -= 90.0
    elif angle < -45.0:
        angle += 90.0
    return angle


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)

    # Grow the canvas to the rotated bounding box: clipping a corner off a
    # receipt to keep the original dimensions would lose the total.
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(round(height * sin + width * cos))
    new_height = int(round(height * cos + width * sin))
    matrix[0, 2] += new_width / 2.0 - width / 2.0
    matrix[1, 2] += new_height / 2.0 - height / 2.0

    # Fill the new corners with paper, not with BORDER_REPLICATE: replicating
    # smears whatever sat on the edge — a page border, the desk — into long
    # diagonal streaks that the next skew estimate reads as ink and the
    # recogniser reads as a rule. Measured on a bordered page, replicate left
    # 5.9 degrees of apparent skew behind where a paper fill left 0.4.
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=_paper_colour(image),
    )


def _encode(image: np.ndarray, suffix: str, params: list[int]) -> bytes:
    ok, buffer = cv2.imencode(suffix, image, params)
    if not ok:
        raise ValueError(f"could not encode a {image.shape} image as {suffix.lstrip('.')}")
    return buffer.tobytes()
