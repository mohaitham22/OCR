"""The engine registry: a key in, an `Extractor` out.

The engine is a request parameter, so something has to turn the string a
request carries into an object, and this is it. Two rules keep it honest:

- Building is lazy. A key maps to a function that imports its module and
  constructs the engine, so importing `app.engines` costs nothing and a
  deployment with no PaddleOCR installed can still list, choose and run the
  three engines that do not need it. `get_engine` is the only place an engine
  module is imported.
- Nothing here reads a page or knows how one is read. This module maps names to
  constructors and describes the choice to a human; the reading lives behind
  `Extractor`, and callers outside `app/engines/` see only that interface.

`available_engines` exists for the UI, which has to offer this choice to
somebody who did not write any of it. Its `note` is written for that person: it
says what the engine is good and bad at, not what its class is called. All four
make at least one model call and so all four need an API key -- the traditional
pair to turn OCR text into fields, the vision pair to read the pages.
"""

from __future__ import annotations

import logging
import textwrap
from collections.abc import Callable
from dataclasses import asdict, dataclass

from app.config import settings
from app.engines.base import Extractor

logger = logging.getLogger(__name__)

__all__ = ["EngineInfo", "ENGINE_KEYS", "Extractor", "available_engines", "get_engine"]


def _traditional(backend: str | None) -> Extractor:
    from app.engines.traditional import TraditionalExtractor

    return TraditionalExtractor(backend)


def _vlm(provider: str | None) -> Extractor:
    from app.engines.vlm import VLMExtractor

    return VLMExtractor(provider)


# The four engines, in the order the UI should offer them. `note` lives here
# because it lives nowhere else; `label` and `gives_boxes` are read off the
# built engine instead of being copied, so the list the UI shows and the engine
# it then runs cannot drift apart.
_BUILDERS: dict[str, Callable[[], Extractor]] = {
    "traditional:paddle": lambda: _traditional("paddle"),
    "traditional:tesseract": lambda: _traditional("tesseract"),
    "vlm:gemini": lambda: _vlm("gemini"),
    "vlm:openai": lambda: _vlm("openai"),
}

_NOTES: dict[str, str] = {
    "traditional:paddle": (
        "Reads the page with PaddleOCR, then asks a text model to fill in the fields. The "
        "stronger of the two readers on Arabic and on mixed Arabic/English pages, and it "
        "records where on the paper every line was found, so a reviewer can see which part "
        "of the page a value was taken from. It wants a reasonably clean printed page; the "
        "OCR models are a large one-off download and the first document after startup is slow."
    ),
    "traditional:tesseract": (
        "Reads the page with Tesseract, then asks a text model to fill in the fields. Quick "
        "to install and quick to start, and it also records where each line sits on the page, "
        "but it is weaker than PaddleOCR on Arabic script and falls further behind on "
        "photographs and faint print. A fair choice for clean English scans, and the second "
        "opinion when PaddleOCR looks wrong."
    ),
    "vlm:gemini": (
        "Sends the page images to Google Gemini, which reads the paper and fills in the "
        "fields in one step. The best choice for a phone photograph, a creased or badly lit "
        "page, handwriting, or a layout that scrambles the OCR readers. In exchange it cannot "
        "say where on the page a value came from, reports no per-line confidence, and may "
        "answer slightly differently on the same page twice, so its figures lean harder on "
        "the arithmetic checks."
    ),
    "vlm:openai": (
        "The same one-step reading as the Gemini engine, through OpenAI's model instead: the "
        "same strength on messy photographs and the same loss of on-page positions and "
        "confidences. Worth running when Gemini's answer looks wrong, and the natural second "
        "opinion in a side-by-side comparison."
    ),
}

ENGINE_KEYS: tuple[str, ...] = tuple(_BUILDERS)

# A bare family name is not a fifth engine; it is "whichever member of this
# family this deployment is configured for". `settings.default_engine` ships as
# `traditional`, so without these a stock configuration would name nothing. The
# choice itself stays in the engine -- `TraditionalExtractor` already reads
# `settings.ocr_backend` and `VLMExtractor` the configured provider -- rather
# than being read a second time, and differently, here.
_FAMILY_DEFAULTS: dict[str, Callable[[], Extractor]] = {
    "traditional": lambda: _traditional(None),
    "vlm": lambda: _vlm(None),
}


@dataclass(frozen=True, slots=True)
class EngineInfo:
    """One engine as the review UI needs to offer it."""

    key: str
    label: str
    family: str
    gives_boxes: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def __str__(self) -> str:
        boxes = "shows where on the page each value was read" if self.gives_boxes else "no on-page positions"
        return f"{self.key}  --  {self.label}  [{boxes}]\n" + textwrap.indent(
            textwrap.fill(self.note, width=76), "    "
        )


def get_engine(key: str | None = None) -> Extractor:
    """The engine named by `key`, or the configured default when `key` is empty.

    Raises `ValueError` rather than falling back to a default, because a request
    that named an engine and silently got a different one produces a result
    nobody can account for afterwards.
    """
    wanted = (key or "").strip().lower() or str(settings.default_engine or "").strip().lower()
    builder = _BUILDERS.get(wanted) or _FAMILY_DEFAULTS.get(wanted)
    if builder is None:
        raise ValueError(
            f"unknown engine {key!r}; expected one of {', '.join(ENGINE_KEYS)} "
            f"(or {', '.join(sorted(_FAMILY_DEFAULTS))} for this deployment's default)"
        )
    # Built fresh each time: an engine holds no state worth reusing, and the one
    # expensive thing behind them -- the PaddleOCR readers -- is cached in
    # `app.engines.traditional`, where the cost actually is.
    return builder()


def available_engines() -> list[EngineInfo]:
    """Every engine a request may name, described for whoever has to pick one."""
    listed: list[EngineInfo] = []
    for key, builder in _BUILDERS.items():
        engine = builder()
        listed.append(
            EngineInfo(
                key=engine.key,
                label=engine.label,
                family=key.split(":", 1)[0],
                gives_boxes=engine.gives_boxes,
                note=_NOTES[key],
            )
        )
    return listed
