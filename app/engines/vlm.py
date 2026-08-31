"""The vision backend: one model call reads the page and fills the schema.

Where the traditional engine has two stages with a seam between them -- a
recogniser that turns pixels into boxes, then a text model that turns boxes into
a document -- this one has a single call that does both at once. That is the
whole difference, and it is a trade, not an upgrade:

- No boxes. `gives_boxes` is False and `text_blocks` comes back empty. The
  review UI cannot highlight the region a value was read from, because nothing
  in the answer says where on the page it was.
- No per-line confidence. `confidence` is None. The model reports nothing about
  how sure it was of a character, and a number invented here would be a claim
  about the model's mood rather than about the page.
- Not deterministic. Temperature is 0, but the same page can still come back
  with a different answer than it did last time, and two providers will
  certainly differ.

`app.validate` is what compensates: on this path the arithmetic checks and the
format rules are the only evidence that the fields match the paper. That is also
why `_transcript` exists -- the model's own plain-text reading of the pages,
asked for as an extra key and kept as `raw_text`, so a reviewer has something on
screen to check the extracted fields against.

What the trade buys is the page an OCR reader gives up on: a creased, skewed,
badly-lit photograph, handwriting, a stamp across a total, a layout no line
grouping recovers. `app.pipeline`'s compare mode exists so the two engines can
be run over the same document and disagree in public.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from app import llm
from app.config import settings
from app.engines.base import SYSTEM_PROMPT, Extractor, build_prompt, coerce_fields
from app.preprocess import encode_jpeg
from app.schemas import ExtractionResult, json_schema_for, schema_for

logger = logging.getLogger(__name__)

# The vision-capable subset of `app.llm.PROVIDERS`, with the names a person
# recognises. DeepSeek is absent because it serves no vision model, and
# `llm.structured_vision` refuses it by name rather than letting the SDK answer
# with a confusing 400 -- so a provider that slipped past here would still fail
# with a sentence someone can act on.
PROVIDERS: dict[str, str] = {"gemini": "Google Gemini", "openai": "OpenAI GPT"}
DEFAULT_PROVIDER = "gemini"

TRANSCRIPT_KEY = "_transcript"
"""Extra top-level key asked of the model, popped before the document is validated."""


# --- Prompt --------------------------------------------------------------
# The extraction policy itself stays in `app.engines.base`: both engines send
# `SYSTEM_PROMPT` and neither restates it, because a rule that lands in one
# engine's prompt and not in the other's is how two interchangeable backends
# start disagreeing about the same page. What is added here is not policy but
# condition -- what the model is looking at. Only this engine's model sees
# pixels; the traditional engine's model reads a transcription and is warned
# about *that* by `build_prompt`'s OCR framing. Neither warning helps the other.

_VISION_NOTES = """\
You are looking at the pages themselves -- photographs and scans, not clean digital \
copies. Expect skew and rotation, glare and shadow, creases and folds, a curled page whose \
lines bend, an edge torn or cropped out of frame, a hand or a desk around the paper, and \
print that is faint, doubled or smeared. Read what is on the paper in the state it is in.

A character you cannot read makes the whole field null. Do not rebuild the field around the \
gap, do not recover a digit from the column it sits in or from the figures near it, and do \
not finish a word from its first half. A total under glare, a date creased along a fold, a \
number whose last digit is cut off by the edge of the frame: all unreadable, and unreadable \
is null. Half a figure recorded as though it were whole is worse than no figure at all -- a \
reviewer can see an empty field and cannot see a confident wrong one.

Look at every page you are given before you answer. A document runs across its pages, line \
items continue onto the next one, and the totals are usually on the last.

Alongside the schema's fields, fill the _transcript key with your own plain-text reading of \
the pages: every line you can read, in reading order, in the script it is printed in, each \
page introduced by a line reading --- page 1 ---. Transcribe it; do not summarise, translate \
or tidy it. It is what the reviewer reads the extracted fields against, and it is the one \
place in your answer where a partly legible line still belongs: write there as much of such \
a line as you can read, and still leave the field it would have filled null."""

VISION_SYSTEM_PROMPT = f"{SYSTEM_PROMPT}\n\n{_VISION_NOTES}"


# --- The engine ----------------------------------------------------------


class VLMExtractor(Extractor):
    """A vision model, once, over all the pages.

    `key` and `label` are set per instance for the same reason the traditional
    engine does it: `vlm:gemini` and `vlm:openai` are two registry entries,
    because asking both and comparing the answers is the point of having two.
    The class-level values exist so the base's `__init_subclass__` check passes
    at import and so a bare `vlm` still names something.
    """

    key: ClassVar[str] = "vlm"
    label: ClassVar[str] = "Vision model"
    gives_boxes: ClassVar[bool] = False

    def __init__(self, provider: str | None = None) -> None:
        chosen = str(provider or _configured_provider()).strip().lower()
        if chosen not in PROVIDERS:
            raise ValueError(
                f"unknown vision provider {provider!r}; expected one of {sorted(PROVIDERS)}"
            )
        self.provider = chosen
        self.key = f"vlm:{chosen}"
        self.label = f"Vision model ({PROVIDERS[chosen]})"

    def extract(
        self,
        pages: Sequence[np.ndarray],
        doc_type: str,
        embedded_text: str | None = None,
    ) -> ExtractionResult:
        schema_for(doc_type)  # a bad doc_type is a caller bug; fail before the clock starts

        # `embedded_text` is ignored on purpose. Reading a PDF's own text layer
        # instead of its pixels would make this a text engine wearing a vision
        # engine's key, and compare mode would then be running two text
        # extractions against each other while calling one of them the vision
        # result. Where the text layer is the better source, preferring it is
        # already the traditional engine's answer.
        if embedded_text and embedded_text.strip():
            logger.debug(
                "%s ignoring a %d-char text layer: this path reads the pages",
                self.key,
                len(embedded_text),
            )

        with self._timed() as elapsed:
            if not pages:
                logger.warning("%s was given no pages", self.key)
                return self._empty(doc_type, pages, elapsed())

            try:
                images = [encode_jpeg(page) for page in pages]
            except Exception:
                logger.exception("%s could not encode %d page(s)", self.key, len(pages))
                return self._empty(doc_type, pages, elapsed())

            logger.info(
                "%s reading %s: %d page(s), %d KB of JPEG",
                self.key,
                doc_type,
                len(images),
                sum(len(image) for image in images) // 1024,
            )

            try:
                raw = llm.structured_vision(
                    build_prompt(doc_type),
                    images,
                    _schema_with_transcript(doc_type),
                    system=VISION_SYSTEM_PROMPT,
                    provider=self.provider,
                )
            except Exception:
                # Per the `Extractor.extract` contract a failed engine returns
                # an empty result rather than raising: in compare mode the other
                # engine still has an answer to give.
                logger.exception("%s could not extract %s", self.key, doc_type)
                return self._empty(doc_type, pages, elapsed())

            transcript = _pop_transcript(raw)
            document, issues = coerce_fields(doc_type, raw)
            for issue in issues:
                # `ExtractionResult` has nowhere to carry these yet, so logging
                # is all an engine can do with them. See CLAUDE.md.
                logger.warning("%s: %s", self.key, issue.message)

            return ExtractionResult(
                doc_type=doc_type,
                document=document,
                engine=self.key,
                # No recogniser ran, so there are no boxes and no per-character
                # score -- and neither is invented. An empty list is a fact; a
                # fabricated 0.95 is a claim the review gate would act on.
                # `app.validate` is what stands in for both on this path: its
                # arithmetic and format rules are the only check that these
                # fields came off the paper.
                text_blocks=[],
                raw_text=transcript,
                confidence=None,
                page_count=len(pages),
                duration_ms=elapsed(),
            )

    def _empty(
        self,
        doc_type: str,
        pages: Sequence[np.ndarray],
        duration_ms: float,
        *,
        raw_text: str | None = None,
    ) -> ExtractionResult:
        """A failure, in the shape every caller already knows how to read."""
        return ExtractionResult(
            doc_type=doc_type,
            document=None,
            engine=self.key,
            text_blocks=[],
            raw_text=raw_text,
            confidence=None,
            page_count=len(pages),
            duration_ms=duration_ms,
        )


# --- The transcript ------------------------------------------------------


_TRANSCRIPT_DESCRIPTION = (
    "Your own plain-text reading of every page, in reading order, in the script it is "
    "printed in, each page introduced by a line reading '--- page 1 ---'. Transcribe it; "
    "do not summarise, translate or reorder it. This is not one of the document's fields: "
    "it is what a human reviewer reads the extracted fields against."
)


def _schema_with_transcript(doc_type: str) -> dict[str, Any]:
    """The document schema plus `_transcript`, so the provider does not strip it.

    Asking for the key in the prompt alone is not enough. Gemini is handed this
    schema as `response_schema` and answers with the keys it names and no
    others, so a `_transcript` mentioned only in prose would be dropped before
    it ever reached us. Declaring it here is also what keeps `build_prompt`'s
    "use only the keys the schema gives you" honest.
    """
    schema = json_schema_for(doc_type)
    properties = dict(schema.get("properties", {}))
    properties[TRANSCRIPT_KEY] = {"type": "string", "description": _TRANSCRIPT_DESCRIPTION}
    return {**schema, "properties": properties}


def _pop_transcript(raw: Any) -> str | None:
    """Take `_transcript` out of the answer before the rest is validated.

    The document models are `extra="ignore"`, so leaving it in would be harmless
    -- but it is not a field of the document, and removing it here is what makes
    that explicit rather than incidental. A model asked for a string
    occasionally answers with one entry per page; joining beats discarding.
    """
    if not isinstance(raw, dict):
        return None
    value = raw.pop(TRANSCRIPT_KEY, None)
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (list, tuple)):
        pages = [str(page).strip() for page in value if str(page).strip()]
        return "\n\n".join(pages) or None
    if value is not None:
        logger.warning("%s came back as %s, not text", TRANSCRIPT_KEY, type(value).__name__)
    return None


def _configured_provider() -> str:
    """Which provider a bare `VLMExtractor()` uses.

    Read through getattr for the same reason `app.llm` reads its own provider
    that way: config has no `llm_provider` field yet, and this is the fallback
    that disappears when it gains one. A configured provider that serves no
    vision model is a misconfiguration worth a log line, not a crash on a
    request that named no provider.
    """
    configured = str(getattr(settings, "llm_provider", "") or "").strip().lower()
    if configured in PROVIDERS:
        return configured
    if configured:
        logger.warning(
            "configured provider %r serves no vision model; the vision engine is using %s",
            configured,
            DEFAULT_PROVIDER,
        )
    return DEFAULT_PROVIDER
