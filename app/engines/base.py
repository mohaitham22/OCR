"""The contract both extraction backends implement.

An engine is a request parameter, not a branch in the code: the traditional OCR
pipeline and the vision LLM are interchangeable because they take the same
arguments, are told the same rules, and hand back the same `ExtractionResult`.
Everything in this module exists to keep that true when only one of the two is
in front of you.

Three pieces do the keeping:

- `SYSTEM_PROMPT` is the whole of the extraction policy -- transcription over
  translation, printed figures over computed ones, null over a guess. Both
  engines send it; neither restates it. A rule added to one engine's prompt and
  not to the other's is how the two start to disagree about the same page.
- `build_prompt` is the only place the schema is described to a model. The
  descriptions on the fields in `app.schemas` are the field-level instructions,
  so the prompt carries the schema rather than paraphrasing it, and the two
  framings -- OCR text as evidence, or page images as the document -- differ
  only in what they say the model is looking at.
- `coerce_fields` is the one reading of a model's answer. Both engines reach a
  `dict` eventually, whether from a vision call or from structuring OCR text,
  and both need the same thing from it: as much of the document as validates,
  and a warning for whatever did not.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ValidationError

from app.schemas import ExtractionResult, Issue, json_schema_for, schema_for

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You read scanned business documents -- receipts, invoices and official paperwork, in \
Arabic, English or both -- and return what is printed on them as JSON.

Transcribe; do not translate and do not transliterate. Text printed in Arabic goes into \
the Arabic field for that value, in Arabic script, exactly as printed. Text printed in \
Latin script goes into the plain field. A document that carries a value in one script only \
fills one of the two fields and leaves the other null. Never write an Arabic name in Latin \
letters, and never write a Latin name in Arabic ones.

Copy amounts as they are printed. Never convert a currency, never rescale a figure, and \
never compute one: if the printed lines do not add up to the printed total, return both as \
printed and let the check downstream find it. An amount is a plain number -- no currency \
symbol, no thousands separator, a dot for the decimal point. The currency itself belongs in \
the currency field, as an ISO 4217 code.

Dates are YYYY-MM-DD. Convert a date written in words or in another calendar; do not \
complete a date that is only partly printed. Read an ambiguous all-numeric date as \
day-month-year: 03/04/2024 is 2024-04-03.

Never infer a value that is not on the page. Do not fill a field from your knowledge of how \
such documents usually look, from another field, or from what the rest of the page implies. \
If a value is absent, cropped, smudged or unreadable, the answer is null. A null is correct \
where a guess is not: a missing field is visible to the reviewer, and a plausible invention \
is not.

Return one JSON object and nothing else -- no prose, no explanation, no code fence -- using \
only the keys the schema gives you."""


def build_prompt(doc_type: str, source_text: str | None = None) -> str:
    """The user-side prompt for one document, schema included.

    `source_text` is what separates the two engines' requests: with it the model
    is reading someone else's transcription and has to be told how far to trust
    it; without it the model is reading the page itself.
    """
    schema = _dumps(compact_schema(doc_type))
    name = _doc_name(doc_type)

    if source_text is None:
        evidence = (
            f"The page images that follow are the {name}. Read all of every page -- header, "
            "footer, margins, stamps, handwriting and any second column -- before you answer."
        )
    else:
        evidence = "\n\n".join(
            (
                f"Below is the text an OCR engine read from the {name}. Treat it as evidence, "
                "not as ground truth. It may hold misread characters, split or merged words, "
                "and numbers that lost a digit or a decimal point. Its reading order may be "
                "wrong: columns can arrive interleaved, a total can be lifted out of its row, "
                "and Arabic runs can appear reversed or broken across lines. Reconstruct the "
                "document from it. You may correct a character OCR clearly misread inside an "
                "otherwise legible word; a value you cannot reconstruct with confidence is "
                "null, not a repair that merely looks plausible.",
                f"<ocr_text>\n{source_text.strip()}\n</ocr_text>",
            )
        )

    return "\n\n".join(
        (
            f"Extract this {name} into a single JSON object.",
            evidence,
            "Fill the schema below. Every field is optional and every one you cannot read is "
            "null. The description on a field is its instruction: where it disagrees with what "
            "you would otherwise do, follow the description.",
            f"<schema>\n{schema}\n</schema>",
            "Return the JSON object only.",
        )
    )


def coerce_fields(doc_type: str, raw: Any) -> tuple[BaseModel | None, list[Issue]]:
    """Validate a model's answer, keeping what parsed and reporting what did not.

    A single bad field -- a quantity that came back as "two", a date written as a
    sentence -- would otherwise throw away an entire correct document. Dropping
    the offending value and warning about it leaves a reviewer with partial data
    and a loud pointer to the hole, which beats a stack trace and nothing.
    """
    model = schema_for(doc_type)

    if not isinstance(raw, dict):
        return None, [
            Issue(
                code="not_an_object",
                message=f"model returned {type(raw).__name__}, not a JSON object",
                severity="error",
                actual=_short(repr(raw)),
            )
        ]

    payload: dict[str, Any] = raw
    issues: list[Issue] = []

    for _ in range(_MAX_COERCE_PASSES):
        try:
            return model.model_validate(payload), issues
        except ValidationError as exc:
            errors = exc.errors()
            issues.extend(_issue_for(error) for error in errors)
            pruned = _prune(payload, [tuple(error.get("loc", ())) for error in errors])
            if pruned is None:
                break
            payload = pruned

    logger.warning("could not coerce a %s: %d unrecoverable error(s)", doc_type, len(issues))
    issues.append(
        Issue(
            code="coercion_failed",
            message=f"nothing in the model's answer validated as a {doc_type}",
            severity="error",
        )
    )
    return None, issues


class Extractor(ABC):
    """One extraction backend.

    Subclasses differ in how they read a page and in nothing else the rest of the
    app can see: same arguments in, same `ExtractionResult` out, same prompt
    policy. `key` is what the registry and the request parameter index on,
    `label` is what the UI shows, and `gives_boxes` says whether `text_blocks`
    will carry coordinates -- the review UI needs to know whether it can
    highlight a field on the page, and that is a capability question, not a
    question about which engine is running.
    """

    key: ClassVar[str] = ""
    label: ClassVar[str] = ""
    gives_boxes: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls.extract, "__isabstractmethod__", False):
            return  # an intermediate base; its concrete subclass answers for it
        if not cls.key or not cls.label:
            raise TypeError(
                f"{cls.__name__} must set both `key` and `label`; the registry indexes on key"
            )

    @abstractmethod
    def extract(
        self,
        pages: Sequence[np.ndarray],
        doc_type: str,
        embedded_text: str | None = None,
    ) -> ExtractionResult:
        """Read `pages` as a `doc_type` and return the result.

        `pages` are preprocessed BGR arrays from `app.preprocess`. `embedded_text`
        is a PDF's own text layer, passed only when triage trusted it; an engine
        may read it instead of the images or ignore it entirely. Failure comes
        back as an `ExtractionResult` with `document=None`, never as an
        exception: one engine falling over must not take a compare-mode run down
        with it.
        """

    @staticmethod
    @contextmanager
    def _timed() -> Iterator[Callable[[], float]]:
        """Milliseconds for `ExtractionResult.duration_ms`, on a monotonic clock.

        Yields the reader rather than a number so the result can still be built
        in one expression at the end of the block:

            with self._timed() as elapsed:
                ...
                return ExtractionResult(..., duration_ms=elapsed())
        """
        start = time.perf_counter()
        yield lambda: (time.perf_counter() - start) * 1000.0


# --- Schema for the prompt -----------------------------------------------
# `app.llm._gemini_schema` rewrites the same schema for a different consumer:
# Gemini's request-time dialect, which is OpenAPI-shaped and rejects a type
# array. This one is read by a model as prose, so it stays valid JSON Schema and
# is pruned only of what carries no instruction. Keeping the two apart is what
# stops the prompt from acquiring one provider's spelling of "nullable".

_PROMPT_NOISE = frozenset({"title", "default", "additionalProperties", "$schema", "$defs", "examples"})
_NESTED_SCHEMA = ("items", "anyOf", "prefixItems", "contains")
# Read order for a human and a model alike: what the field is, then what to put in it.
_KEY_ORDER = ("type", "const", "enum", "format", "description", "required", "properties", "items")


def compact_schema(doc_type: str) -> dict[str, Any]:
    """The document schema stripped to the parts that instruct the model.

    `$ref` is inlined because the line items live behind one, and a model asked
    to fill `items` cannot follow a pointer into a `$defs` it was never shown.
    """
    schema = json_schema_for(doc_type)
    return _shrink(schema, schema.get("$defs", {}), ())


def _shrink(node: Any, defs: dict[str, Any], seen: tuple[str, ...]) -> Any:
    if isinstance(node, list):
        return [_shrink(item, defs, seen) for item in node]
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        return _inline(node, defs, seen)
    if "anyOf" in node:
        node = _flatten_nullable(node)

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _PROMPT_NOISE:
            continue
        if key == "properties":
            # Property *names* are data: filter the schemas under this map, never
            # its keys. Receipt has a field called `items` and GenericDocument one
            # called `title`, and both would vanish if the filter ran a level up.
            out["properties"] = {name: _shrink(sub, defs, seen) for name, sub in value.items()}
        elif key in _NESTED_SCHEMA:
            out[key] = _shrink(value, defs, seen)
        else:
            out[key] = value
    return {key: out[key] for key in sorted(out, key=_key_rank)}


_INLINE_LIST = re.compile(r'\[\s*\n\s*("(?:[^"\\\n]|\\.)*"(?:,\s*\n\s*"(?:[^"\\\n]|\\.)*")*)\s*\n\s*\]')


def _dumps(schema: dict[str, Any]) -> str:
    """Indented JSON, but with the all-string arrays kept on one line.

    Every list in this schema is a `type` or an `enum` of two or three words,
    and at indent=2 each one costs three lines and reads worse than
    `["string", "null"]` does. The pattern cannot match inside a description:
    field descriptions are single-line, so no string literal contains the
    newline the pattern requires.
    """
    dumped = json.dumps(schema, ensure_ascii=False, indent=2)
    return _INLINE_LIST.sub(lambda m: "[" + " ".join(m.group(1).split()) + "]", dumped)


def _key_rank(key: str) -> tuple[int, str]:
    return (_KEY_ORDER.index(key), "") if key in _KEY_ORDER else (len(_KEY_ORDER), key)


def _inline(node: dict[str, Any], defs: dict[str, Any], seen: tuple[str, ...]) -> dict[str, Any]:
    ref = str(node["$ref"])
    name = ref.rsplit("/", 1)[-1]
    if name in seen:
        logger.warning("recursive $ref %r left untyped in the prompt schema", ref)
        return {"type": "object"}
    target = defs.get(name)
    if not isinstance(target, dict):
        raise ValueError(f"unresolvable $ref {ref!r} in schema")
    # Siblings of the $ref win: Pydantic puts the field's description there, and
    # the description is the instruction.
    merged = {**target, **{key: value for key, value in node.items() if key != "$ref"}}
    return _shrink(merged, defs, (*seen, name))


def _flatten_nullable(node: dict[str, Any]) -> dict[str, Any]:
    """`anyOf: [X, null]` is how Pydantic spells `X | None`; say it in one line instead."""
    branches = [branch for branch in node["anyOf"] if isinstance(branch, dict)]
    concrete = [branch for branch in branches if branch.get("type") != "null"]
    if len(concrete) != 1 or len(concrete) == len(branches):
        return node  # a real union, or nothing but null: leave it as written
    merged = {**concrete[0], **{key: value for key, value in node.items() if key != "anyOf"}}
    if isinstance(merged.get("type"), str):
        merged["type"] = [merged["type"], "null"]
    return merged


# --- Salvage -------------------------------------------------------------

_MAX_COERCE_PASSES = 4
_DROP = object()


def _issue_for(error: dict[str, Any]) -> Issue:
    path = ".".join(str(part) for part in error.get("loc", ()))
    return Issue(
        code="field_invalid",
        message=f"{path or 'document'}: {error.get('msg', 'invalid')} -- dropped, needs a human",
        severity="warning",
        field=path or None,
        expected=str(error.get("type", "")) or None,
        actual=_short(repr(error.get("input"))),
    )


def _prune(payload: dict[str, Any], locs: Sequence[tuple[Any, ...]]) -> dict[str, Any] | None:
    """`payload` without the values at `locs`, or None if nothing could be removed.

    Removing is safe to do blind because every extracted field is optional: the
    next pass sees the same document minus the values that did not parse.
    Returning None on a no-op is what stops the caller looping on an error whose
    location it cannot walk.
    """
    pruned = copy.deepcopy(payload)
    removed = sum(_mark(pruned, loc) for loc in locs if loc)
    if not removed:
        # An error we could not locate -- a union tag inserted into `loc`, say.
        # Fall back to dropping the top-level key the path started at.
        blamed = {str(loc[0]) for loc in locs if loc}
        shrunk = {key: value for key, value in pruned.items() if key not in blamed}
        return shrunk if len(shrunk) != len(pruned) else None
    return _sweep(pruned)


def _mark(node: Any, loc: tuple[Any, ...]) -> bool:
    for part in loc[:-1]:
        node = _step(node, part)
        if node is None:
            return False
    last = loc[-1]
    if isinstance(node, dict) and isinstance(last, str) and last in node:
        del node[last]
        return True
    if isinstance(node, list) and isinstance(last, int) and -len(node) <= last < len(node):
        node[last] = _DROP
        return True
    return False


def _step(node: Any, part: Any) -> Any:
    if isinstance(node, dict) and isinstance(part, str):
        return node.get(part)
    if isinstance(node, list) and isinstance(part, int) and -len(node) <= part < len(node):
        return node[part]
    return None


def _sweep(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _sweep(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_sweep(item) for item in node if item is not _DROP]
    return node


def _doc_name(doc_type: str) -> str:
    schema_for(doc_type)  # raises on an unknown type before a prompt is built
    return doc_type.strip().lower()


def _short(text: str, limit: int = 120) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[:limit]}..."
