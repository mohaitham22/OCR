"""Orchestration: bytes in, a `ProcessResult` out, and never an exception.

This is the module that turns the pieces into a system. It loads a file,
decides whether the pages need correcting, hands them to an engine, checks the
arithmetic, runs the review gate and stores the row -- and every one of those
steps is allowed to fail without taking the run down with it.

Two rules shape all of it.

The first is that nothing raises to the caller. `process` is called from a web
handler and from a Streamlit form, and in both places an exception is a stack
trace where an answer should be. So each stage is caught, and a failure comes
back as `ProcessResult(status="failed", error=...)` -- a value the caller can
render, log and store, on the same code path as a success. The stages are not
equal in what they cost, though, and the handling says so: a file that will not
load has no document to return and is fatal to the run, while a save that fails
is logged and swallowed, because returning the extraction beats losing it, and
a page cleanup that fails falls back to the page as loaded, because a
correction is an improvement to a page we can already read.

The second is that the engine is a parameter. Nothing in here asks which engine
ran or branches on the answer -- `get_engine` maps a string to an `Extractor`
and everything after that is the interface. That is what makes `compare` four
lines of thread pool rather than a special mode: running two engines over one
document is running the same code twice.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app import db
from app.engines import get_engine
from app.preprocess import LoadedDocument, encode_png, load_document, preprocess_page
from app.schemas import Issue, ProcessResult
from app.validate import decide_status, validate

logger = logging.getLogger(__name__)

__all__ = [
    "FieldRow",
    "agreement",
    "compare",
    "field_diff",
    "process",
    "render_pages",
]


# --- One document, one engine --------------------------------------------


def process(
    data: bytes,
    filename: str,
    doc_type: str,
    engine_key: str | None = None,
    *,
    clean_images: bool = True,
    persist: bool = True,
) -> ProcessResult:
    """Read one document end to end.

    `engine_key` is a registry key (`traditional:paddle`, `vlm:gemini`), a bare
    family name, or None for `settings.default_engine`. `clean_images` turns off
    the correction chain for a page that does not need it; `persist` turns off
    storage for a run that is not meant to be a record.

    Never raises. A failure comes back as `status="failed"` with the reason in
    `error`.
    """
    started = time.perf_counter()
    try:
        prepared = _prepare(data, filename, clean_images=clean_images)
    except Exception as exc:  # noqa: BLE001 - a file we cannot open is a result, not a crash
        logger.exception("could not load %r", filename)
        return _failed(filename, doc_type, exc, started)

    return _run(prepared, doc_type, engine_key, persist=persist, started=started)


def _run(
    prepared: _Prepared,
    doc_type: str,
    engine_key: str | None,
    *,
    persist: bool,
    started: float,
) -> ProcessResult:
    """Engine, validation, review gate, storage -- the half that has pages already.

    Split out of `process` because `compare` runs it once per engine over one
    shared preparation: loading and cleaning the same file N times would cost N
    times as much and, worse, would not guarantee that the engines were shown
    the same pixels, which is the only thing that makes their disagreement mean
    anything.
    """
    filename = prepared.loaded.filename

    try:
        engine = get_engine(engine_key)
    except Exception as exc:  # noqa: BLE001 - an unknown engine key is the caller's, and reportable
        logger.warning("no such engine %r for %r: %s", engine_key, filename, exc)
        return _failed(filename, doc_type, exc, started)

    try:
        extraction = engine.extract(
            prepared.pages,
            doc_type,
            embedded_text=prepared.loaded.embedded_text or None,
        )
    except Exception as exc:  # noqa: BLE001 - the contract says engines do not raise; trust nothing
        logger.exception("%s failed on %r", engine.key, filename)
        return _failed(filename, doc_type, exc, started)

    result = ProcessResult(source=filename, doc_type=doc_type, extraction=extraction)

    # Validation is where a misread number is caught, so losing it silently
    # would be worse than most failures -- but losing the extraction with it
    # would be worse still. A rule that raises becomes an issue of its own and
    # forces review, which is what an unchecked document deserves anyway.
    notes = list(prepared.notes)
    try:
        issues = validate(doc_type, extraction.document)
    except Exception as exc:  # noqa: BLE001
        logger.exception("validation failed for %r", filename)
        issues = [
            Issue(
                code="validation_failed",
                message=f"the checks could not be run: {exc}",
                severity="error",
            )
        ]

    # `issues` is what the reviewer sees and what `save_document` writes, so the
    # run-level notes go in it too. They reach `decide_status` as `warnings`
    # rather than as issues because that is what they are: statements about this
    # run -- pages dropped, a cleanup declined -- and not about the document
    # disagreeing with itself. Both force review; only one of them is a fault
    # found on the paper.
    result.issues = [*issues, *notes]
    result.needs_review = decide_status(issues, extraction.confidence, warnings=notes)

    # Set before the row is written, not after: `duration_ms` is stored, and a
    # number that stops short of the work it is measuring is worse than none.
    result.duration_ms = (time.perf_counter() - started) * 1000.0

    if persist and db.enabled():
        try:
            result.document_id = db.save_document(result, content_hash=prepared.content_hash)
            result.stored = result.document_id is not None
        except Exception:  # noqa: BLE001 - a lost row is not a lost extraction
            logger.exception("could not store %r; returning the extraction unstored", filename)
            result.stored = False

    logger.info(
        "%s %s: engine=%s issues=%d review=%s stored=%s in %.0f ms",
        filename,
        doc_type,
        extraction.engine,
        len(result.issues),
        result.needs_review,
        result.stored,
        result.duration_ms or 0.0,
    )
    return result


# --- Compare mode --------------------------------------------------------


def compare(
    data: bytes,
    filename: str,
    doc_type: str,
    engine_keys: Sequence[str],
    *,
    clean_images: bool = True,
    max_workers: int | None = None,
) -> list[ProcessResult]:
    """The same document through several engines at once, stored by none of them.

    Results come back in the order the keys were given, one per key, so a caller
    can zip them together with what it asked for -- including when the load
    failed and every engine gets the same failure.

    `persist` is not a parameter. A comparison is an experiment: it exists to be
    read next to `field_diff` and thrown away, and writing N rows for one
    document would put N answers in the table where the archive expects the one
    that was accepted. The engine whose answer is kept goes back through
    `process`.

    A repeated key is honoured rather than collapsed. The vision engines are not
    deterministic, so asking one of them twice over the same pages is a real
    question about how much its answer moves.
    """
    keys = list(engine_keys)
    if not keys:
        return []

    started = time.perf_counter()
    try:
        prepared = _prepare(data, filename, clean_images=clean_images)
    except Exception as exc:  # noqa: BLE001
        logger.exception("could not load %r", filename)
        return [_failed(filename, doc_type, exc, started) for _ in keys]

    def run(key: str) -> ProcessResult:
        # Each engine gets its own clock, started here: the shared load and
        # cleanup happened once and belong to none of them, and charging all N
        # for it would make every engine look equally slow.
        return _run(prepared, doc_type, key, persist=False, started=time.perf_counter())

    # Threads, because every engine spends its time waiting on a network call or
    # inside a recogniser that releases the GIL. The pages are shared across
    # them and never written to -- engines only read and encode -- so there is
    # nothing to copy and nothing to lock.
    with ThreadPoolExecutor(max_workers=max_workers or len(keys)) as pool:
        results = list(pool.map(run, keys))

    logger.info(
        "compared %r across %s",
        filename,
        ", ".join(f"{key}={'ok' if r.status == 'ok' else 'failed'}" for key, r in zip(keys, results)),
    )
    return results


@dataclass(frozen=True, slots=True)
class FieldRow:
    """One field across every engine that was asked: a row of the comparison table."""

    path: str
    values: dict[str, Any]
    agree: bool


def field_diff(results: Sequence[ProcessResult]) -> list[FieldRow]:
    """One row per field, one column per engine, and whether they agree.

    Paths are `app.db.flatten_fields` paths -- `total`, `items.2.unit_price` --
    which are also how `app.validate` spells `Issue.field` and how a correction
    is recorded, so a disagreement, the rule that flagged it and the row a
    reviewer eventually writes all name the same cell.

    Equality is `app.db.diff_fields`'s, for the same reason it is careful there:
    a null and a missing key are one statement, 12 and 12.0 are one number, and
    two engines that both declined to answer have not disagreed.

    An engine that returned no document still gets a column -- an empty one --
    and it disagrees on every row. That is deliberately not the same rule as
    "missing means null": a null is an answer, and an engine that returned
    nothing did not give one. Reading its absence as a page full of nulls would
    make agreement rise on exactly the runs where one engine fell over, since
    most fields on most documents are null.
    """
    # `db._same` and `db._path_key` are reached into rather than reimplemented.
    # They encode decisions this module must not answer differently -- what
    # counts as the same value, and that `items.2` comes before `items.10` --
    # and a second copy here would drift from the one that writes the
    # corrections rows. They should be public on `app.db`; see CLAUDE.md.
    columns = [_column_name(index, result) for index, result in enumerate(results)]
    # None, not {}, for a result with no document: the difference between "this
    # engine answered null" and "this engine did not answer" is the whole of the
    # paragraph above, and {} would erase it.
    flattened = [_answers(result) for result in results]

    paths = sorted({path for flat in flattened if flat for path in flat}, key=db._path_key)
    rows: list[FieldRow] = []
    for path in paths:
        values = {
            name: (flat.get(path) if flat is not None else None)
            for name, flat in zip(columns, flattened)
        }
        rows.append(FieldRow(path=path, values=values, agree=_agree(flattened, path)))
    return rows


def _answers(result: ProcessResult) -> dict[str, Any] | None:
    extraction = result.extraction
    document = extraction.document if extraction is not None else None
    return db.flatten_fields(document) if document is not None else None


def _agree(flattened: Sequence[dict[str, Any] | None], path: str) -> bool:
    if any(flat is None for flat in flattened):
        return False
    first = flattened[0].get(path)  # type: ignore[union-attr]
    return all(db._same(first, flat.get(path)) for flat in flattened[1:])  # type: ignore[union-attr]


def agreement(rows: Sequence[FieldRow]) -> float | None:
    """Fraction of fields the engines agreed on, or None when there was nothing to compare.

    This is what `ProcessResult.agreement` holds, computed where the comparison
    is rather than inside it: `compare` returns N results and the pairing is the
    caller's to choose.
    """
    if not rows:
        return None
    return sum(1 for row in rows if row.agree) / len(rows)


def _column_name(index: int, result: ProcessResult) -> str:
    """The engine key, kept unique: the same engine may legitimately appear twice."""
    extraction = result.extraction
    base = (extraction.engine if extraction is not None else None) or f"engine{index + 1}"
    return f"{base}#{index + 1}" if index else base


# --- Pages for the UI ----------------------------------------------------


def render_pages(
    data: bytes,
    filename: str,
    *,
    clean_images: bool = True,
    max_pages: int | None = None,
) -> list[bytes]:
    """PNG bytes for exactly the pages an engine would have been given.

    `clean_images` is the same switch `process` takes and produces the same
    pages, so a reviewer looking at page 2 is looking at what the extraction was
    read from -- including the crop and the lighting correction, which change
    what is legible.

    PNG rather than JPEG: this is the copy a human inspects, and the engine's
    own JPEG is a wire format sized down for a model.

    Returns `[]` on a file that will not load, rather than raising. The caller
    is a UI, and `process` has already reported the same failure through
    `ProcessResult.error`, where a reviewer can read it.
    """
    try:
        prepared = _prepare(data, filename, clean_images=clean_images, max_pages=max_pages)
    except Exception:  # noqa: BLE001
        logger.exception("could not render %r", filename)
        return []

    rendered: list[bytes] = []
    for number, page in enumerate(prepared.pages, start=1):
        try:
            rendered.append(encode_png(page))
        except Exception:  # noqa: BLE001 - one unencodable page must not blank the whole document
            logger.exception("could not encode page %d of %r", number, filename)
    return rendered


# --- Preparation ---------------------------------------------------------


@dataclass(slots=True)
class _Prepared:
    """A file reduced to what an engine takes, plus what the run has to say about it."""

    loaded: LoadedDocument
    pages: list[np.ndarray]
    content_hash: str
    cleaned: bool
    notes: list[Issue] = field(default_factory=list)


def _prepare(
    data: bytes,
    filename: str,
    *,
    clean_images: bool,
    max_pages: int | None = None,
) -> _Prepared:
    """Load, truncate, and clean the pages -- once, whatever runs on them afterwards.

    Truncation is `load_document`'s: it reads `settings.max_pages` and is the
    only place the cap is applied. What happens here is that the cap is said out
    loud, as a note on the result, because an extraction from the first 20 pages
    of a 400-page file is not wrong so much as partial, and a reviewer who is
    not told cannot know that the totals came off a page nobody read.
    """
    loaded = load_document(data, filename, max_pages=max_pages)

    notes: list[Issue] = []
    if loaded.truncated:
        notes.append(
            Issue(
                code="pages_truncated",
                message=(
                    f"read the first {loaded.page_count} of {loaded.source_page_count} pages; "
                    "anything on the rest was not extracted"
                ),
                severity="warning",
            )
        )

    # The correction chain is for pages that were photographed or scanned. A PDF
    # whose text layer we trust was rendered from vectors: there is no skew to
    # remove, no page edge to find in a scene, and no lighting gradient -- and
    # `crop_to_document` mistaking a printed border for the edge of the paper is
    # a failure this repo has already had once. So cleaning is skipped there,
    # which is also why the switch is `clean_images AND no text layer` rather
    # than either one alone.
    cleaned = clean_images and not loaded.has_text_layer
    pages = _clean(loaded.pages, notes) if cleaned else list(loaded.pages)

    return _Prepared(
        loaded=loaded,
        pages=pages,
        # The bytes are here and nowhere else afterwards, so this is where the
        # hash is taken. It is what `documents.content_hash` is indexed on, and
        # duplicate filings are found on it later.
        content_hash=hashlib.sha256(data).hexdigest(),
        cleaned=cleaned,
        notes=notes,
    )


def _clean(pages: Sequence[np.ndarray], notes: list[Issue]) -> list[np.ndarray]:
    """`preprocess_page` per page, falling back to the page as loaded.

    A cleanup that raises is not a document we cannot read; it is an improvement
    we did not get. Failing the run over it would throw away a page the engine
    would have managed, so the original goes through instead and the reviewer is
    told which page it was -- the raw page is likelier to be misread, and that
    is worth a look.
    """
    cleaned: list[np.ndarray] = []
    for number, page in enumerate(pages, start=1):
        try:
            cleaned.append(preprocess_page(page))
        except Exception:  # noqa: BLE001
            logger.exception("could not clean page %d; using it as loaded", number)
            cleaned.append(page)
            notes.append(
                Issue(
                    code="page_not_cleaned",
                    message=f"page {number} could not be corrected and was read as it arrived",
                    severity="warning",
                    field=None,
                )
            )
    return cleaned


def _failed(filename: str, doc_type: str | None, exc: Exception, started: float) -> ProcessResult:
    """A run that produced nothing, as a value.

    `needs_review` is True and stays true: a document that failed to extract is
    exactly a document a human still has to deal with, and a caller listing the
    queue should find it there rather than have to know that failures are kept
    somewhere else.
    """
    return ProcessResult(
        status="failed",
        error=f"{type(exc).__name__}: {exc}",
        source=filename,
        doc_type=doc_type,
        needs_review=True,
        duration_ms=(time.perf_counter() - started) * 1000.0,
    )
