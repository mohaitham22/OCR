"""The HTTP surface: multipart in, `ProcessResult` out, and nothing decided here.

This module is a translator and a doorman, and deliberately nothing else. It
turns a multipart upload into the arguments `app.pipeline.process` already
takes, turns what comes back into JSON, and turns the handful of things a
caller can get wrong into status codes. Every decision about *how* a document
is read belongs behind `app.pipeline` and `app.engines`; nothing in here may
look at an engine key and act on it, which is why `engine` is passed straight
through as the string it arrived as.

Two properties are worth stating because they are easy to erode.

The pipeline never raises, so a handler has nothing to catch. `process` returns
`ProcessResult(status="failed", error=...)` for a file it cannot read, an
engine that fell over, or a validation rule that broke, and all of those are
answers a caller can render -- so they come back as 200 with `status="failed"`
in the body, not as 500. The codes in here are reserved for requests that were
malformed before any document was read: an empty upload, a file too large, a
doc_type that does not exist, a comparison of fewer than two engines, and a
database-backed endpoint on a deployment that has no database. A 5xx from this
API therefore means the API itself broke, which is a different bug from a
document that could not be read, and the distinction is worth keeping legible
in a log of status codes.

Persistence stays optional, the way it is everywhere else. The three endpoints
that need Postgres answer 503 naming `DATABASE_URL`; extraction and comparison
do not need it and never ask.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app import db, pipeline
from app.config import settings
from app.engines import ENGINE_KEYS, available_engines, get_engine
from app.schemas import DOC_TYPES, ProcessResult

logger = logging.getLogger(__name__)

# The cap this API enforces on one upload. It is a constant here rather than
# `settings.max_upload_mb` because the two currently disagree -- config ships
# 20 -- and the limit a client is refused by has to be the documented one. When
# `app/config.py` is next touched that field should become the source and this
# constant should go; see CLAUDE.md.
MAX_UPLOAD_MB = 25
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Apply the schema if there is a database, and boot either way.

    `init_schema` is create-if-absent and returns False when persistence is
    off, so the no-database deployment passes straight through. A database that
    *is* configured and cannot be reached is logged and not raised: the
    extraction endpoints do not need Postgres, and refusing to start would take
    them down over a dependency they never use. The endpoints that do need it
    fail their own request, loudly, at the moment somebody asks for it.
    """
    logging.basicConfig(level=settings.log_level.upper())
    try:
        if db.init_schema():
            logger.info("database ready")
        else:
            logger.info("no DATABASE_URL; running without persistence")
    except Exception:  # noqa: BLE001 - a broken database must not stop the API booting
        logger.exception("could not apply the schema; storage endpoints will fail until it is fixed")
    yield
    db.close_pool()


app = FastAPI(
    title="OCR DMS",
    version="0.1.0",
    summary="Structured data out of scanned receipts, invoices and documents.",
    lifespan=lifespan,
)


# --- Rejections ----------------------------------------------------------
# One shape for every refusal: a stable `code` a client can branch on, a
# message a human can read, and whatever list makes the message actionable.
# `app.schemas.Issue` spells its findings the same way and for the same reason
# -- the string is for a person, the code is what anything else keys on.


def _reject(status_code: int, code: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message, **extra})


def _read_upload(file: UploadFile) -> bytes:
    """The uploaded bytes, or a 413/400.

    The size is checked twice. `UploadFile.size` is what the multipart parser
    counted, and refusing on it costs nothing; the length of what was actually
    read is checked too, because a client should not get past the limit by
    misdeclaring a part.

    An oversized body still reaches this process either way. A real deployment
    caps it in front -- `client_max_body_size` in nginx, or an ASGI middleware
    counting bytes as they arrive -- and this is the backstop, not the wall.
    """
    declared = file.size
    if declared is not None and declared > MAX_UPLOAD_BYTES:
        raise _reject(
            413,
            "file_too_large",
            f"the upload is {declared / 1024 / 1024:.1f} MB and the limit is {MAX_UPLOAD_MB} MB",
            limit_mb=MAX_UPLOAD_MB,
        )

    data = file.file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise _reject(
            413,
            "file_too_large",
            f"the upload is {len(data) / 1024 / 1024:.1f} MB and the limit is {MAX_UPLOAD_MB} MB",
            limit_mb=MAX_UPLOAD_MB,
        )
    if not data:
        raise _reject(400, "empty_upload", "the uploaded file has no bytes in it")
    return data


def _checked_doc_type(doc_type: str) -> str:
    """`doc_type` normalised the way `app.schemas.schema_for` normalises it, or a 400.

    Checked here rather than left to the pipeline because it is the caller's
    mistake and it has an answer -- the list of what would have worked -- and
    because `process` would otherwise report it as a failed *run*, which is the
    result for a document that could not be read.
    """
    key = doc_type.strip().lower()
    if key not in DOC_TYPES:
        raise _reject(
            400,
            "unknown_doc_type",
            f"unknown doc_type {doc_type!r}",
            valid=sorted(DOC_TYPES),
        )
    return key


def _checked_engine(key: str | None) -> str | None:
    """An engine key the registry recognises, or a 400 carrying the registry's own message.

    `get_engine` is asked rather than `ENGINE_KEYS` consulted, so the family
    aliases (`traditional`, `vlm`) and the empty-means-default rule stay defined
    in one place instead of being restated in an HTTP handler. Constructing an
    engine is cheap -- the recognisers and the SDKs are imported inside the
    methods that use them -- so this costs an object and no I/O.
    """
    wanted = (key or "").strip() or None
    try:
        get_engine(wanted)
    except ValueError as exc:
        raise _reject(400, "unknown_engine", str(exc), valid=list(ENGINE_KEYS)) from exc
    return wanted


def _require_database() -> None:
    if not db.enabled():
        raise _reject(
            503,
            "persistence_disabled",
            "this endpoint needs Postgres and DATABASE_URL is not set; "
            "extraction and comparison work without it",
        )


def _checked_id(document_id: str) -> str:
    """`documents.id` is a uuid column; a string that is not one is a 400, not a driver error."""
    try:
        return str(UUID(document_id))
    except ValueError as exc:
        raise _reject(400, "bad_document_id", f"{document_id!r} is not a uuid") from exc


# --- Health and capabilities ---------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness, and what this deployment is configured to do.

    It does not open a database connection. A health check that touches
    Postgres becomes a load generator the moment a balancer polls it every few
    seconds, and it would report a database that is merely busy as a dead API.
    `persistence` here means configured, not reachable.
    """
    return {
        "status": "ok",
        "version": app.version,
        "persistence": db.enabled(),
        "default_engine": settings.default_engine,
        "engines": list(ENGINE_KEYS),
        "doc_types": sorted(DOC_TYPES),
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.get("/v1/engines")
def engines() -> dict[str, Any]:
    """Every engine a request may name, with the note written for whoever has to pick one.

    `label` and `gives_boxes` are read off the built engine by
    `available_engines`, so this listing and the engine a request then gets
    cannot drift apart.
    """
    return {"engines": [asdict(info) for info in available_engines()]}


# --- Extraction ----------------------------------------------------------


@app.post("/v1/documents/extract")
def extract(
    file: UploadFile = File(..., description="The document: PDF, PNG, JPEG, TIFF, BMP or WebP."),
    doc_type: str = Form("receipt", description="receipt | invoice | document."),
    engine: str | None = Form(None, description="Registry key, family name, or empty for the default."),
    clean_images: bool = Form(True, description="Run the crop/deskew/lighting chain before reading."),
    persist: bool = Form(True, description="Store the result. Ignored when there is no database."),
) -> ProcessResult:
    """Read one document and return everything the pipeline produced.

    The response is a `ProcessResult` verbatim -- the extraction, the issues,
    the review gate's verdict and the database id when a row was written. A run
    that failed comes back 200 with `status="failed"` and the reason in
    `error`, because that is an answer about a document rather than a fault in
    this API.

    Extraction runs inline, on the request thread, and that is the thing in here
    with a known ceiling: a 20-page scan through PaddleOCR holds one worker for
    the better part of a minute, and enough of those at once is an API that
    stops answering `/health`.

    The upgrade path does not reach below this handler. Enqueue
    `(bytes, filename, doc_type, engine, clean_images, persist)` to Redis,
    return 202 with a job id and a `Location` pointing at a status endpoint, and
    have a worker process call `pipeline.process` with exactly those arguments
    and write the result where the status endpoint can read it -- the row in
    `documents` is already the natural place, with the job id beside it.
    Nothing in `app/` changes: `process` is synchronous, takes bytes, never
    raises and returns one value, which is precisely the shape a queue wants.
    The only thing that changes here is that this function stops waiting.
    """
    doc_type = _checked_doc_type(doc_type)
    engine = _checked_engine(engine)
    data = _read_upload(file)

    return pipeline.process(
        data,
        file.filename or "upload",
        doc_type,
        engine,
        clean_images=clean_images,
        persist=persist,
    )


@app.post("/v1/documents/compare")
def compare(
    file: UploadFile = File(..., description="The document to run through several engines."),
    doc_type: str = Form("receipt", description="receipt | invoice | document."),
    engines: str = Form(..., description="Two or more registry keys, comma-separated."),
    clean_images: bool = Form(True, description="Run the correction chain before reading."),
) -> dict[str, Any]:
    """The same pages through several engines at once, plus the field-by-field diff.

    Nothing is stored. A comparison is an experiment -- `pipeline.compare` does
    not take a `persist` argument at all -- and writing N rows for one document
    would put N answers in the table where the archive expects the one that was
    accepted. The engine whose answer is kept goes back through `/extract`.

    A repeated key is honoured rather than collapsed: the vision engines are not
    deterministic, so asking one of them twice over the same pages is a real
    question about how far its answer moves.
    """
    keys = [part.strip() for part in engines.split(",") if part.strip()]
    if len(keys) < 2:
        raise _reject(
            400,
            "too_few_engines",
            f"compare needs at least two engines and got {len(keys)}; "
            "pass them comma-separated, e.g. engines=traditional:paddle,vlm:gemini",
            valid=list(ENGINE_KEYS),
        )
    doc_type = _checked_doc_type(doc_type)
    for key in keys:
        _checked_engine(key)
    data = _read_upload(file)

    results = pipeline.compare(data, file.filename or "upload", doc_type, keys, clean_images=clean_images)
    rows = pipeline.field_diff(results)
    return {
        "engines": keys,
        "results": results,
        # One row per field, one column per engine, under the column names
        # `field_diff` assigned -- which carry a #n suffix when the same engine
        # was asked twice, so two answers from one engine stay distinguishable.
        "diff": [asdict(row) for row in rows],
        "agreement": pipeline.agreement(rows),
    }


# --- Stored documents ----------------------------------------------------


@app.get("/v1/documents", dependencies=[Depends(_require_database)])
def list_documents(
    status: str | None = Query(None, description="pending_review | approved | rejected."),
    doc_type: str | None = Query(None, description="receipt | invoice | document."),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """The review queue and the browse list, newest first.

    `raw_extraction`, `comparison` and `raw_text` are not in these rows: they
    are what makes a stored document worth keeping and also what would not fit
    down the wire a page at a time. `GET /v1/documents/{id}` returns them.
    """
    if doc_type is not None:
        doc_type = _checked_doc_type(doc_type)
    rows = db.list_documents(status=status, doc_type=doc_type, limit=limit, offset=offset)
    return {"documents": rows, "count": len(rows), "limit": limit, "offset": offset}


@app.get("/v1/documents/{document_id}", dependencies=[Depends(_require_database)])
def get_document(document_id: str) -> dict[str, Any]:
    """One document in full, with the corrections a reviewer has already made on it."""
    row = db.get_document(_checked_id(document_id))
    if row is None:
        raise _reject(404, "no_such_document", f"no document {document_id}")
    return row


class Approval(BaseModel):
    """What a reviewer submits: the fields as they should be stored, and who they are."""

    fields: dict[str, Any] | None = Field(
        default=None,
        description=(
            "The corrected document. Omit it to approve the extraction as it stands -- an "
            "omitted body is not an empty document."
        ),
    )
    reviewed_by: str | None = Field(default=None, description="Recorded on the row and on every correction.")


@app.post("/v1/documents/{document_id}/approve", dependencies=[Depends(_require_database)])
def approve(document_id: str, approval: Approval | None = None) -> dict[str, Any]:
    """Record a review: one `corrections` row per changed field, then approve.

    An absent `fields` means "approve what the engine returned", and it is
    resolved here by reading the stored fields and handing them straight back
    rather than by passing None down. `approve_document` flattens whatever it is
    given and diffs it against the stored document, so a None would flatten to
    `{}`, record every field on the page as deleted, and write that into the
    eval set as the answer a human wanted. One extra read is worth not doing
    that.

    It leaves a gap between that read and the locked read inside
    `approve_document`: a correction committed in between would be approved away
    by an empty-bodied request. A reviewer submitting their own `fields` --
    which is what the UI does -- does not go through it.
    """
    document_id = _checked_id(document_id)
    approval = approval or Approval()

    corrected = approval.fields
    if corrected is None:
        stored = db.get_document(document_id)
        if stored is None:
            raise _reject(404, "no_such_document", f"no document {document_id}")
        corrected = stored.get("fields") or {}

    if not db.approve_document(document_id, corrected, reviewed_by=approval.reviewed_by):
        raise _reject(404, "no_such_document", f"no document {document_id}")

    return {"status": "approved", "document": db.get_document(document_id)}
