"""Postgres persistence, and the only place SQL is written.

Persistence is optional and this module is where that is enforced. With
`settings.database_url` empty every function here returns its empty answer --
None, [], False -- and returns it immediately, without importing a driver,
opening a socket or raising. The pipeline calls these the same way whether a
database is configured or not, and the only difference downstream is that
`ProcessResult.stored` stays False. Nothing above this module may ask whether
Postgres is there.

An empty URL is the only thing that gets that treatment. A URL that *is* set
and does not work -- no driver installed, host unreachable, credentials wrong
-- raises, because a database somebody configured and expects to be written to
is a different situation from one that was never configured, and answering both
with a quiet None makes the first indistinguishable from the second until
somebody goes looking for a month of documents that were never stored.

psycopg is imported inside the functions that need it, for the same reason
`paddleocr` is: the app has to boot on a machine that does not have it.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, NamedTuple
from uuid import UUID

from pydantic import BaseModel

from app.config import settings
from app.schemas import ProcessResult

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"

# Everything a listing needs and nothing that is measured in megabytes.
# `raw_extraction`, `comparison` and `raw_text` are excluded on purpose: they
# are the reason a row is worth keeping and the reason a page of them would not
# fit down the wire. `fields` stays, because a list with no values in it is a
# list of nothing.
_LIST_COLUMNS = """
    id, source, content_hash, doc_type, engine, model, status, fields, issues,
    confidence, agreement, page_count, duration_ms,
    created_at, updated_at, reviewed_at, reviewed_by
"""

_pool: Any = None
_pool_lock = threading.Lock()


class FieldChange(NamedTuple):
    """One leaf a reviewer changed. `path` is dotted, as `Issue.field` is."""

    path: str
    old: Any
    new: Any


def enabled() -> bool:
    """Whether anything in this module will actually touch a database."""
    return settings.persistence_enabled


# --- Connection ----------------------------------------------------------


def get_pool() -> Any | None:
    """The process-wide connection pool, or None when persistence is off.

    Cached behind a lock rather than built per call: two web threads arriving
    together must not each open `db_pool_min` connections and throw one set
    away.
    """
    if not enabled():
        return None

    global _pool
    if _pool is not None:
        return _pool

    with _pool_lock:
        if _pool is not None:
            return _pool
        try:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as exc:  # a configured database with no driver is a deployment fault
            raise RuntimeError(
                "DATABASE_URL is set but psycopg is not installed; "
                "run: pip install 'psycopg[binary,pool]'"
            ) from exc

        pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=settings.db_pool_min,
            max_size=settings.db_pool_max,
            # Rows come back as dicts because everything above this module is
            # JSON-shaped -- the API returns them and the UI renders them.
            kwargs={"row_factory": dict_row},
            open=False,
        )
        pool.open()
        _pool = pool
        logger.info("postgres pool open (min=%s max=%s)", settings.db_pool_min, settings.db_pool_max)
    return _pool


def close_pool() -> None:
    """Release the pool. For tests and for a clean shutdown; safe to call twice."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


def init_schema() -> bool:
    """Apply `sql/schema.sql`. False when persistence is off.

    The file is idempotent, so this runs on every boot rather than being
    guarded by a migration table nobody has written yet.
    """
    if not enabled():
        logger.debug("no DATABASE_URL; schema not applied")
        return False
    pool = get_pool()
    assert pool is not None  # enabled() already said so
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with pool.connection() as conn:
        conn.execute(sql)
        _warn_if_ctype_cannot_see_arabic(conn)
    logger.info("schema applied from %s", SCHEMA_PATH)
    return True


def _warn_if_ctype_cannot_see_arabic(conn: Any) -> None:
    """The trigram index is a no-op for Arabic on a C-locale database.

    pg_trgm asks the database's LC_CTYPE what counts as a letter, and under C
    nothing outside ASCII does, so every Arabic character is dropped before the
    trigrams are cut: `show_trgm` on an Arabic word returns an empty array and
    the index over `raw_text` matches nothing Arabic, for the life of the
    cluster and without one error along the way. It cannot be fixed here --
    LC_CTYPE is fixed at CREATE DATABASE -- so the only thing this can do is
    say so once, loudly, at the moment somebody is looking at the logs.
    """
    row = conn.execute(
        "SELECT datctype FROM pg_database WHERE datname = current_database()"
    ).fetchone()
    ctype = (row["datctype"] if isinstance(row, dict) else row[0]) or ""
    if ctype.strip().upper() in ("C", "POSIX"):
        logger.warning(
            "database LC_CTYPE is %r: pg_trgm will index no Arabic at all, so "
            "trigram search over raw_text silently matches Latin only. Recreate "
            "the database with a UTF-8-aware LC_CTYPE.",
            ctype,
        )


# --- Writing -------------------------------------------------------------


def save_document(result: ProcessResult, *, content_hash: str | None = None) -> str | None:
    """Store one processed document. Returns its id, or None when persistence is off.

    `fields` gets the document as extracted and `raw_extraction` the whole
    `ExtractionResult` behind it. The split is the point: `fields` is replaced
    by `approve_document` and `raw_extraction` never is, so the engine's
    original answer stays recoverable for the life of the row. See the comment
    on the columns in `sql/schema.sql`.
    """
    if not enabled():
        return None
    pool = get_pool()
    assert pool is not None

    from psycopg.types.json import Jsonb

    extraction = result.extraction
    document = extraction.document if extraction is not None else None
    doc_type = result.doc_type or (extraction.doc_type if extraction is not None else None)
    if not doc_type:
        raise ValueError("cannot store a document with no doc_type")

    row = {
        "source": result.source,
        "content_hash": content_hash,
        "doc_type": doc_type,
        "engine": extraction.engine if extraction is not None else None,
        "model": extraction.model if extraction is not None else None,
        # decide_status has already run; this is only its spelling in SQL.
        "status": "pending_review" if result.needs_review else "approved",
        "fields": Jsonb(document.model_dump() if document is not None else {}),
        "raw_extraction": Jsonb(extraction.model_dump()) if extraction is not None else None,
        "raw_text": extraction.raw_text if extraction is not None else None,
        "issues": Jsonb([issue.model_dump() for issue in result.issues]),
        "confidence": extraction.confidence if extraction is not None else None,
        "comparison": Jsonb(result.comparison.model_dump()) if result.comparison is not None else None,
        "agreement": result.agreement,
        "page_count": extraction.page_count if extraction is not None else None,
        "duration_ms": result.duration_ms,
    }

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (
                source, content_hash, doc_type, engine, model, status,
                fields, raw_extraction, raw_text, issues,
                confidence, comparison, agreement, page_count, duration_ms
            ) VALUES (
                %(source)s, %(content_hash)s, %(doc_type)s, %(engine)s, %(model)s, %(status)s,
                %(fields)s, %(raw_extraction)s, %(raw_text)s, %(issues)s,
                %(confidence)s, %(comparison)s, %(agreement)s, %(page_count)s, %(duration_ms)s
            )
            RETURNING id
            """,
            row,
        )
        stored = cur.fetchone()

    document_id = str(stored["id"])  # type: ignore[index]
    logger.info("stored document %s (%s, %s)", document_id, doc_type, row["status"])
    return document_id


def approve_document(
    document_id: str | UUID,
    corrected_fields: BaseModel | dict[str, Any] | None = None,
    *,
    reviewed_by: str | None = None,
) -> bool:
    """Record a review: one `corrections` row per changed field, then approve.

    Both halves happen in one transaction, and the order matters. The UPDATE
    overwrites `documents.fields`, and after it there is nothing left that says
    a human looked at `items.2.unit_price` and disagreed -- `raw_extraction`
    still holds the old answer, but it holds the old answer for every field,
    including the ones the reviewer read and accepted. The `corrections` rows
    are what separate the two, they are the eval set, and they only exist if
    they are written at the instant the change is made. A commit that landed the
    UPDATE without them would silently cost a training pair, so neither lands
    without the other.

    False means nothing was written: persistence is off, or no document has
    that id. A reviewer who changed nothing still approves -- that is a
    judgement about the document -- and simply produces no `corrections` rows.
    """
    if not enabled():
        return False
    pool = get_pool()
    assert pool is not None

    from psycopg.types.json import Jsonb

    corrected = _as_mapping(corrected_fields)

    with pool.connection() as conn, conn.cursor() as cur:
        # FOR UPDATE: two reviewers on the same document would otherwise both
        # diff against the pre-review values and write the same corrections
        # twice, with the second UPDATE discarding the first one's edits.
        cur.execute(
            "SELECT doc_type, engine, model, fields FROM documents WHERE id = %s::uuid FOR UPDATE",
            (str(document_id),),
        )
        stored = cur.fetchone()
        if stored is None:
            logger.warning("approve_document: no document %s", document_id)
            return False

        changes = diff_fields(stored["fields"], corrected)  # type: ignore[index]
        if changes:
            cur.executemany(
                """
                INSERT INTO corrections (
                    document_id, field_path, old_value, new_value,
                    doc_type, engine, model, corrected_by
                ) VALUES (%s::uuid, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        str(document_id),
                        change.path,
                        Jsonb(change.old) if change.old is not None else None,
                        Jsonb(change.new) if change.new is not None else None,
                        stored["doc_type"],  # type: ignore[index]
                        stored["engine"],  # type: ignore[index]
                        stored["model"],  # type: ignore[index]
                        reviewed_by,
                    )
                    for change in changes
                ],
            )

        cur.execute(
            """
            UPDATE documents
               SET fields = %s,
                   status = 'approved',
                   reviewed_at = now(),
                   reviewed_by = %s,
                   updated_at = now()
             WHERE id = %s::uuid
            """,
            (Jsonb(corrected), reviewed_by, str(document_id)),
        )

    logger.info("approved document %s with %d correction(s)", document_id, len(changes))
    return True


# --- Reading -------------------------------------------------------------


def list_documents(
    *,
    status: str | None = None,
    doc_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """The review queue and the browse list. [] when persistence is off.

    Ordered newest first, and filtered on `status` and `doc_type`, because
    those are the two composite indexes: any combination of the two filters is
    an index scan in the order the rows are already wanted in.
    """
    if not enabled():
        return []
    pool = get_pool()
    assert pool is not None

    where: list[str] = []
    params: list[Any] = []
    if status:
        where.append("status = %s")
        params.append(status)
    if doc_type:
        where.append("doc_type = %s")
        params.append(doc_type)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    params.extend([limit, offset])

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {_LIST_COLUMNS} FROM documents {clause} "
            "ORDER BY created_at DESC LIMIT %s OFFSET %s",
            params,
        )
        return list(cur.fetchall())


def get_document(document_id: str | UUID) -> dict[str, Any] | None:
    """One document in full, its corrections attached. None when persistence is off.

    Unlike `list_documents` this returns `raw_extraction` and `raw_text`: this
    is the call the review UI makes when somebody opens a document, and the
    transcription is what they check the fields against.
    """
    if not enabled():
        return None
    pool = get_pool()
    assert pool is not None

    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM documents WHERE id = %s::uuid", (str(document_id),))
        row = cur.fetchone()
        if row is None:
            return None
        cur.execute(
            """
            SELECT field_path, old_value, new_value, engine, model, corrected_by, created_at
              FROM corrections
             WHERE document_id = %s::uuid
             ORDER BY created_at, id
            """,
            (str(document_id),),
        )
        row["corrections"] = list(cur.fetchall())  # type: ignore[index]
    return dict(row)


# --- Flatten and diff ----------------------------------------------------
# The two halves of `approve_document` worth testing on their own, and the only
# thing in this module that does not need a database.


def flatten_fields(fields: BaseModel | dict[str, Any] | None) -> dict[str, Any]:
    """A nested document as `{dotted path: leaf value}`.

    The paths are spelled the way `app.validate` spells `Issue.field` --
    `total`, `items.2.unit_price` -- so a correction and the rule that flagged
    it name the same cell, and the review UI can put them side by side without
    translating between two path languages.

    An empty list or dict is a leaf rather than nothing. A reviewer who deletes
    every line item leaves no `items.N.*` paths behind, and without `items: []`
    the diff would record only disappearances, which read the same as a
    document that never had lines.
    """
    data = _as_mapping(fields)
    if not data:
        # The empty-container rule below is about a field *inside* a document.
        # An empty document has no fields, not one field whose name is "".
        return {}
    flat: dict[str, Any] = {}
    _walk(data, "", flat)
    return flat


def _walk(value: Any, path: str, out: dict[str, Any]) -> None:
    if isinstance(value, dict) and value:
        for key, child in value.items():
            _walk(child, f"{path}.{key}" if path else str(key), out)
    elif isinstance(value, list) and value:
        for index, child in enumerate(value):
            _walk(child, f"{path}.{index}" if path else str(index), out)
    else:
        out[path] = value


def diff_fields(
    stored: BaseModel | dict[str, Any] | None,
    corrected: BaseModel | dict[str, Any] | None,
) -> list[FieldChange]:
    """Every leaf whose value a reviewer actually changed, in document order.

    "Actually" is doing work in that sentence, because a correction row that
    records a difference nobody made is worse than no row: it goes into the
    eval set as a page the engine got wrong, and into the fine-tuning set as an
    example of what to answer instead. So two values that mean the same thing
    are not a change. A missing key and a null are both "this field carries no
    value", which is what every optional field in `app.schemas` returns when
    the document does not show it. An empty string is the same statement typed
    into a form, and is recorded as the null it means rather than as itself. 12
    and 12.0 are one number that survived a round trip through JSON.
    """
    before = flatten_fields(stored)
    after = flatten_fields(corrected)
    changes = [
        FieldChange(path, _normalise(before.get(path)), _normalise(after.get(path)))
        for path in sorted(before.keys() | after.keys(), key=_path_key)
        if not _same(before.get(path), after.get(path))
    ]
    return changes


def _as_mapping(fields: BaseModel | dict[str, Any] | None) -> dict[str, Any]:
    """The pipeline holds a model; a correction coming back from the UI is a dict."""
    if fields is None:
        return {}
    data = fields.model_dump() if isinstance(fields, BaseModel) else fields
    return data if isinstance(data, dict) else {}


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _normalise(value: Any) -> Any:
    return None if _blank(value) else value


def _same(old: Any, new: Any) -> bool:
    if _blank(old) and _blank(new):
        return True
    # Before the numeric branch: bool is an int in Python, and True == 1.
    if isinstance(old, bool) or isinstance(new, bool):
        return old is new
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        return float(old) == float(new)
    return old == new


def _path_key(path: str) -> tuple[tuple[int, int, str], ...]:
    """Sort `items.2` before `items.10`, which a plain string sort does not."""
    return tuple(
        (1, int(part), "") if part.isdigit() else (0, 0, part) for part in path.split(".")
    )


__all__ = [
    "FieldChange",
    "approve_document",
    "close_pool",
    "diff_fields",
    "enabled",
    "flatten_fields",
    "get_document",
    "get_pool",
    "init_schema",
    "list_documents",
    "save_document",
]
