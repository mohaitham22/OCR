"""Arithmetic, format and completeness rules, and the review gate.

This is the layer that stops wrong numbers reaching the database, and it does
that with arithmetic rather than with confidence. A recogniser reporting 0.97
on a digit it read wrong is reporting how certain it is that those pixels are
the glyph it chose, which is a claim about ink and not about the number: a 3
read as an 8 comes back confident, and every field on the document stays
individually plausible. What that misread cannot survive is the document
checking itself -- the lines still have to sum to the subtotal, and the
subtotal still has to reach the total.

Every rule declines rather than guesses. A rule whose inputs are not all
present reports nothing, because a missing subtotal is a null the extractor was
right to return and not a failed reconciliation. Only a check that actually ran
and actually disagreed produces an `Issue`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import date as date_type, datetime, timedelta
from typing import Any

from pydantic import BaseModel

from app.config import settings
from app.schemas import Issue, schema_for

logger = logging.getLogger(__name__)

_DATE_SHAPE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CURRENCY_SHAPE = re.compile(r"^[A-Z]{3}$")

# A scanned business document dated before this is a misread year far more
# often than it is a real archive scan.
_EARLIEST = date_type(1990, 1, 1)

# A document dated tomorrow was issued in a timezone ahead of ours, or on a
# till whose clock drifts. Two days out is a misread.
_FUTURE_SLACK = timedelta(days=1)

# Required fields, as (label, accepted spellings). The Arabic variant satisfies
# the requirement: the extraction policy is that an Arabic-only name goes into
# the Arabic field in Arabic script, so demanding the Latin field would fail
# every Arabic receipt for obeying it.
_REQUIRED: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "receipt": (
        ("merchant_name", ("merchant_name", "merchant_name_ar")),
        ("total", ("total",)),
    ),
    "invoice": (
        ("invoice_number", ("invoice_number",)),
        # The seller. app.schemas spells it vendor_name.
        ("vendor_name", ("vendor_name", "vendor_name_ar")),
        ("total", ("total",)),
    ),
    "document": (),
}

# Date fields per doc type, with whether a date after today is legitimate. A
# due date and the end of a service period are meant to be in the future; a
# purchase date is not, and one that is means the day and month were swapped.
_DATE_FIELDS: dict[str, tuple[tuple[str, bool], ...]] = {
    "receipt": (("date", False),),
    "invoice": (
        ("issue_date", False),
        ("due_date", True),
        ("service_period_start", False),
        ("service_period_end", True),
    ),
    "document": (("date", False),),
}

# Charges carried in the printed total alongside tax. An absent field is
# absent, not a zero the document forgot to print.
_ADDITIVE = ("tax_amount", "service_charge", "tip", "shipping")


def validate(doc_type: str, fields: BaseModel | dict[str, Any] | None) -> list[Issue]:
    """Every rule that applies to `doc_type`, run against `fields`.

    Takes the parsed document or the mapping behind it, because the pipeline
    holds a model and a correction coming back from the review UI is a dict.
    """
    schema_for(doc_type)  # an unknown doc_type is a caller bug: raise before validating nothing
    key = doc_type.strip().lower()

    if fields is None:
        return [
            Issue(
                code="no_document",
                message="extraction produced no document, so nothing could be checked",
                severity="error",
            )
        ]

    data = fields.model_dump() if isinstance(fields, BaseModel) else fields
    if not isinstance(data, dict):
        return [
            Issue(
                code="not_an_object",
                message=f"expected a {key} object, got {type(data).__name__}",
                severity="error",
            )
        ]

    issues: list[Issue] = []
    issues.extend(_check_required(key, data))
    issues.extend(_check_dates(key, data))
    issues.extend(_check_currency(data))
    if key in ("receipt", "invoice"):
        issues.extend(_check_totals(data))
        issues.extend(_check_line_arithmetic(data))
    if key == "invoice":
        issues.extend(_check_due_after_issue(data))
    return issues


def decide_status(
    issues: Sequence[Issue],
    ocr_confidence: float | None = None,
    warnings: Sequence[Any] | None = None,
) -> bool:
    """True when a human has to see this document before it is trusted.

    Deliberately conservative: *any* issue routes to review, whatever its
    severity, and so does a confidence below the threshold on a clean rule
    sheet. The asymmetry is the whole argument. A wrong number that reaches the
    database silently is found months later by someone reconciling accounts,
    with every downstream report already built on it and no record of which
    document it came from; a reviewer glancing at a document that turns out to
    be fine costs seconds. Buy the cheap error every time.

    An `ocr_confidence` of None is not a low score. The vision engine has no
    recogniser and reports None by design, and the PDF text-layer path reports
    None because nothing was recognised there either. Treating a missing score
    as a failing one would send both to review on the strength of a number
    nobody claimed.
    """
    if issues:
        return True
    if warnings:
        return True
    if ocr_confidence is not None and ocr_confidence < settings.review_confidence_threshold:
        return True
    return False


# --- Completeness --------------------------------------------------------


def _check_required(key: str, data: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    for label, spellings in _REQUIRED.get(key, ()):
        if any(_present(data.get(name)) for name in spellings):
            continue
        issues.append(
            Issue(
                code="missing_required_field",
                message=f"{key} has no {label}; it cannot be filed without one",
                severity="error",
                field=spellings[0],
            )
        )
    return issues


def _present(value: Any) -> bool:
    if value is None:
        return False
    return bool(value.strip()) if isinstance(value, str) else True


# --- Dates ---------------------------------------------------------------


def _check_dates(key: str, data: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    today = datetime.now().date()
    for name, may_be_future in _DATE_FIELDS.get(key, ()):
        if not _present(data.get(name)):
            continue
        parsed, problem = _parse_date(name, data[name])
        if problem is not None:
            issues.append(problem)
            continue
        if parsed is None:
            continue
        if parsed < _EARLIEST:
            issues.append(
                Issue(
                    code="date_out_of_range",
                    message=f"{name} {parsed.isoformat()} is before 1990; the year was probably misread",
                    severity="error",
                    field=name,
                    expected=f"on or after {_EARLIEST.isoformat()}",
                    actual=parsed.isoformat(),
                )
            )
        elif not may_be_future and parsed > today + _FUTURE_SLACK:
            issues.append(
                Issue(
                    code="date_in_future",
                    message=f"{name} {parsed.isoformat()} is in the future; day and month may be swapped",
                    severity="error",
                    field=name,
                    expected=f"on or before {today.isoformat()}",
                    actual=parsed.isoformat(),
                )
            )
    return issues


def _parse_date(name: str, raw: Any) -> tuple[date_type | None, Issue | None]:
    """A date has to be YYYY-MM-DD *and* be a real day: 2024-02-31 is neither."""
    if isinstance(raw, datetime):
        return raw.date(), None
    if isinstance(raw, date_type):
        return raw, None
    text = raw.strip() if isinstance(raw, str) else str(raw)
    if not _DATE_SHAPE.match(text):
        return None, Issue(
            code="date_format",
            message=f"{name} {text!r} is not written as YYYY-MM-DD",
            severity="error",
            field=name,
            expected="YYYY-MM-DD",
            actual=text,
        )
    try:
        return date_type.fromisoformat(text), None
    except ValueError:
        return None, Issue(
            code="date_invalid",
            message=f"{name} {text} is not a real calendar date",
            severity="error",
            field=name,
            expected="an existing calendar date",
            actual=text,
        )


def _check_due_after_issue(data: dict[str, Any]) -> list[Issue]:
    issued = _parse_date("issue_date", data["issue_date"]) if _present(data.get("issue_date")) else (None, None)
    due = _parse_date("due_date", data["due_date"]) if _present(data.get("due_date")) else (None, None)
    if issued[0] is None or due[0] is None:
        return []  # a date that would not parse is already reported; not twice, as an ordering fault
    if due[0] < issued[0]:
        return [
            Issue(
                code="due_before_issue",
                message=f"due_date {due[0].isoformat()} is before issue_date {issued[0].isoformat()}",
                severity="error",
                field="due_date",
                expected=f"on or after {issued[0].isoformat()}",
                actual=due[0].isoformat(),
            )
        ]
    return []


# --- Currency ------------------------------------------------------------


def _check_currency(data: dict[str, Any]) -> list[Issue]:
    """A warning, not an error: the amounts are still right, we just cannot name the unit."""
    if not _present(data.get("currency")):
        return []
    raw = data["currency"]
    text = raw.strip() if isinstance(raw, str) else str(raw)
    if _CURRENCY_SHAPE.match(text):
        return []
    return [
        Issue(
            code="currency_format",
            message=f"currency {text!r} is not a three-letter ISO 4217 code",
            severity="warning",
            field="currency",
            expected="three uppercase letters, e.g. SAR",
            actual=text,
        )
    ]


# --- Arithmetic ----------------------------------------------------------


def _check_totals(data: dict[str, Any]) -> list[Issue]:
    """The two document-level reconciliations, both errors.

    The tolerance is absolute currency, not a ratio: per-item tax and per-item
    rounding put a correct document a cent or two out, and no amount of
    proportion makes that scale with the size of the bill.
    """
    issues: list[Issue] = []
    tol = settings.amount_tolerance

    subtotal = _num(data.get("subtotal"))
    total = _num(data.get("total"))
    discount = _num(data.get("discount_total")) or 0.0

    line_sum = _line_total_sum(data)
    if subtotal is not None and line_sum is not None and abs(line_sum - subtotal) > tol:
        issues.append(
            Issue(
                code="lines_do_not_sum",
                message=(
                    f"line totals sum to {line_sum:.2f} but the subtotal reads {subtotal:.2f}, "
                    f"a difference of {line_sum - subtotal:+.2f}"
                ),
                severity="error",
                field="subtotal",
                expected=f"{line_sum:.2f}",
                actual=f"{subtotal:.2f}",
            )
        )

    if subtotal is not None and total is not None:
        charges = sum(value for value in (_num(data.get(name)) for name in _ADDITIVE) if value is not None)
        expected = subtotal + charges - discount
        if abs(expected - total) > tol:
            issues.append(
                Issue(
                    code="total_mismatch",
                    message=(
                        f"subtotal plus charges minus discount comes to {expected:.2f} "
                        f"but the total reads {total:.2f}, a difference of {total - expected:+.2f}"
                    ),
                    severity="error",
                    field="total",
                    expected=f"{expected:.2f}",
                    actual=f"{total:.2f}",
                )
            )
    return issues


def _line_total_sum(data: dict[str, Any]) -> float | None:
    """Sum of the printed line totals, or None where the sum would be meaningless.

    One line whose total was unreadable leaves the sum short by an unknown
    amount, and reporting that as a reconciliation failure would point the
    reviewer at the subtotal instead of at the line that is actually missing --
    which `coerce_fields` has already warned about.
    """
    items = data.get("items")
    if not isinstance(items, list) or not items:
        return None
    running = 0.0
    for item in items:
        if not isinstance(item, dict):
            return None
        value = _num(item.get("total"))
        if value is None:
            return None
        running += value
    return running


def _check_line_arithmetic(data: dict[str, Any]) -> list[Issue]:
    """quantity x unit_price - discount against the printed line total.

    A warning rather than an error: a line that disagrees does not by itself
    make the total the document is filed for wrong, and weighed goods priced
    per kilo round at the line far more often than the bill fails to add up.
    """
    items = data.get("items")
    if not isinstance(items, list):
        return []
    tol = settings.amount_tolerance
    issues: list[Issue] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        quantity = _num(item.get("quantity"))
        unit_price = _num(item.get("unit_price"))
        line_total = _num(item.get("total"))
        if quantity is None or unit_price is None or line_total is None:
            continue
        expected = quantity * unit_price - (_num(item.get("discount")) or 0.0)
        if abs(expected - line_total) > tol:
            issues.append(
                Issue(
                    code="line_total_mismatch",
                    message=(
                        f"line {index + 1}: {quantity:g} x {unit_price:.2f} comes to {expected:.2f} "
                        f"but the line reads {line_total:.2f}"
                    ),
                    severity="warning",
                    field=f"items.{index}.total",
                    expected=f"{expected:.2f}",
                    actual=f"{line_total:.2f}",
                )
            )
    return issues


def _num(value: Any) -> float | None:
    """A float, or None for anything a rule must not run on -- bool included."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None
