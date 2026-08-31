"""app.validate tests.

The case this whole module exists for is the first one below: a receipt whose
every field is individually plausible and whose arithmetic is off by 15. No
confidence score catches it -- the recogniser is as sure of the wrong digit as
of the right one -- and the only thing that does is the document disagreeing
with itself.

`settings` is replaced wholesale where it is read, so a developer's real `.env`
cannot decide an outcome.
"""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from app import validate as validate_mod
from app.schemas import Issue, Receipt
from app.validate import decide_status, validate


@pytest.fixture(autouse=True)
def fixed_settings(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    fake = SimpleNamespace(amount_tolerance=0.02, review_confidence_threshold=0.80)
    monkeypatch.setattr(validate_mod, "settings", fake)
    return fake


def codes(issues: list[Issue]) -> list[str]:
    return [issue.code for issue in issues]


def sound_receipt(**overrides: Any) -> dict[str, Any]:
    """A receipt that reconciles: 100.00 of lines, 15% tax, 115.00 paid."""
    receipt: dict[str, Any] = {
        "merchant_name": "Al Nakheel Market",
        "receipt_number": "R-0098431",
        "date": "2024-03-14",
        "time": "19:05",
        "currency": "SAR",
        "items": [
            {"description": "Bottled water 1.5L", "quantity": 4, "unit_price": 2.50, "total": 10.00},
            {"description": "Basmati rice 5kg", "quantity": 1, "unit_price": 62.00, "total": 62.00},
            {"description": "Dates 500g", "quantity": 2, "unit_price": 14.00, "total": 28.00},
        ],
        "subtotal": 100.00,
        "tax_rate": 15,
        "tax_amount": 15.00,
        "total": 115.00,
        "payment_method": "mada",
    }
    receipt.update(overrides)
    return receipt


# --- The case confidence scoring misses ----------------------------------


def test_receipt_off_by_fifteen_is_caught_although_every_field_is_plausible() -> None:
    """The total reads 130.00 on a 115.00 receipt. Nothing about 130.00 looks wrong."""
    receipt = sound_receipt(total=130.00)

    issues = validate("receipt", receipt)

    assert codes(issues) == ["total_mismatch"]
    mismatch = issues[0]
    assert mismatch.severity == "error"
    assert mismatch.field == "total"
    assert mismatch.expected == "115.00"
    assert mismatch.actual == "130.00"
    assert "+15.00" in mismatch.message

    # And a recogniser that was sure of every character does not rescue it.
    assert decide_status(issues, ocr_confidence=0.99) is True


def test_the_corrected_receipt_auto_approves() -> None:
    """Same document, total read right: no rule fires and no human is needed."""
    issues = validate("receipt", sound_receipt())

    assert issues == []
    assert decide_status(issues, ocr_confidence=0.99) is False


def test_low_confidence_with_no_issues_still_goes_to_review() -> None:
    issues = validate("receipt", sound_receipt())

    assert issues == []
    assert decide_status(issues, ocr_confidence=0.42) is True


# --- Review gate ---------------------------------------------------------


def test_a_warning_alone_routes_to_review() -> None:
    """Severity does not buy an exemption: any issue at all is a review."""
    warning = Issue(code="currency_format", message="", severity="warning")

    assert decide_status([warning], ocr_confidence=0.99) is True


def test_an_info_issue_alone_routes_to_review() -> None:
    info = Issue(code="whatever", message="", severity="info")

    assert decide_status([info], ocr_confidence=1.0) is True


def test_pipeline_warnings_route_to_review() -> None:
    assert decide_status([], ocr_confidence=0.99, warnings=["engine fell back to tesseract"]) is True


def test_missing_confidence_is_not_a_low_one() -> None:
    """The vision engine reports None by design; that must not fail every extraction."""
    assert decide_status([], ocr_confidence=None) is False


def test_confidence_exactly_at_the_threshold_passes() -> None:
    assert decide_status([], ocr_confidence=0.80) is False
    assert decide_status([], ocr_confidence=0.79) is True


def test_clean_document_with_no_confidence_and_no_warnings_approves() -> None:
    assert decide_status([]) is False


# --- Totals --------------------------------------------------------------


def test_lines_that_do_not_reach_the_subtotal_are_an_error() -> None:
    receipt = sound_receipt()
    receipt["items"][0]["total"] = 4.00  # 10.00 misread as 4.00

    issues = validate("receipt", receipt)

    assert codes(issues) == ["lines_do_not_sum", "line_total_mismatch"]
    assert issues[0].field == "subtotal"
    assert issues[0].expected == "94.00"


def test_tolerance_is_two_cents_absolute() -> None:
    assert validate("receipt", sound_receipt(total=115.02)) == []
    assert codes(validate("receipt", sound_receipt(total=115.03))) == ["total_mismatch"]


def test_discount_is_subtracted_from_the_expected_total() -> None:
    assert validate("receipt", sound_receipt(discount_total=5.00, total=110.00)) == []


def test_service_charge_and_tip_count_towards_the_total() -> None:
    """A tip printed on the receipt is part of what was paid, not a mismatch."""
    assert validate("receipt", sound_receipt(service_charge=10.00, tip=5.00, total=130.00)) == []


def test_a_missing_subtotal_runs_no_total_rule() -> None:
    """A null the extractor was right to return is not a failed reconciliation."""
    receipt = sound_receipt(subtotal=None, total=999.00)

    assert validate("receipt", receipt) == []


def test_one_unreadable_line_total_declines_the_sum_rule() -> None:
    """Short by an unknown amount is not the same as disagreeing by a known one."""
    receipt = sound_receipt()
    receipt["items"][1]["total"] = None

    assert validate("receipt", receipt) == []


def test_an_empty_item_list_declines_the_sum_rule() -> None:
    assert validate("receipt", sound_receipt(items=[])) == []


def test_invoice_totals_reconcile_through_shipping() -> None:
    invoice = {
        "invoice_number": "INV-2024-0117",
        "vendor_name": "Gulf Supplies Co.",
        "issue_date": "2024-05-02",
        "due_date": "2024-06-01",
        "currency": "AED",
        "items": [{"description": "Consulting", "quantity": 10, "unit_price": 200.00, "total": 2000.00}],
        "subtotal": 2000.00,
        "tax_amount": 100.00,
        "shipping": 50.00,
        "total": 2150.00,
    }

    assert validate("invoice", invoice) == []


# --- Line arithmetic -----------------------------------------------------


def test_quantity_times_price_is_a_warning_not_an_error() -> None:
    receipt = sound_receipt()
    receipt["items"][0]["unit_price"] = 3.00  # 4 x 3.00 = 12.00, line reads 10.00

    issues = validate("receipt", receipt)

    assert codes(issues) == ["line_total_mismatch"]
    assert issues[0].severity == "warning"
    assert issues[0].field == "items.0.total"
    assert issues[0].expected == "12.00"


def test_a_line_discount_is_taken_off_before_comparing() -> None:
    receipt = sound_receipt()
    receipt["items"][0].update({"unit_price": 3.00, "discount": 2.00})  # 12.00 - 2.00 = 10.00

    assert validate("receipt", receipt) == []


def test_a_line_missing_a_quantity_runs_no_line_rule() -> None:
    receipt = sound_receipt()
    receipt["items"][0]["quantity"] = None

    assert validate("receipt", receipt) == []


# --- Dates ---------------------------------------------------------------


def test_a_date_in_another_format_is_an_error() -> None:
    issues = validate("receipt", sound_receipt(date="14/03/2024"))

    assert codes(issues) == ["date_format"]
    assert issues[0].actual == "14/03/2024"


def test_a_well_shaped_but_impossible_date_is_an_error() -> None:
    """The shape check alone would pass 2024-02-31 straight into a DATE column."""
    assert codes(validate("receipt", sound_receipt(date="2024-02-31"))) == ["date_invalid"]


def test_a_date_before_1990_is_an_error() -> None:
    issues = validate("receipt", sound_receipt(date="1974-03-14"))

    assert codes(issues) == ["date_out_of_range"]


def test_a_future_purchase_date_is_an_error() -> None:
    ahead = (date.today() + timedelta(days=30)).isoformat()

    assert codes(validate("receipt", sound_receipt(date=ahead))) == ["date_in_future"]


def test_tomorrow_is_allowed_for_a_timezone_ahead_of_ours() -> None:
    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    assert validate("receipt", sound_receipt(date=tomorrow)) == []


def test_a_future_due_date_is_normal() -> None:
    """A due date is supposed to be ahead of today; the future rule must not fire on it."""
    invoice = {
        "invoice_number": "INV-1",
        "vendor_name": "Gulf Supplies Co.",
        "issue_date": date.today().isoformat(),
        "due_date": (date.today() + timedelta(days=60)).isoformat(),
        "total": 100.00,
    }

    assert validate("invoice", invoice) == []


def test_due_before_issue_is_an_error() -> None:
    invoice = {
        "invoice_number": "INV-1",
        "vendor_name": "Gulf Supplies Co.",
        "issue_date": "2024-05-02",
        "due_date": "2024-04-02",
        "total": 100.00,
    }

    assert codes(validate("invoice", invoice)) == ["due_before_issue"]


def test_an_unparseable_date_is_not_reported_twice() -> None:
    invoice = {
        "invoice_number": "INV-1",
        "vendor_name": "Gulf Supplies Co.",
        "issue_date": "2024-05-02",
        "due_date": "02/04/2024",
        "total": 100.00,
    }

    assert codes(validate("invoice", invoice)) == ["date_format"]


def test_a_missing_date_is_not_an_issue() -> None:
    assert validate("receipt", sound_receipt(date=None)) == []


# --- Currency ------------------------------------------------------------


def test_a_currency_that_is_not_three_uppercase_letters_is_a_warning() -> None:
    issues = validate("receipt", sound_receipt(currency="SR"))

    assert codes(issues) == ["currency_format"]
    assert issues[0].severity == "warning"


def test_a_lowercase_currency_is_a_warning() -> None:
    assert codes(validate("receipt", sound_receipt(currency="sar"))) == ["currency_format"]


def test_a_missing_currency_is_not_an_issue() -> None:
    assert validate("receipt", sound_receipt(currency=None)) == []


# --- Required fields -----------------------------------------------------


def test_a_receipt_with_no_merchant_name_is_an_error() -> None:
    issues = validate("receipt", sound_receipt(merchant_name=None))

    assert codes(issues) == ["missing_required_field"]
    assert issues[0].field == "merchant_name"


def test_an_arabic_only_merchant_name_satisfies_the_requirement() -> None:
    """Transcribing an Arabic name into the Arabic field is the policy, not a gap."""
    receipt = sound_receipt(merchant_name=None, merchant_name_ar="سوق النخيل")

    assert validate("receipt", receipt) == []


def test_a_blank_merchant_name_does_not_count_as_present() -> None:
    assert codes(validate("receipt", sound_receipt(merchant_name="   "))) == ["missing_required_field"]


def test_a_receipt_with_no_total_is_an_error() -> None:
    issues = validate("receipt", sound_receipt(total=None))

    assert codes(issues) == ["missing_required_field"]
    assert issues[0].field == "total"


def test_a_zero_total_is_present() -> None:
    """0.00 is a real reading -- a fully discounted receipt -- not a missing field."""
    assert validate("receipt", sound_receipt(subtotal=0.00, tax_amount=0.00, total=0.00, items=[])) == []


def test_an_invoice_reports_each_missing_required_field() -> None:
    issues = validate("invoice", {"issue_date": "2024-05-02"})

    assert codes(issues) == ["missing_required_field"] * 3
    assert [issue.field for issue in issues] == ["invoice_number", "vendor_name", "total"]


def test_a_generic_document_requires_nothing() -> None:
    assert validate("document", {"title": "Lease agreement"}) == []


# --- Inputs --------------------------------------------------------------


def test_a_parsed_model_validates_the_same_as_its_mapping() -> None:
    receipt = Receipt.model_validate(sound_receipt(total=130.00))

    assert codes(validate("receipt", receipt)) == ["total_mismatch"]


def test_amounts_that_arrive_as_strings_still_reconcile() -> None:
    """A correction from the review UI arrives as form values, not as floats."""
    receipt = sound_receipt(subtotal="100.00", tax_amount="15.00", total="130.00")

    assert codes(validate("receipt", receipt)) == ["total_mismatch"]


def test_no_document_is_an_error_rather_than_a_crash() -> None:
    issues = validate("receipt", None)

    assert codes(issues) == ["no_document"]
    assert decide_status(issues) is True


def test_a_non_mapping_is_an_error() -> None:
    assert codes(validate("receipt", ["merchant"])) == ["not_an_object"]


def test_an_unknown_doc_type_raises() -> None:
    """A caller bug, not a document problem: it must not come back as a clean sheet."""
    with pytest.raises(ValueError):
        validate("purchase_order", sound_receipt())
