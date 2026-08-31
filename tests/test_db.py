"""app.db tests: the flatten/diff pair, and the no-op contract. No database.

Two things are pinned here. The first is that `diff_fields` produces a
correction row for every change a reviewer made and for nothing else, because
both halves of that cost something real: a change it misses is a training pair
thrown away at the moment it was created and never recoverable, and a change it
invents is a page entered into the eval set as one the engine got wrong when it
did not.

The second is that with `DATABASE_URL` empty every entry point returns its
empty answer without building a pool, which is the hard constraint that the
pipeline runs with no Postgres. `settings` is replaced wholesale, so a
developer's real `.env` cannot decide either outcome.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app import db as db_mod
from app.db import FieldChange, diff_fields, flatten_fields
from app.schemas import ExtractionResult, Issue, ProcessResult, Receipt


@pytest.fixture(autouse=True)
def no_database(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Persistence off by default in every test here, and no pool left behind."""
    fake = SimpleNamespace(
        persistence_enabled=False,
        database_url="",
        db_pool_min=1,
        db_pool_max=5,
    )
    monkeypatch.setattr(db_mod, "settings", fake)
    monkeypatch.setattr(db_mod, "_pool", None)
    return fake


def receipt_fields(**overrides: Any) -> dict[str, Any]:
    """The receipt from tests/test_validate.py, so the two modules argue about one document."""
    fields: dict[str, Any] = {
        "doc_type": "receipt",
        "merchant_name": "Al Nakheel Market",
        "merchant_name_ar": "سوق النخيل",
        "receipt_number": "R-0098431",
        "date": "2024-03-14",
        "currency": "SAR",
        "items": [
            {"description": "Bottled water 1.5L", "quantity": 4, "unit_price": 2.50, "total": 10.00},
            {"description": "Basmati rice 5kg", "quantity": 1, "unit_price": 62.00, "total": 62.00},
            {"description": "Dates 500g", "quantity": 2, "unit_price": 14.00, "total": 28.00},
        ],
        "subtotal": 100.00,
        "tax_amount": 15.00,
        "total": 115.00,
    }
    fields.update(overrides)
    return fields


def paths(changes: list[FieldChange]) -> list[str]:
    return [change.path for change in changes]


# --- flatten -------------------------------------------------------------


def test_flatten_scalars_keep_their_names() -> None:
    flat = flatten_fields({"merchant_name": "Al Nakheel Market", "total": 115.0})
    assert flat == {"merchant_name": "Al Nakheel Market", "total": 115.0}


def test_flatten_indexes_list_items() -> None:
    flat = flatten_fields(receipt_fields())
    assert flat["items.0.description"] == "Bottled water 1.5L"
    assert flat["items.2.total"] == 28.00
    assert "items" not in flat  # a non-empty container is not itself a leaf


def test_flatten_paths_are_spelled_the_way_validate_spells_issue_field() -> None:
    """app.validate reports `items.2.total`; a correction on it must use the same string."""
    issue = Issue(code="line_total_mismatch", message="x", severity="warning", field="items.2.total")
    assert issue.field in flatten_fields(receipt_fields())


def test_flatten_nested_dict() -> None:
    flat = flatten_fields({"a": {"b": {"c": 1}}})
    assert flat == {"a.b.c": 1}


def test_flatten_empty_containers_are_leaves() -> None:
    """Otherwise 'the reviewer deleted every line' and 'there were never any lines' look alike."""
    assert flatten_fields({"items": [], "meta": {}}) == {"items": [], "meta": {}}


def test_flatten_keeps_explicit_nulls() -> None:
    assert flatten_fields({"branch": None}) == {"branch": None}


def test_flatten_accepts_a_pydantic_model() -> None:
    flat = flatten_fields(Receipt.model_validate(receipt_fields()))
    assert flat["total"] == 115.0
    assert flat["items.1.description"] == "Basmati rice 5kg"


@pytest.mark.parametrize("value", [None, [], "not a document", 7])
def test_flatten_of_a_non_document_is_empty(value: Any) -> None:
    assert flatten_fields(value) == {}


# --- diff: what counts as a change ---------------------------------------


def test_diff_of_an_untouched_document_is_empty() -> None:
    assert diff_fields(receipt_fields(), receipt_fields()) == []


def test_diff_reports_the_one_field_that_moved() -> None:
    """The case app.validate flags: a total that reads 130.00 on a 115.00 receipt."""
    stored = receipt_fields(total=130.00)
    changes = diff_fields(stored, receipt_fields())
    assert changes == [FieldChange("total", 130.00, 115.00)]


def test_diff_reports_every_field_that_moved() -> None:
    stored = receipt_fields(total=130.00, merchant_name="AI Nakheel Markel")
    changes = diff_fields(stored, receipt_fields())
    assert paths(changes) == ["merchant_name", "total"]


def test_diff_inside_a_line_item() -> None:
    corrected = receipt_fields()
    corrected["items"][1]["unit_price"] = 82.00
    changes = diff_fields(receipt_fields(), corrected)
    assert changes == [FieldChange("items.1.unit_price", 62.00, 82.00)]


def test_diff_ignores_a_number_that_only_changed_type() -> None:
    """12 and 12.0 are one number that went through JSON, not a correction."""
    assert diff_fields({"total": 12}, {"total": 12.0}) == []


def test_diff_ignores_a_missing_key_against_an_explicit_null() -> None:
    """A form that omits a field and one that sends null both say 'not on the document'."""
    assert diff_fields({"branch": None, "total": 1}, {"total": 1}) == []
    assert diff_fields({"total": 1}, {"branch": None, "total": 1}) == []


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_diff_ignores_a_blank_against_a_null(blank: Any) -> None:
    assert diff_fields({"branch": None}, {"branch": blank}) == []


def test_diff_records_a_cleared_field_as_null_not_as_an_empty_string() -> None:
    """The reviewer's blank box means 'not on the page', which is what the extractor must answer."""
    changes = diff_fields({"branch": "Branch 12"}, {"branch": "  "})
    assert changes == [FieldChange("branch", "Branch 12", None)]


def test_diff_does_not_confuse_a_boolean_with_one() -> None:
    assert diff_fields({"flag": True}, {"flag": 1}) == [FieldChange("flag", True, 1)]


def test_diff_treats_whitespace_inside_a_value_as_significant() -> None:
    """Only a *wholly* blank value is an absence; a trimmed name is still an edit."""
    assert diff_fields({"merchant_name": " Souq "}, {"merchant_name": "Souq"}) == [
        FieldChange("merchant_name", " Souq ", "Souq")
    ]


# --- diff: lines appearing and disappearing ------------------------------


def test_diff_records_a_line_the_reviewer_added() -> None:
    corrected = receipt_fields()
    corrected["items"].append({"description": "Bag", "quantity": 1, "unit_price": 1.00, "total": 1.00})
    changes = diff_fields(receipt_fields(), corrected)
    assert paths(changes) == [
        "items.3.description",
        "items.3.quantity",
        "items.3.total",
        "items.3.unit_price",
    ]
    assert all(change.old is None for change in changes)


def test_diff_records_a_document_whose_lines_were_all_deleted() -> None:
    """Every leaf goes to null, and `items: []` is the row that says there are none left."""
    changes = diff_fields(receipt_fields(), receipt_fields(items=[]))
    by_path = {change.path: change for change in changes}
    assert by_path["items"] == FieldChange("items", None, [])
    assert by_path["items.0.total"] == FieldChange("items.0.total", 10.00, None)
    assert all(change.new is None for change in changes if change.path != "items")


# --- diff: order ---------------------------------------------------------


def test_diff_orders_line_numbers_numerically() -> None:
    """A plain string sort puts items.10 before items.2, which reads as a bug in the UI."""
    stored = {"items": [{"total": float(n)} for n in range(12)]}
    corrected = {"items": [{"total": float(n) + 1} for n in range(12)]}
    assert paths(diff_fields(stored, corrected)) == [f"items.{n}.total" for n in range(12)]


def test_diff_orders_a_parent_before_its_children() -> None:
    stored = {"items": [], "total": 1.0}
    corrected = {"items": [{"total": 5.0}], "total": 2.0}
    assert paths(diff_fields(stored, corrected)) == ["items", "items.0.total", "total"]


# --- The no-op contract --------------------------------------------------


def test_enabled_is_false_without_a_url() -> None:
    assert db_mod.enabled() is False


def test_get_pool_returns_none_and_builds_nothing() -> None:
    assert db_mod.get_pool() is None
    assert db_mod._pool is None


def test_init_schema_is_a_no_op() -> None:
    assert db_mod.init_schema() is False


def test_save_document_stores_nothing_and_does_not_raise() -> None:
    result = ProcessResult(
        source="receipt.jpg",
        doc_type="receipt",
        extraction=ExtractionResult(
            doc_type="receipt",
            document=Receipt.model_validate(receipt_fields()),
            engine="traditional:paddle",
        ),
        needs_review=True,
    )
    assert db_mod.save_document(result) is None


def test_list_documents_is_empty() -> None:
    assert db_mod.list_documents() == []
    assert db_mod.list_documents(status="pending_review", doc_type="invoice", limit=5) == []


def test_get_document_is_none() -> None:
    assert db_mod.get_document("2b0f0b3e-0000-4000-8000-000000000000") is None


def test_approve_document_is_false() -> None:
    assert db_mod.approve_document("2b0f0b3e-0000-4000-8000-000000000000", receipt_fields()) is False


def test_no_entry_point_imports_a_driver_when_persistence_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hard constraint, checked rather than assumed: no psycopg, no problem."""
    import builtins

    real_import = builtins.__import__

    def refuse_psycopg(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("psycopg"):
            raise AssertionError(f"imported {name} with persistence off")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_psycopg)
    assert db_mod.get_pool() is None
    assert db_mod.init_schema() is False
    assert db_mod.list_documents() == []
    assert db_mod.get_document("2b0f0b3e-0000-4000-8000-000000000000") is None
    assert db_mod.approve_document("2b0f0b3e-0000-4000-8000-000000000000", {}) is False
    assert db_mod.save_document(ProcessResult(doc_type="receipt")) is None
