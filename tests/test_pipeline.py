"""app.pipeline tests: the first ones that run the whole thing end to end.

No network, no database, no recogniser. `app.llm.structured_vision` is stubbed
with a fixed answer -- the seam the vision engine calls through -- so what is
under test is everything on either side of it: loading a real encoded image,
the correction chain, the engine, `app.validate`, the review gate and the
result the caller gets back.

The two the task named come first. A synthetic receipt has to arrive as a
`ProcessResult` carrying the fields the model returned and a status that is not
"failed"; a corrupt byte string has to come back as `status="failed"` and must
not raise, because `process` is called from a web handler and from a Streamlit
form and an exception in either is a stack trace where an answer should be.

A stub is not evidence that a provider works. It is evidence that everything
around the provider works, which is a different and smaller claim, and the live
run is recorded separately.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from app import db as db_mod
from app import llm as llm_mod
from app import pipeline as pipeline_mod
from app import validate as validate_mod
from app.pipeline import agreement, compare, field_diff, process, render_pages
from app.schemas import ExtractionResult, ProcessResult, Receipt

# --- Fixtures ------------------------------------------------------------


@pytest.fixture(autouse=True)
def fixed_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every module that reads settings in this path gets a known one.

    Replaced wholesale, in each module that imported it, so a developer's real
    `.env` cannot decide an outcome -- and `persistence_enabled=False` is what
    keeps the hard constraint under test: the whole pipeline runs with no
    Postgres and only storage is skipped.
    """
    fake = SimpleNamespace(
        max_pages=20,
        pdf_dpi=150,
        max_image_px=2500,
        auto_crop=True,
        deskew=True,
        fix_lighting=True,
        amount_tolerance=0.02,
        review_confidence_threshold=0.80,
        persistence_enabled=False,
        database_url="",
        db_pool_min=1,
        db_pool_max=5,
        default_engine="vlm:gemini",
    )
    import app.engines as engines_mod
    import app.preprocess as preprocess_mod

    monkeypatch.setattr(preprocess_mod, "settings", fake)
    monkeypatch.setattr(engines_mod, "settings", fake)
    monkeypatch.setattr(validate_mod, "settings", fake)
    monkeypatch.setattr(db_mod, "settings", fake)
    monkeypatch.setattr(db_mod, "_pool", None)


ANSWER: dict[str, Any] = {
    "doc_type": "receipt",
    "merchant_name": "Al Nakheel Market",
    "merchant_name_ar": "سوق النخيل",
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
    "_transcript": "--- page 1 ---\nAl Nakheel Market\nTOTAL 115.00 SAR",
}
"""The receipt tests/test_validate.py and tests/test_db.py argue about, so all
three modules are talking about one document. It reconciles: 100.00 of lines,
15% tax, 115.00 paid."""


@pytest.fixture
def vision(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Stub the one call the vision engine makes; record what it was given.

    Patched on `app.llm` rather than on the engine, because that module
    attribute is the seam `app.engines.vlm` actually reaches through -- a stub
    anywhere closer would stop testing the engine.
    """
    calls: list[dict[str, Any]] = []

    def fake_structured_vision(prompt, images, json_schema, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"prompt": prompt, "images": list(images), "schema": json_schema, **kwargs})
        return copy.deepcopy(ANSWER)

    monkeypatch.setattr(llm_mod, "structured_vision", fake_structured_vision)
    return calls


def receipt_image(width: int = 700, height: int = 1000) -> bytes:
    """A synthetic receipt as encoded PNG bytes, the way an upload arrives.

    Encoded rather than handed over as an array on purpose: `load_document` is
    part of what is under test, and a page that never went through a decoder
    would skip it.
    """
    page = np.full((height, width, 3), 245, dtype=np.uint8)
    lines = [
        (60, "Al Nakheel Market"),
        (110, "Receipt R-0098431"),
        (150, "2024-03-14  19:05"),
        (240, "Bottled water 1.5L   4 x 2.50    10.00"),
        (290, "Basmati rice 5kg     1 x 62.00   62.00"),
        (340, "Dates 500g           2 x 14.00   28.00"),
        (430, "SUBTOTAL                        100.00"),
        (470, "VAT 15%                          15.00"),
        (520, "TOTAL                           115.00"),
    ]
    for y, text in lines:
        cv2.putText(page, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    ok, buffer = cv2.imencode(".png", page)
    assert ok
    return buffer.tobytes()


# --- The two the task named ----------------------------------------------


def test_synthetic_receipt_produces_the_expected_fields(vision: list[dict[str, Any]]) -> None:
    """A real image in, a parsed document out, and a status that is not "failed"."""
    result = process(receipt_image(), "receipt.png", "receipt", "vlm:gemini", persist=False)

    assert isinstance(result, ProcessResult)
    assert result.status == "ok"
    assert result.error is None

    document = result.extraction.document
    assert isinstance(document, Receipt)
    assert document.merchant_name == "Al Nakheel Market"
    assert document.merchant_name_ar == "سوق النخيل"
    assert document.receipt_number == "R-0098431"
    assert document.date == "2024-03-14"
    assert document.currency == "SAR"
    assert document.total == 115.00
    assert document.subtotal == 100.00
    assert [item.total for item in document.items] == [10.00, 62.00, 28.00]

    assert result.source == "receipt.png"
    assert result.doc_type == "receipt"
    assert result.extraction.engine == "vlm:gemini"
    assert result.extraction.page_count == 1
    assert result.duration_ms is not None


def test_corrupt_bytes_fail_without_raising() -> None:
    """The contract that lets a web handler call this without a try block."""
    result = process(b"\x00\x01not an image at all\xff", "junk.png", "receipt", "vlm:gemini")

    assert result.status == "failed"
    assert result.error
    assert result.extraction is None
    assert result.source == "junk.png"
    assert result.doc_type == "receipt"
    # A document that failed to extract is exactly one a human still has to deal
    # with, so it belongs in the queue rather than in a quieter category.
    assert result.needs_review is True
    assert result.stored is False


# --- Everything else on the happy path -----------------------------------


def test_a_sound_receipt_auto_approves(vision: list[dict[str, Any]]) -> None:
    """No issues, no notes, and the vision engine's None confidence is not a low score."""
    result = process(receipt_image(), "receipt.png", "receipt", "vlm:gemini", persist=False)

    assert result.issues == []
    assert result.needs_review is False
    assert result.extraction.confidence is None


def test_a_misread_total_routes_to_review(
    monkeypatch: pytest.MonkeyPatch, vision: list[dict[str, Any]]
) -> None:
    """The case the whole system exists for, run end to end for the first time.

    Every field is individually plausible and the document disagrees with
    itself by 15.00. No confidence score catches it; the arithmetic does, and
    the pipeline is what connects the two.
    """
    wrong = copy.deepcopy(ANSWER)
    wrong["total"] = 130.00
    monkeypatch.setattr(llm_mod, "structured_vision", lambda *a, **k: copy.deepcopy(wrong))

    result = process(receipt_image(), "receipt.png", "receipt", "vlm:gemini", persist=False)

    assert result.status == "ok"
    assert result.extraction.document.total == 130.00
    assert [issue.code for issue in result.issues] == ["total_mismatch"]
    assert result.needs_review is True


def test_the_transcript_becomes_raw_text(vision: list[dict[str, Any]]) -> None:
    """`_transcript` never reaches the document, and is not thrown away either."""
    result = process(receipt_image(), "receipt.png", "receipt", "vlm:gemini", persist=False)

    assert "115.00" in (result.extraction.raw_text or "")
    assert "_transcript" not in result.extraction.document.model_dump()


def test_engine_key_is_a_parameter(vision: list[dict[str, Any]]) -> None:
    """The named engine is the one that runs, and the provider reaches `app.llm`."""
    result = process(receipt_image(), "receipt.png", "receipt", "vlm:openai", persist=False)

    assert result.extraction.engine == "vlm:openai"
    assert vision[0]["provider"] == "openai"


def test_no_engine_key_uses_the_configured_default(vision: list[dict[str, Any]]) -> None:
    assert process(receipt_image(), "r.png", "receipt", persist=False).extraction.engine == "vlm:gemini"


# --- Failure, stage by stage ---------------------------------------------


def test_unknown_engine_fails_rather_than_falling_back(vision: list[dict[str, Any]]) -> None:
    """A request that named an engine must not silently get a different one."""
    result = process(receipt_image(), "receipt.png", "receipt", "vlm:llama", persist=False)

    assert result.status == "failed"
    assert "vlm:llama" in result.error
    assert vision == []


def test_unknown_doc_type_fails(vision: list[dict[str, Any]]) -> None:
    result = process(receipt_image(), "receipt.png", "manifest", "vlm:gemini", persist=False)

    assert result.status == "failed"
    assert "manifest" in result.error


def test_an_empty_upload_fails(vision: list[dict[str, Any]]) -> None:
    result = process(b"", "empty.pdf", "receipt", "vlm:gemini", persist=False)

    assert result.status == "failed"
    assert result.extraction is None


def test_a_raising_engine_does_not_escape(monkeypatch: pytest.MonkeyPatch) -> None:
    """The `Extractor` contract says engines return failures; this trusts nothing."""

    class Exploding:
        key = "vlm:gemini"

        def extract(self, *args: Any, **kwargs: Any) -> ExtractionResult:
            raise RuntimeError("provider on fire")

    monkeypatch.setattr(pipeline_mod, "get_engine", lambda key: Exploding())

    result = process(receipt_image(), "receipt.png", "receipt", "vlm:gemini", persist=False)

    assert result.status == "failed"
    assert "provider on fire" in result.error


def test_an_engine_that_returns_no_document_is_not_a_failed_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engines' own failure mode: an empty result, kept, flagged, reviewed.

    "Failed" is about the run, not about the answer. The OCR text an engine
    already paid for comes back on this path, and losing it in order to report a
    tidier status would throw away the only evidence of what was on the page.
    """

    class Empty:
        key = "vlm:gemini"

        def extract(self, pages: Any, doc_type: str, embedded_text: Any = None) -> ExtractionResult:
            return ExtractionResult(
                doc_type=doc_type,
                document=None,
                engine="vlm:gemini",
                raw_text="TOTAL 115.00",
                page_count=len(pages),
            )

    monkeypatch.setattr(pipeline_mod, "get_engine", lambda key: Empty())

    result = process(receipt_image(), "receipt.png", "receipt", "vlm:gemini", persist=False)

    assert result.status == "ok"
    assert result.error is None
    assert result.extraction.raw_text == "TOTAL 115.00"
    assert [issue.code for issue in result.issues] == ["no_document"]
    assert result.needs_review is True


def test_a_page_that_will_not_clean_is_read_as_it_arrived(
    monkeypatch: pytest.MonkeyPatch, vision: list[dict[str, Any]]
) -> None:
    """A cleanup is an improvement, not a prerequisite: losing the run over it costs more."""

    def exploding_preprocess(page: Any, **kwargs: Any) -> Any:
        raise RuntimeError("cv2 said no")

    monkeypatch.setattr(pipeline_mod, "preprocess_page", exploding_preprocess)

    result = process(receipt_image(), "receipt.png", "receipt", "vlm:gemini", persist=False)

    assert result.status == "ok"
    assert result.extraction.document is not None
    assert [issue.code for issue in result.issues] == ["page_not_cleaned"]
    assert result.needs_review is True


def test_validation_failure_keeps_the_extraction(
    monkeypatch: pytest.MonkeyPatch, vision: list[dict[str, Any]]
) -> None:
    def exploding_validate(doc_type: str, fields: Any) -> Any:
        raise RuntimeError("rule blew up")

    monkeypatch.setattr(pipeline_mod, "validate", exploding_validate)

    result = process(receipt_image(), "receipt.png", "receipt", "vlm:gemini", persist=False)

    assert result.status == "ok"
    assert result.extraction.document is not None
    assert [issue.code for issue in result.issues] == ["validation_failed"]
    assert result.needs_review is True


# --- Persistence is optional ---------------------------------------------


def test_it_runs_with_no_database(vision: list[dict[str, Any]]) -> None:
    """The hard constraint: `DATABASE_URL` empty means storage is skipped, nothing else."""
    result = process(receipt_image(), "receipt.png", "receipt", "vlm:gemini", persist=True)

    assert result.status == "ok"
    assert result.stored is False
    assert result.document_id is None


def test_persist_false_does_not_reach_the_database(
    monkeypatch: pytest.MonkeyPatch, vision: list[dict[str, Any]]
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("persist=False must not save")

    monkeypatch.setattr(db_mod, "enabled", lambda: True)
    monkeypatch.setattr(db_mod, "save_document", forbidden)

    assert process(receipt_image(), "r.png", "receipt", "vlm:gemini", persist=False).stored is False


def test_a_save_failure_is_swallowed(
    monkeypatch: pytest.MonkeyPatch, vision: list[dict[str, Any]]
) -> None:
    """Returning the extraction beats losing it."""

    def failing_save(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(db_mod, "enabled", lambda: True)
    monkeypatch.setattr(db_mod, "save_document", failing_save)

    result = process(receipt_image(), "receipt.png", "receipt", "vlm:gemini", persist=True)

    assert result.status == "ok"
    assert result.extraction.document is not None
    assert result.stored is False
    assert result.document_id is None


def test_the_content_hash_reaches_the_row(
    monkeypatch: pytest.MonkeyPatch, vision: list[dict[str, Any]]
) -> None:
    """`save_document` computes nothing; the bytes are here and nowhere later."""
    import hashlib

    seen: dict[str, Any] = {}

    def capture(result: ProcessResult, *, content_hash: str | None = None) -> str:
        seen["hash"] = content_hash
        return "11111111-2222-3333-4444-555555555555"

    monkeypatch.setattr(db_mod, "enabled", lambda: True)
    monkeypatch.setattr(db_mod, "save_document", capture)

    data = receipt_image()
    result = process(data, "receipt.png", "receipt", "vlm:gemini", persist=True)

    assert seen["hash"] == hashlib.sha256(data).hexdigest()
    assert result.stored is True
    assert result.document_id == "11111111-2222-3333-4444-555555555555"


# --- Compare mode --------------------------------------------------------


def test_compare_runs_every_engine_over_one_preparation(
    monkeypatch: pytest.MonkeyPatch, vision: list[dict[str, Any]]
) -> None:
    """One load, one cleanup, N engines -- and the same pixels to each of them."""
    loads: list[str] = []
    real_load = pipeline_mod.load_document

    def counting_load(data: bytes, filename: str, **kwargs: Any) -> Any:
        loads.append(filename)
        return real_load(data, filename, **kwargs)

    monkeypatch.setattr(pipeline_mod, "load_document", counting_load)

    results = compare(receipt_image(), "receipt.png", "receipt", ["vlm:gemini", "vlm:openai"])

    assert [r.extraction.engine for r in results] == ["vlm:gemini", "vlm:openai"]
    assert loads == ["receipt.png"]
    assert {call["provider"] for call in vision} == {"gemini", "openai"}


def test_compare_never_persists(monkeypatch: pytest.MonkeyPatch, vision: list[dict[str, Any]]) -> None:
    """A comparison is an experiment, not a record."""

    def forbidden(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("compare must not write rows")

    monkeypatch.setattr(db_mod, "enabled", lambda: True)
    monkeypatch.setattr(db_mod, "save_document", forbidden)

    results = compare(receipt_image(), "r.png", "receipt", ["vlm:gemini", "vlm:openai"])

    assert all(r.stored is False for r in results)


def test_compare_keeps_one_result_per_key_when_loading_fails() -> None:
    """A caller zipping keys to results must not have the list change length under it."""
    results = compare(b"not an image", "junk.png", "receipt", ["vlm:gemini", "vlm:openai"])

    assert len(results) == 2
    assert all(r.status == "failed" for r in results)


def test_compare_survives_one_engine_failing(vision: list[dict[str, Any]]) -> None:
    results = compare(receipt_image(), "r.png", "receipt", ["vlm:gemini", "vlm:llama"])

    assert results[0].status == "ok"
    assert results[1].status == "failed"


def test_compare_with_no_engines_returns_nothing() -> None:
    assert compare(receipt_image(), "r.png", "receipt", []) == []


# --- field_diff ----------------------------------------------------------


def result_for(engine: str, **overrides: Any) -> ProcessResult:
    fields = {key: value for key, value in ANSWER.items() if key != "_transcript"}
    fields.update(overrides)
    return ProcessResult(
        source="r.png",
        doc_type="receipt",
        extraction=ExtractionResult(
            doc_type="receipt",
            document=Receipt.model_validate(fields),
            engine=engine,
        ),
    )


def test_field_diff_marks_the_one_field_that_differs() -> None:
    rows = field_diff([result_for("vlm:gemini"), result_for("vlm:openai", total=130.00)])
    disagreed = [row for row in rows if not row.agree]

    assert [row.path for row in disagreed] == ["total"]
    assert disagreed[0].values == {"vlm:gemini": 115.00, "vlm:openai#2": 130.00}
    assert rows[0].values.keys() == {"vlm:gemini", "vlm:openai#2"}


def test_field_diff_agrees_across_json_number_shapes() -> None:
    """The same equality `app.db.diff_fields` uses: 115 and 115.0 are one number."""
    rows = field_diff([result_for("a", total=115), result_for("b", total=115.0)])

    assert all(row.agree for row in rows)
    assert agreement(rows) == 1.0


def test_field_diff_sorts_line_items_numerically() -> None:
    """`items.2` before `items.10`, as everywhere else that spells these paths."""
    many = [dict(ANSWER["items"][0], description=f"line {n}") for n in range(11)]
    rows = field_diff([result_for("a", items=many), result_for("b", items=many)])
    paths = [row.path for row in rows if row.path.startswith("items.")]

    assert paths.index("items.2.description") < paths.index("items.10.description")


def test_an_engine_that_answered_nothing_is_a_column_that_disagrees() -> None:
    """Hiding the empty column would read as consensus. It is the opposite."""
    empty = ProcessResult(
        source="r.png",
        doc_type="receipt",
        extraction=ExtractionResult(doc_type="receipt", document=None, engine="vlm:openai"),
    )
    rows = field_diff([result_for("vlm:gemini"), empty])

    assert rows
    assert not any(row.agree for row in rows)
    assert agreement(rows) == 0.0


def test_agreement_of_nothing_is_none() -> None:
    assert agreement([]) is None


def test_field_diff_of_one_result_agrees_with_itself() -> None:
    rows = field_diff([result_for("vlm:gemini")])

    assert rows
    assert all(row.agree for row in rows)


def test_compare_and_field_diff_line_up(vision: list[dict[str, Any]]) -> None:
    """The two halves of compare mode, used the way a caller would use them."""
    results = compare(receipt_image(), "r.png", "receipt", ["vlm:gemini", "vlm:openai"])
    rows = field_diff(results)

    # Both engines were handed the same stubbed answer, so they cannot disagree.
    assert agreement(rows) == 1.0


# --- render_pages --------------------------------------------------------


def test_render_pages_returns_a_png_per_page() -> None:
    pages = render_pages(receipt_image(), "receipt.png")

    assert len(pages) == 1
    assert pages[0].startswith(b"\x89PNG")
    assert cv2.imdecode(np.frombuffer(pages[0], np.uint8), cv2.IMREAD_COLOR) is not None


def test_render_pages_shows_what_the_engine_saw(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same switch, same chain: a reviewer looks at the pixels that were read."""
    calls: list[str] = []
    real = pipeline_mod.preprocess_page

    def counting(page: Any, **kwargs: Any) -> Any:
        calls.append("cleaned")
        return real(page, **kwargs)

    monkeypatch.setattr(pipeline_mod, "preprocess_page", counting)

    render_pages(receipt_image(), "receipt.png", clean_images=True)
    assert calls == ["cleaned"]

    calls.clear()
    render_pages(receipt_image(), "receipt.png", clean_images=False)
    assert calls == []


def test_render_pages_returns_nothing_on_a_file_it_cannot_open() -> None:
    """A display helper has no error channel; `process` already reported this one."""
    assert render_pages(b"\x00\x01junk", "junk.png") == []
