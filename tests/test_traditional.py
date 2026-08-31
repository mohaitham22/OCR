"""app.engines.traditional tests.

The recognisers are never imported. What is worth pinning here is everything
between them and the model: that detection order is turned back into reading
order, that a mixed Arabic/Latin row survives that, that the PDF text layer
short-circuits the recogniser entirely, and that no failure downstream of the
pixels leaves the engine raising instead of returning an empty result.

Every reading-order test feeds its blocks scrambled. Pre-sorted input proves
nothing: the identity function passes it.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from app.engines import traditional
from app.engines.traditional import (
    TraditionalExtractor,
    _find_amount,
    _find_currency,
    _find_date,
    _find_label_value,
    _first_named_line,
    _overall_confidence,
    _paddle_lines,
    _structure,
    _SUBTOTAL_MARKERS,
    _tesseract_lines,
    _TOTAL_LABELS,
    _TOTAL_LABELS_LOOSE,
    reading_order_text,
)
from app.schemas import TextBlock


def block(text: str, x0: float, y0: float, x1: float, y1: float, **kwargs: Any) -> TextBlock:
    return TextBlock(text=text, bbox=[x0, y0, x1, y1], **kwargs)


def page() -> np.ndarray:
    return np.zeros((40, 40, 3), dtype=np.uint8)


# --- Reading order -------------------------------------------------------


def test_scrambled_blocks_are_rebuilt_into_rows() -> None:
    """A receipt read bottom-up and right-to-left comes back as the printed page."""
    printed = [
        block("ACME Store", 40, 10, 160, 30),
        block("Item", 10, 50, 60, 70),
        block("Qty", 200, 50, 240, 70),
        block("Price", 350, 50, 410, 70),
        block("Water", 10, 90, 70, 110),
        block("2", 200, 90, 215, 110),
        block("5.00", 350, 90, 400, 110),
        block("TOTAL", 10, 150, 80, 170),
        block("10.00", 350, 150, 410, 170),
    ]
    scrambled = list(reversed(printed))

    assert reading_order_text(scrambled) == (
        "ACME Store\nItem Qty Price\nWater 2 5.00\nTOTAL 10.00"
    )


def test_reading_order_is_independent_of_detection_order() -> None:
    blocks = [
        block("one", 0, 0, 30, 20),
        block("two", 40, 0, 70, 20),
        block("three", 0, 60, 40, 80),
        block("four", 50, 60, 90, 80),
    ]
    expected = "one two\nthree four"

    rng = random.Random(0)
    for _ in range(20):
        shuffled = blocks[:]
        rng.shuffle(shuffled)
        assert reading_order_text(shuffled) == expected


def test_a_total_detected_out_of_its_row_returns_to_it() -> None:
    """The failure this function exists for: an amount arriving before its label."""
    blocks = [
        block("120.00", 300, 40, 360, 60),
        block("Subtotal", 10, 40, 90, 60),
    ]

    assert reading_order_text(blocks) == "Subtotal 120.00"


def test_mixed_arabic_and_latin_stay_in_box_order() -> None:
    """Pins the deliberate choice: boxes are never reversed for Arabic.

    The recogniser emits each box's characters in logical order already. A row
    with an Arabic label on the left and a Latin amount on the right must come
    back label-then-amount; reversing boxes by script would swap them.
    """
    blocks = [
        block("15.00", 300, 20, 350, 40),
        block("الإجمالي", 10, 20, 90, 40),
        block("VAT", 150, 20, 190, 40),
    ]

    assert reading_order_text(blocks) == "الإجمالي VAT 15.00"


def test_blocks_within_the_tolerance_join_one_line() -> None:
    """Centres 8px apart: a row whose boxes are not perfectly aligned."""
    blocks = [
        block("right", 200, 16, 260, 36),  # centre 26
        block("left", 10, 8, 60, 28),  # centre 18
    ]

    assert reading_order_text(blocks, line_tolerance=12) == "left right"


def test_the_tolerance_boundary_is_inclusive() -> None:
    pair = [block("a", 0, 0, 20, 20), block("b", 100, 12, 120, 32)]  # centres 10 and 22

    assert reading_order_text(pair, line_tolerance=12) == "a b"
    assert reading_order_text(pair, line_tolerance=11) == "a\nb"


def test_a_drifting_column_does_not_swallow_the_page() -> None:
    """The running mean is what stops each near-miss extending the same line."""
    blocks = [block(str(n), 0, n * 10, 20, n * 10 + 20) for n in range(4)]  # centres 10..40

    assert reading_order_text(blocks, line_tolerance=12) == "0 1\n2 3"


def test_pages_are_kept_apart_and_in_order() -> None:
    blocks = [
        block("second page", 0, 0, 80, 20, page=2),
        block("first page", 0, 0, 80, 20, page=1),
    ]

    assert reading_order_text(blocks) == "first page\n\nsecond page"


def test_blocks_without_boxes_keep_their_arrival_order_at_the_end() -> None:
    blocks = [
        TextBlock(text="no box first"),
        block("placed", 0, 0, 40, 20),
        TextBlock(text="no box second"),
    ]

    assert reading_order_text(blocks) == "placed\nno box first\nno box second"


def test_empty_and_blank_blocks_produce_no_lines() -> None:
    assert reading_order_text([]) == ""
    assert reading_order_text([block("   ", 0, 0, 10, 10)]) == ""


# --- Backend identity ----------------------------------------------------


def test_each_backend_is_its_own_registry_entry() -> None:
    paddle = TraditionalExtractor(backend="paddle")
    tesseract = TraditionalExtractor(backend="tesseract")

    assert paddle.key == "traditional:paddle"
    assert tesseract.key == "traditional:tesseract"
    assert paddle.label != tesseract.label
    assert paddle.gives_boxes and tesseract.gives_boxes


def test_an_unknown_backend_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown OCR backend"):
        TraditionalExtractor(backend="easyocr")


# --- extract ---------------------------------------------------------------
# Stage two is rule-based now: no model to stub, so these exercise `extract`
# end to end -- recognised (or embedded) text in, a coerced document out --
# and the rule-matching itself is pinned separately, below.


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> TraditionalExtractor:
    """A Paddle extractor whose recogniser is stubbed out, never imported."""
    extractor = TraditionalExtractor(backend="paddle")
    monkeypatch.setattr(
        extractor,
        "_recognise",
        lambda pages: pytest.fail("the recogniser should not have run"),
    )
    return extractor


def test_an_embedded_text_layer_skips_the_recogniser(engine: TraditionalExtractor) -> None:
    """`engine` fails the test if its recogniser runs, so this asserts by not failing."""
    result = engine.extract([page()], "receipt", embedded_text="  Bakery\nTOTAL 10.00  ")

    assert result.raw_text == "Bakery\nTOTAL 10.00"
    assert result.text_blocks == []
    assert result.document is not None
    assert result.document.merchant_name == "Bakery"
    assert result.document.total == 10.0


def test_the_text_layer_path_reports_no_ocr_confidence(engine: TraditionalExtractor) -> None:
    result = engine.extract([page()], "receipt", embedded_text="TOTAL 10.00")

    assert result.confidence is None


def test_a_blank_text_layer_falls_through_to_the_recogniser(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = TraditionalExtractor(backend="paddle")
    monkeypatch.setattr(extractor, "_recognise", lambda pages: [block("Kiosk", 0, 0, 50, 20)])

    result = extractor.extract([page()], "receipt", embedded_text="   \n  ")

    assert result.raw_text == "Kiosk"


def test_the_result_carries_reordered_ocr_not_detection_order(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = TraditionalExtractor(backend="tesseract")
    scrambled = [
        block("10.00", 300, 60, 350, 80, confidence=0.90),
        block("Cafe", 10, 10, 70, 30, confidence=0.99),
        block("TOTAL", 10, 60, 70, 80, confidence=0.95),
    ]
    monkeypatch.setattr(extractor, "_recognise", lambda pages: scrambled)

    result = extractor.extract([page()], "receipt")

    assert result.raw_text == "Cafe\nTOTAL 10.00"
    assert result.engine == "traditional:tesseract"
    assert result.text_blocks == scrambled
    assert result.page_count == 1
    assert result.duration_ms is not None
    assert result.document is not None
    assert result.document.merchant_name == "Cafe"
    assert result.document.total == 10.0


def test_a_page_with_no_text_returns_an_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = TraditionalExtractor(backend="paddle")
    monkeypatch.setattr(extractor, "_recognise", lambda pages: [])

    result = extractor.extract([page()], "invoice")

    assert result.document is None
    assert result.doc_type == "invoice"


def test_a_recogniser_that_will_not_load_returns_an_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing paddleocr must not take a compare-mode run down with it."""
    extractor = TraditionalExtractor(backend="paddle")

    def explode(pages: Any) -> list[TextBlock]:
        raise RuntimeError("paddleocr is not installed; run `pip install paddleocr paddlepaddle`")

    monkeypatch.setattr(extractor, "_recognise", explode)

    result = extractor.extract([page()], "receipt")

    assert result.document is None
    assert result.engine == "traditional:paddle"


def test_a_failing_structuring_step_keeps_the_ocr_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never-raise holds even if the rule-matching itself blows up."""
    extractor = TraditionalExtractor(backend="paddle")
    monkeypatch.setattr(extractor, "_recognise", lambda pages: [block("Cafe", 0, 0, 50, 20)])
    monkeypatch.setattr(traditional, "_structure", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))

    result = extractor.extract([page()], "receipt")

    assert result.document is None
    assert result.raw_text == "Cafe"  # the expensive half is still on the result


def test_an_unusable_answer_comes_back_as_an_empty_result(
    engine: TraditionalExtractor, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(traditional, "_structure", lambda *a, **k: ["not", "an", "object"])

    result = engine.extract([page()], "receipt", embedded_text="TOTAL 10.00")

    assert result.document is None


def test_one_bad_field_does_not_lose_the_document(
    engine: TraditionalExtractor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """coerce_fields's prune-and-retry still applies to whatever _structure returns."""
    answer = {"merchant_name": "ACME", "total": "10.00", "items": [{"description": "Tea", "quantity": "two"}]}
    monkeypatch.setattr(traditional, "_structure", lambda *a, **k: answer)

    result = engine.extract([page()], "receipt", embedded_text="TOTAL 10.00")

    assert result.document is not None
    assert result.document.total == 10.0
    assert result.document.items[0].description == "Tea"
    assert result.document.items[0].quantity is None


def test_an_unknown_doc_type_is_a_caller_bug_and_raises(engine: TraditionalExtractor) -> None:
    with pytest.raises(ValueError, match="unknown doc_type"):
        engine.extract([page()], "manifest", embedded_text="anything")


# --- Rule-based structuring --------------------------------------------------


def test_structure_reads_a_typical_receipt() -> None:
    text = "\n".join(
        [
            "Corner Bakery",
            "123 Main St",
            "Bread          1  5.00",
            "Milk           2  3.50",
            "Subtotal          12.00",
            "VAT 15%            1.80",
            "Total             13.80",
        ]
    )

    fields = _structure("receipt", text)

    assert fields["merchant_name"] == "Corner Bakery"
    assert fields["merchant_name_ar"] is None
    assert fields["total"] == "13.80"


def test_total_is_not_confused_with_subtotal() -> None:
    text = "Shop\nSubtotal 100.00\nTotal 115.00"

    assert _find_amount(text, _TOTAL_LABELS, _TOTAL_LABELS_LOOSE, _SUBTOTAL_MARKERS) == "115.00"


def test_a_bare_total_label_still_matches_when_no_subtotal_is_printed() -> None:
    text = "Shop\nItems 2\nTotal 42.00"

    assert _find_amount(text, _TOTAL_LABELS, _TOTAL_LABELS_LOOSE, _SUBTOTAL_MARKERS) == "42.00"


def test_no_matching_label_returns_no_amount() -> None:
    text = "Shop\nItems 2\nThank you for visiting"

    assert _find_amount(text, _TOTAL_LABELS, _TOTAL_LABELS_LOOSE, _SUBTOTAL_MARKERS) is None


def test_arabic_first_line_goes_into_the_arabic_name_field() -> None:
    latin, arabic = _first_named_line("مطعم الرياض\nTotal 20.00")

    assert latin is None
    assert arabic == "مطعم الرياض"


def test_latin_first_line_goes_into_the_latin_name_field() -> None:
    latin, arabic = _first_named_line("Riyadh Restaurant\nTotal 20.00")

    assert latin == "Riyadh Restaurant"
    assert arabic is None


def test_blank_leading_lines_are_skipped_before_the_name() -> None:
    latin, _ = _first_named_line("\n   \nActual Name\nTotal 1.00")

    assert latin == "Actual Name"


def test_invoice_number_is_read_from_its_label() -> None:
    text = "Some Vendor\nInvoice No: INV-2024-00123\nTotal 500.00"

    assert _find_label_value(text, ("invoice no", "invoice number")) == "INV-2024-00123"


def test_label_with_no_line_returns_none() -> None:
    assert _find_label_value("Vendor\nTotal 500.00", ("invoice no",)) is None


def test_date_iso_is_read_directly() -> None:
    assert _find_date("Issued 2024-04-03") == "2024-04-03"


def test_date_dmy_is_read_day_first() -> None:
    """Matches the model prompt's own rule: an ambiguous all-numeric date is day-month-year."""
    assert _find_date("Date: 03/04/2024") == "2024-04-03"


def test_an_impossible_date_is_not_returned() -> None:
    assert _find_date("Ref 99/99/9999") is None


def test_currency_code_is_recognised() -> None:
    assert _find_currency("Total SAR 125.00") == "SAR"


def test_currency_symbol_maps_to_its_iso_code() -> None:
    assert _find_currency("Total $10.00") == "USD"


def test_no_currency_present_returns_none() -> None:
    assert _find_currency("Total 125.00") is None


def test_structure_reads_a_typical_invoice() -> None:
    text = "\n".join(
        [
            "Gulf Supplies LLC",
            "Invoice No: INV-778",
            "Issue Date: 2024-01-15",
            "Subtotal 200.00",
            "Total 230.00",
        ]
    )

    fields = _structure("invoice", text)

    assert fields["vendor_name"] == "Gulf Supplies LLC"
    assert fields["invoice_number"] == "INV-778"
    assert fields["issue_date"] == "2024-01-15"
    assert fields["total"] == "230.00"


def test_structure_keeps_the_full_transcript_for_generic_documents() -> None:
    text = "A letter.\nSecond line."

    assert _structure("document", text) == {"full_text": text}


# --- Confidence ----------------------------------------------------------


def test_confidence_is_weighted_by_how_much_text_a_block_carries() -> None:
    blocks = [
        block("a", 0, 0, 10, 10, confidence=0.10),
        block("a much longer line of text", 0, 20, 200, 40, confidence=1.00),
    ]

    weighted = _overall_confidence(blocks)

    assert weighted is not None
    assert weighted > 0.95  # an unweighted mean would say 0.55


def test_confidence_is_none_when_no_block_reported_one() -> None:
    assert _overall_confidence([block("text", 0, 0, 10, 10)]) is None
    assert _overall_confidence([]) is None


# --- Recogniser result shapes --------------------------------------------


def test_paddle_3x_dict_results_are_read() -> None:
    raw = [
        {
            "rec_texts": ["ACME", "10.00"],
            "rec_scores": [0.99, 0.95],
            "rec_polys": [
                [[10, 10], [80, 10], [80, 30], [10, 30]],
                [[300, 60], [360, 60], [360, 80], [300, 80]],
            ],
        }
    ]

    lines = list(_paddle_lines(raw))

    assert [line[0] for line in lines] == ["ACME", "10.00"]
    assert [line[1] for line in lines] == [0.99, 0.95]


def test_paddle_2x_nested_results_are_read() -> None:
    raw = [[[[[10, 10], [80, 10], [80, 30], [10, 30]], ("ACME", 0.99)]]]

    assert list(_paddle_lines(raw))[0][:2] == ("ACME", 0.99)


def test_paddle_results_from_an_unwrapped_page_are_read() -> None:
    """Older builds return the page's lines directly instead of a list of pages."""
    raw = [[[[10, 10], [80, 10], [80, 30], [10, 30]], ("ACME", 0.99)]]

    assert list(_paddle_lines(raw))[0][:2] == ("ACME", 0.99)


def test_paddle_none_pages_are_skipped() -> None:
    assert list(_paddle_lines([None])) == []


def test_tesseract_words_are_regrouped_into_lines() -> None:
    data = {
        "block_num": [1, 1, 1, 1],
        "par_num": [1, 1, 1, 1],
        "line_num": [1, 1, 2, 2],
        "word_num": [2, 1, 1, 2],
        "text": ["Store", "ACME", "TOTAL", "10.00"],
        "conf": [96, 98, 90, -1],
        "left": [60, 10, 10, 300],
        "top": [10, 10, 60, 60],
        "width": [50, 40, 60, 50],
        "height": [20, 20, 20, 20],
    }

    lines = list(_tesseract_lines(data))

    # Word order is Tesseract's own, not left-to-right: inside a line it is
    # already the logical order, which is what Arabic depends on.
    assert lines[0][0] == "ACME Store"
    assert lines[0][1] == pytest.approx(0.97)
    assert lines[0][2] == [10.0, 10.0, 110.0, 30.0]
    assert lines[1][0] == "TOTAL"  # conf -1 is a box with no text in it


def test_tesseract_blank_words_are_dropped() -> None:
    data = {
        "block_num": [1, 1],
        "par_num": [1, 1],
        "line_num": [1, 1],
        "word_num": [1, 2],
        "text": ["", "   "],
        "conf": [-1, -1],
        "left": [0, 0],
        "top": [0, 0],
        "width": [0, 0],
        "height": [0, 0],
    }

    assert list(_tesseract_lines(data)) == []


# --- Recognition, language and the confidence floor ----------------------


@pytest.fixture
def stub_settings(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Settings the module reads, replaced wholesale so a real .env cannot decide a test."""
    stub = SimpleNamespace(
        ocr_backend="paddle",
        ocr_min_confidence=0.60,
        languages=["ar", "en"],
        tesseract_cmd=None,
    )
    monkeypatch.setattr(traditional, "settings", stub)
    return stub


def test_blocks_below_the_confidence_floor_are_dropped(
    monkeypatch: pytest.MonkeyPatch, stub_settings: SimpleNamespace
) -> None:
    extractor = TraditionalExtractor(backend="paddle")
    monkeypatch.setattr(
        extractor,
        "_read_paddle",
        lambda image, number: iter(
            [
                block("solid", 0, 0, 40, 20, confidence=0.95, page=number),
                block("smudge", 0, 30, 40, 50, confidence=0.20, page=number),
                block("unscored", 0, 60, 40, 80, page=number),
            ]
        ),
    )

    kept = extractor._recognise([page()])

    # An unscored block is kept: no confidence is not low confidence.
    assert [b.text for b in kept] == ["solid", "unscored"]


def test_every_block_is_stamped_with_its_page(
    monkeypatch: pytest.MonkeyPatch, stub_settings: SimpleNamespace
) -> None:
    extractor = TraditionalExtractor(backend="paddle")
    monkeypatch.setattr(
        extractor,
        "_read_paddle",
        lambda image, number: iter([block(f"page {number}", 0, 0, 40, 20, page=number)]),
    )

    kept = extractor._recognise([page(), page(), page()])

    assert [b.page for b in kept] == [1, 2, 3]


def test_paddle_reads_arabic_when_arabic_is_configured(stub_settings: SimpleNamespace) -> None:
    assert traditional._paddle_lang() == "arabic"

    stub_settings.languages = ["en"]
    assert traditional._paddle_lang() == "en"


def test_tesseract_languages_are_mapped_to_traineddata_names(stub_settings: SimpleNamespace) -> None:
    assert traditional._tesseract_lang() == "ara+eng"

    stub_settings.languages = ["en", "chi_sim"]  # anything unmapped passes through
    assert traditional._tesseract_lang() == "eng+chi_sim"

    stub_settings.languages = []
    assert traditional._tesseract_lang() == "eng"
