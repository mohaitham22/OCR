"""Document and pipeline types.

The `description` on every extracted field is prompt text, not documentation:
`json_schema_for` hands these strings to the model at request time, so editing
one changes extraction behaviour. Write them as instructions to the model.

Every extracted scalar is optional. A field the document does not carry must
come back as null, never as a guess and never as a validation error.
"""

from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Discriminator, Field

# Arabic-Indic and Extended Arabic-Indic digits, plus the Arabic decimal and
# thousands separators. Applied to numeric fields only -- normalising digits in
# a text field would corrupt the transcription we promised not to alter.
_DIGIT_MAP = str.maketrans(
    {
        **{chr(0x0660 + i): str(i) for i in range(10)},
        **{chr(0x06F0 + i): str(i) for i in range(10)},
        "٫": ".",
        "٬": "",
    }
)
_NUMERIC_NOISE = re.compile(r"[^0-9.\-]")
_NULLISH = {"", "-", "--", "n/a", "na", "none", "null", "nil", "unknown", "not available"}


def _to_number(value: Any) -> Any:
    """Salvage the numbers OCR and models actually emit: "SAR 1,234.50", "(12.00)", Arabic digits."""
    if not isinstance(value, str):
        return value
    text = value.strip().translate(_DIGIT_MAP)
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = _NUMERIC_NOISE.sub("", text)
    if text.endswith("-"):
        text, negative = text[:-1], True
    if text in ("", "-", ".", "-."):
        return None
    return f"-{text}" if negative and not text.startswith("-") else text


def _to_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = " ".join(value.split())
    return None if text.lower() in _NULLISH else text


Amount = Annotated[float | None, BeforeValidator(_to_number)]
Rate = Annotated[float | None, BeforeValidator(_to_number)]
Quantity = Annotated[float | None, BeforeValidator(_to_number)]
Text = Annotated[str | None, BeforeValidator(_to_text)]

DocType = Literal["receipt", "invoice", "document"]


class _Extracted(BaseModel):
    """Base for the model-facing schemas: an unexpected key is dropped, not a retry."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class LineItem(_Extracted):
    description: Text = Field(
        default=None,
        description=(
            "Item name exactly as printed, in Latin script. Transcribe, never translate. "
            "If the line is printed in Arabic, leave this null and put the text in description_ar."
        ),
    )
    description_ar: Text = Field(
        default=None,
        description=(
            "Item name in Arabic script, exactly as printed. Transcribe, never translate "
            "and never transliterate. Null if the line is not printed in Arabic."
        ),
    )
    sku: Text = Field(
        default=None,
        description=(
            "Item code, SKU or barcode printed beside the line, copied character for character. "
            "Null if absent."
        ),
    )
    quantity: Quantity = Field(
        default=None,
        description=(
            "Number of units on this line as a plain number. Use 1 when the line shows no "
            "quantity. Use a decimal for weighed goods, e.g. 0.75 for 750 g."
        ),
    )
    unit: Text = Field(
        default=None,
        description="Unit of measure as printed, e.g. kg, g, L, pc, hr. Null if the line shows no unit.",
    )
    unit_price: Amount = Field(
        default=None,
        description=(
            "Price of one unit before any line discount, as a plain number with no currency "
            "symbol and no thousands separator, e.g. 12.50."
        ),
    )
    discount: Amount = Field(
        default=None,
        description=(
            "Amount subtracted from this line, as a positive number. Null if the line shows no "
            "discount."
        ),
    )
    tax_rate: Rate = Field(
        default=None,
        description=(
            "Tax percentage applied to this line as a number, e.g. 15 for 15%. This is the rate, "
            "not the tax amount."
        ),
    )
    tax_amount: Amount = Field(
        default=None,
        description="Tax charged on this line. Null if tax appears only as a document-level total.",
    )
    total: Amount = Field(
        default=None,
        description=(
            "Amount for this line after its discount, exactly as printed. Read it, do not "
            "calculate it: if no line total is printed, return null."
        ),
    )


class Receipt(_Extracted):
    doc_type: Literal["receipt"] = Field(default="receipt", description="Always 'receipt'.")

    merchant_name: Text = Field(
        default=None,
        description=(
            "Shop or business name as printed, in Latin script. Transcribe, never translate. "
            "Null if the name appears only in Arabic -- put that in merchant_name_ar."
        ),
    )
    merchant_name_ar: Text = Field(
        default=None,
        description=(
            "Shop or business name in Arabic script, exactly as printed. Transcribe, never "
            "translate and never transliterate. Null if the receipt shows no Arabic name."
        ),
    )
    branch: Text = Field(
        default=None,
        description="Branch or store name or number as printed. Null if the receipt names no branch.",
    )
    merchant_address: Text = Field(
        default=None,
        description=(
            "Street address on the receipt, as one line, in the script it is printed in. Do not "
            "translate it."
        ),
    )
    merchant_phone: Text = Field(
        default=None,
        description="Merchant phone number as printed, keeping only digits and a leading '+'.",
    )
    tax_number: Text = Field(
        default=None,
        description=(
            "Merchant VAT or tax registration number, digits only. Look for labels such as "
            "VAT No, TRN, CR, or the Arabic label for tax number."
        ),
    )

    receipt_number: Text = Field(
        default=None,
        description=(
            "Receipt, bill or transaction number as printed, copied exactly including any prefix "
            "and leading zeros."
        ),
    )
    date: Text = Field(
        default=None,
        description=(
            "Purchase date as YYYY-MM-DD. Convert from whatever format is printed, using the "
            "merchant's locale to resolve day/month order. Null if no date is printed -- never "
            "guess one."
        ),
    )
    time: Text = Field(
        default=None,
        description="Purchase time as HH:MM on a 24-hour clock, e.g. 19:05. Null if no time is printed.",
    )
    currency: Text = Field(
        default=None,
        description=(
            "ISO 4217 code for the amounts on this receipt, e.g. SAR, AED, EGP, USD. Infer it "
            "from the printed symbol or the merchant's country. Null if you cannot tell."
        ),
    )

    subtotal: Amount = Field(
        default=None,
        description=(
            "Total before tax and before discounts, exactly as printed. Read it, do not "
            "calculate it; null if no subtotal line exists."
        ),
    )
    discount_total: Amount = Field(
        default=None,
        description="Total discount on the receipt as a positive number, as printed. Null if none.",
    )
    tax_rate: Rate = Field(
        default=None,
        description="Tax percentage as a number, e.g. 15 for 15%. This is the rate, not the tax amount.",
    )
    tax_amount: Amount = Field(
        default=None,
        description="Total tax charged, as printed. Null if the receipt shows no tax.",
    )
    service_charge: Amount = Field(
        default=None,
        description="Service charge as printed. Null if none.",
    )
    tip: Amount = Field(default=None, description="Gratuity as printed. Null if none.")
    total: Amount = Field(
        default=None,
        description=(
            "Final amount due after tax and discounts, exactly as printed -- the figure the "
            "customer pays. Read it, do not calculate it."
        ),
    )
    amount_paid: Amount = Field(
        default=None,
        description="Cash tendered or amount charged to the card, as printed. Null if not shown.",
    )
    change_due: Amount = Field(
        default=None,
        description="Change returned to the customer, as printed. Null if not shown.",
    )

    payment_method: Text = Field(
        default=None,
        description="How it was paid, lower case, one of: cash, card, mada, transfer, voucher, other.",
    )
    card_last4: Text = Field(
        default=None,
        description=(
            "Last four digits of the card, digits only. Null if the number is fully masked or "
            "absent."
        ),
    )
    cashier: Text = Field(
        default=None,
        description="Cashier or server name or ID as printed, in its own script.",
    )
    terminal_id: Text = Field(default=None, description="POS terminal, register or till ID as printed.")

    items: list[LineItem] = Field(
        default_factory=list,
        description=(
            "Every purchased line, in printed order. Do not merge, split, reorder or invent "
            "lines, and do not put subtotal, tax or total rows here. Empty list if no line is "
            "legible."
        ),
    )


class Invoice(_Extracted):
    doc_type: Literal["invoice"] = Field(default="invoice", description="Always 'invoice'.")

    invoice_number: Text = Field(
        default=None,
        description=(
            "Invoice number as printed, copied exactly including any prefix and leading zeros."
        ),
    )
    purchase_order_number: Text = Field(
        default=None,
        description=(
            "Customer purchase order or contract reference printed on the invoice. Null if absent."
        ),
    )
    issue_date: Text = Field(
        default=None,
        description=(
            "Date the invoice was issued, as YYYY-MM-DD. Convert from the printed format; never "
            "guess a date."
        ),
    )
    due_date: Text = Field(
        default=None,
        description=(
            "Date payment is due, as YYYY-MM-DD. Only if a due date is printed -- do not derive "
            "it from the payment terms."
        ),
    )
    service_period_start: Text = Field(
        default=None,
        description="First day of the period this invoice covers, as YYYY-MM-DD. Null if it covers no period.",
    )
    service_period_end: Text = Field(
        default=None,
        description="Last day of the period this invoice covers, as YYYY-MM-DD. Null if it covers no period.",
    )

    vendor_name: Text = Field(
        default=None,
        description=(
            "Name of the party issuing the invoice, in Latin script, as printed. Transcribe, "
            "never translate. Null if it appears only in Arabic -- put that in vendor_name_ar."
        ),
    )
    vendor_name_ar: Text = Field(
        default=None,
        description=(
            "Issuing party's name in Arabic script, exactly as printed. Never translate or "
            "transliterate."
        ),
    )
    vendor_address: Text = Field(
        default=None,
        description="Issuing party's address as one line, in its printed script.",
    )
    vendor_tax_number: Text = Field(
        default=None,
        description="Issuing party's VAT or tax registration number, digits only.",
    )
    vendor_phone: Text = Field(
        default=None,
        description="Issuing party's phone number, digits and a leading '+' only.",
    )
    vendor_email: Text = Field(
        default=None,
        description="Issuing party's email address, lower case, exactly as printed.",
    )

    customer_name: Text = Field(
        default=None,
        description=(
            "Name of the party being billed, in Latin script, as printed -- the 'Bill To' party, "
            "not the issuer. Null if it appears only in Arabic."
        ),
    )
    customer_name_ar: Text = Field(
        default=None,
        description=(
            "Billed party's name in Arabic script, exactly as printed. Never translate or "
            "transliterate."
        ),
    )
    customer_address: Text = Field(
        default=None,
        description="Billed party's address as one line, in its printed script.",
    )
    customer_tax_number: Text = Field(
        default=None,
        description="Billed party's VAT or tax registration number, digits only.",
    )

    currency: Text = Field(
        default=None,
        description=(
            "ISO 4217 code for the amounts on this invoice, e.g. SAR, AED, EGP, USD. Null if you "
            "cannot tell."
        ),
    )
    subtotal: Amount = Field(
        default=None,
        description="Total before tax and discounts, exactly as printed. Read it, do not calculate it.",
    )
    discount_total: Amount = Field(
        default=None,
        description="Total discount as a positive number, as printed. Null if none.",
    )
    tax_rate: Rate = Field(
        default=None,
        description="Tax percentage as a number, e.g. 15 for 15%. This is the rate, not the amount.",
    )
    tax_amount: Amount = Field(
        default=None,
        description="Total tax charged, as printed. Null if the invoice shows no tax.",
    )
    shipping: Amount = Field(
        default=None,
        description="Shipping, freight or delivery charge, as printed. Null if none.",
    )
    total: Amount = Field(
        default=None,
        description=(
            "Full invoice amount after tax and discounts, exactly as printed. Read it, do not "
            "calculate it."
        ),
    )
    amount_paid: Amount = Field(
        default=None,
        description="Amount already paid or prepaid, as printed. Null if not shown.",
    )
    balance_due: Amount = Field(
        default=None,
        description="Amount still owed, as printed. Null if the invoice states no separate balance.",
    )

    payment_terms: Text = Field(
        default=None,
        description=(
            "Payment terms exactly as printed, e.g. 'Net 30'. Transcribe, never translate or "
            "normalise."
        ),
    )
    payment_method: Text = Field(
        default=None,
        description="Requested or recorded payment method, lower case, e.g. transfer, cash, card, cheque.",
    )
    bank_name: Text = Field(default=None, description="Beneficiary bank name as printed, in its own script.")
    iban: Text = Field(default=None, description="IBAN or account number as printed, upper case, spaces removed.")

    items: list[LineItem] = Field(
        default_factory=list,
        description=(
            "Every billed line, in printed order. Do not merge, split, reorder or invent lines, "
            "and do not put subtotal, tax or total rows here."
        ),
    )


class GenericDocument(_Extracted):
    doc_type: Literal["document"] = Field(default="document", description="Always 'document'.")

    title: Text = Field(
        default=None,
        description=(
            "Document title in Latin script, as printed. Transcribe, never translate. Null if "
            "the title appears only in Arabic -- put that in title_ar."
        ),
    )
    title_ar: Text = Field(
        default=None,
        description=(
            "Document title in Arabic script, exactly as printed. Never translate or transliterate."
        ),
    )
    document_number: Text = Field(
        default=None,
        description=(
            "Reference, file or serial number printed on the document, copied exactly. Null if "
            "absent."
        ),
    )
    date: Text = Field(
        default=None,
        description=(
            "Date printed on the document, as YYYY-MM-DD. Null if no date is printed -- never "
            "guess one."
        ),
    )
    issuer: Text = Field(
        default=None,
        description=(
            "Organisation or person the document is from, in Latin script, as printed. Null if "
            "it appears only in Arabic."
        ),
    )
    issuer_ar: Text = Field(
        default=None,
        description=(
            "Issuing organisation or person in Arabic script, exactly as printed. Never translate."
        ),
    )
    recipient: Text = Field(
        default=None,
        description="Party the document is addressed to, as printed, in its own script.",
    )
    subject: Text = Field(
        default=None,
        description=(
            "Subject or 'Re:' line as printed, in its own script. Do not compose one if the "
            "document has none."
        ),
    )
    language: Text = Field(
        default=None,
        description=(
            "Main language of the document: 'ar', 'en', or 'mixed' when both scripts carry content."
        ),
    )
    page_count: int | None = Field(default=None, description="Number of pages you were shown.")
    summary: Text = Field(
        default=None,
        description=(
            "Two or three sentences on what the document says, written in the document's own "
            "language. This is the only field you may compose rather than transcribe; do not "
            "translate the document in order to write it."
        ),
    )
    full_text: str | None = Field(
        default=None,
        description=(
            "Complete text of the document, transcribed line by line in reading order with line "
            "breaks preserved. Never translate, summarise, reorder or correct spelling."
        ),
    )


Document = Annotated[Receipt | Invoice | GenericDocument, Discriminator("doc_type")]

DOC_TYPES: dict[str, type[BaseModel]] = {
    "receipt": Receipt,
    "invoice": Invoice,
    "document": GenericDocument,
}


def schema_for(doc_type: str) -> type[BaseModel]:
    key = doc_type.strip().lower() if isinstance(doc_type, str) else doc_type
    if key not in DOC_TYPES:
        raise ValueError(f"unknown doc_type {doc_type!r}; expected one of {sorted(DOC_TYPES)}")
    return DOC_TYPES[key]


def json_schema_for(doc_type: str) -> dict[str, Any]:
    """The schema the model is asked to fill, field descriptions included."""
    return schema_for(doc_type).model_json_schema()


# --- Pipeline types ------------------------------------------------------
# These are built by our own code rather than by a model, so their descriptions
# are plain documentation and the handful of fields we always set stay required.


class TextBlock(BaseModel):
    text: str = Field(description="Transcribed text of one OCR block, unaltered.")
    confidence: float | None = Field(default=None, description="Backend confidence for this block, 0.0-1.0.")
    bbox: list[float] | None = Field(
        default=None,
        description="Block box as [x0, y0, x1, y1] in pixels on the preprocessed page.",
    )
    page: int | None = Field(default=None, description="1-based page the block was read from.")
    language: str | None = Field(
        default=None,
        description="Script detected for this block: 'ar', 'en' or 'mixed'.",
    )


class ExtractionResult(BaseModel):
    """What every engine returns, whichever backend produced it."""

    doc_type: str | None = Field(
        default=None,
        description="Key in DOC_TYPES that `document` was parsed against.",
    )
    document: Document | None = Field(
        default=None,
        description="Parsed document, or None if extraction failed.",
    )
    engine: str | None = Field(
        default=None,
        description=(
            "Backend that produced this result. Audit and compare-mode metadata only; nothing "
            "outside app/engines/ may branch on it."
        ),
    )
    model: str | None = Field(default=None, description="Model id used, when a model was involved.")
    text_blocks: list[TextBlock] = Field(
        default_factory=list,
        description="OCR blocks behind the extraction, when the backend produced any.",
    )
    raw_text: str | None = Field(default=None, description="Full transcribed page text, before structuring.")
    confidence: float | None = Field(default=None, description="Overall confidence in this extraction, 0.0-1.0.")
    page_count: int | None = Field(default=None, description="Pages processed.")
    duration_ms: float | None = Field(default=None, description="Wall-clock time spent inside the engine.")


class Issue(BaseModel):
    code: str = Field(description="Stable machine-readable id, e.g. 'total_mismatch'.")
    message: str = Field(description="Human-readable explanation shown in the review UI.")
    severity: Literal["error", "warning", "info"] = Field(
        description="'error' forces review, 'warning' flags it, 'info' is advisory."
    )
    field: str | None = Field(
        default=None,
        description="Dotted path to the offending field, e.g. 'items.2.total'.",
    )
    expected: str | None = Field(default=None, description="Value the rule expected, stringified.")
    actual: str | None = Field(default=None, description="Value found, stringified.")


class ProcessResult(BaseModel):
    """One document through the pipeline, including the review gate and storage outcome."""

    status: Literal["ok", "failed"] = Field(
        default="ok",
        description=(
            "Whether the run itself completed. 'failed' means no extraction was produced and "
            "`error` says why. This is not documents.status in SQL, which is the review gate's "
            "verdict on a document that *was* read: a failed run has no such verdict to give."
        ),
    )
    error: str | None = Field(
        default=None,
        description="Why the run failed, when it did. None on every completed run.",
    )
    document_id: str | None = Field(default=None, description="Database id, None when persistence is off.")
    source: str | None = Field(default=None, description="Original filename or upload reference.")
    doc_type: str | None = Field(default=None, description="Key in DOC_TYPES used for this document.")
    extraction: ExtractionResult | None = Field(default=None, description="Primary result.")
    comparison: ExtractionResult | None = Field(
        default=None,
        description="Second result in compare mode, otherwise None.",
    )
    agreement: float | None = Field(
        default=None,
        description="Fraction of fields on which the two compare-mode results match, 0.0-1.0.",
    )
    issues: list[Issue] = Field(default_factory=list, description="Everything app.validate flagged.")
    needs_review: bool = Field(
        default=False,
        description="Review gate: True when a human must confirm before the data is trusted.",
    )
    stored: bool = Field(default=False, description="True only if the row reached Postgres.")
    duration_ms: float | None = Field(default=None, description="Wall-clock time for the whole pipeline run.")
