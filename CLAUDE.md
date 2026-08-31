# OCR DMS

Extracts structured data from scanned business documents (receipts, invoices,
general documents) and stores it in Postgres. Two interchangeable extraction
backends: a traditional OCR pipeline and a vision LLM.

## Architecture rule

The engine is a **request parameter, not a branch in the code**. Both backends
implement the same `Extractor` interface in `app/engines/base.py` and return
the same `ExtractionResult`. Nothing outside `app/engines/` may check which
engine is in use. If you find yourself writing `if engine == "vlm"` anywhere
else, the abstraction is wrong — fix the interface instead.

## Layout

```
app/config.py       settings from environment, nothing else
app/schemas.py      Pydantic document types — single source of truth
app/preprocess.py   PDF triage, deskew, crop, lighting
app/llm.py          every model call goes through here
app/validate.py     arithmetic and format rules, review gate
app/db.py           Postgres persistence (optional at runtime)
app/pipeline.py     orchestration and compare mode
app/engines/        base.py, traditional.py, vlm.py, __init__.py registry
api/main.py         FastAPI
ui/streamlit_app.py Streamlit review UI
sql/schema.sql      documents and corrections tables
eval.py             field-level accuracy harness
tests/              pytest
```

## Status

Built:

- `app/config.py` — settings; every default is safe with no API key, no
  database and no OCR engine installed.
- `app/schemas.py` — `Receipt`, `Invoice`, `GenericDocument` and the pipeline
  types. Field descriptions are prompt text, not documentation.
- `app/preprocess.py` — both halves, loading and correction.
  - Loading and triage: `load_document`. PDFs through PyMuPDF (imported
    lazily), images through OpenCV, pages returned as BGR arrays rendered at
    `settings.pdf_dpi` and capped, plus text-layer triage: a page must average
    `MIN_CHARS_PER_PAGE` non-whitespace characters before its embedded text is
    trusted, so a PDF of blank lines or a lone stamp still goes to OCR.
  - Correction: `crop_to_document`, `deskew`, `normalise_lighting`,
    `preprocess_page`, `encode_jpeg`, `encode_png`. `preprocess_page` runs
    crop → deskew → resolution cap → lighting, each step defaulting to its
    `settings` flag and each free to decline: a correction applied to a page
    that did not need it is a net loss, so every function returns its input
    unchanged rather than guessing. Lighting is division by a heavy median
    blur, then CLAHE, then non-local means, and stops deliberately short of
    binarising — thresholding eats the thin joins and dots Arabic depends on.
- `app/llm.py` — one entry point for every model call.
  - `structured_text` and `structured_vision` take the document schema as an
    argument and return a parsed JSON object; Pydantic validation stays with
    the caller. Gemini goes through `google-genai`, OpenAI and DeepSeek through
    the `openai` SDK with `base_url` swapped. Both SDKs are imported lazily
    inside the call, temperature is 0 everywhere, and the OpenAI client is
    built with `max_retries=0` so tenacity is the only thing retrying.
    DeepSeek is marked as serving no vision model and `structured_vision`
    refuses it rather than letting the SDK return a confusing 400.
  - Retries are tenacity: attempts from `settings.llm_max_retries`,
    exponential 1–12s, reraise. A response that will not parse raises
    `LLMError`, which is in the retry set — that is what makes "a malformed
    response is a retry" true rather than aspirational. A missing API key is
    resolved *before* the retry wrapper and names its provider, because
    retrying a missing key only delays the message by three seconds.
  - `_gemini_schema` rewrites the schema on the way out. Gemini rejects most of
    what Pydantic emits and rejects it as a 400 at request time, so nothing
    local catches it: `$ref` is inlined against `$defs`, an `anyOf` of a type
    and null collapses to that type plus `nullable: true`, `const` becomes a
    one-member `enum`, and `title`, `default` and `additionalProperties` are
    dropped. The walk recurses because the fields most likely to break are the
    line items, and they live behind a `$ref` into `$defs`.
  - `parse_json_object` recovers the object from a code fence or from behind a
    sentence of preamble — the two things models add when told not to — and
    raises `LLMError` on anything else. OpenAI and DeepSeek need it more than
    Gemini does: their portable JSON mode constrains syntax but not shape, so
    those two get the schema as prose in the system message.
- `app/engines/base.py` — the contract both engines implement, and nothing that
  reads a page.
  - `Extractor` is the ABC: `key` (what the registry and the request parameter
    index on), `label` (what the UI shows), `gives_boxes` (whether
    `text_blocks` will carry coordinates, so the review UI can ask about the
    capability instead of about the engine), abstract
    `extract(pages, doc_type, embedded_text=None) -> ExtractionResult`, and the
    `_timed` context manager, which yields a reader rather than a number so the
    result can still be built in one expression at the end of the block.
    `__init_subclass__` rejects a concrete subclass that leaves `key` or
    `label` empty: the registry keys on `key`, and a missing one should fail at
    import, not at the first request.
  - `SYSTEM_PROMPT` carries the whole extraction policy — transcribe, never
    translate, never transliterate, Arabic into the Arabic field in Arabic
    script; amounts as printed, no currency conversion and no arithmetic, so a
    receipt whose lines disagree with its total returns both and lets
    `app.validate` find it; dates as YYYY-MM-DD with ambiguous all-numeric
    dates read day-month-year; unreadable means null, and a null is correct
    where a guess is not. Both engines send it and neither restates it: a rule
    added to one engine's prompt and not the other's is how two backends that
    are supposed to be interchangeable start disagreeing about the same page.
  - `build_prompt(doc_type, source_text=None)` embeds the schema rather than
    paraphrasing it — the field descriptions in `app/schemas.py` are the
    field-level instructions, so a paraphrase would be a second, staler copy of
    them. The two framings differ only in what they say the model is looking
    at: `source_text` frames the OCR as evidence and not ground truth, warns
    about misread characters and wrong reading order (interleaved columns, a
    total lifted out of its row, Arabic runs reversed or broken across lines),
    and permits repairing a clearly misread character inside an otherwise
    legible word while insisting that anything less certain is null. Without
    it, the prompt states that the page images follow. Schema last in both,
    closest to the answer.
  - `compact_schema` prunes the Pydantic schema to what instructs: `$ref`
    inlined (a model told to fill `items` cannot follow a pointer into a
    `$defs` it was never shown), `anyOf: [X, null]` collapsed to
    `"type": ["X", "null"]`, and `title`, `default`, `additionalProperties` and
    `$schema` dropped. Keys are reordered to type → const → description →
    properties → items so a field reads as what it is, then what to put in it.
    It is deliberately *not* `app.llm._gemini_schema`: that one targets
    Gemini's request-time dialect, which is OpenAPI-shaped and rejects a type
    array, while this one is read as prose by whichever model is answering.
    Sharing them would put one provider's spelling of "nullable" into the
    prompt every provider sees.
  - `coerce_fields(doc_type, raw)` validates and, on `ValidationError`, drops
    the values at the failing locations and revalidates, returning the document
    that survived plus one `Issue` per error — `severity="warning"`,
    `field` as the dotted path pydantic reported (`items.1.quantity`), so the
    warning points at the cell the reviewer has to look at. Pruning is safe to
    do blind because every extracted field is optional. A `loc` the walk cannot
    follow falls back to dropping the top-level key the path started at, and a
    pass that removes nothing ends the loop rather than spinning; only then
    does the document come back as `None`.
- `app/engines/traditional.py` — the OCR backend, in two deliberately separable
  stages: a recogniser reads pixels into `TextBlock`s, then a text model maps
  those onto the schema. The seam between them is a string plus boxes, and that
  is what buys the review UI coordinates and per-line confidences and what lets
  stage two be swapped for rules or LayoutLM later without a line of stage one
  changing.
  - `TraditionalExtractor(backend="paddle" | "tesseract")` sets `key` and
    `label` per instance — `traditional:paddle` and `traditional:tesseract` are
    two registry entries, because running one recogniser against the other is
    the comparison someone will want. The class-level `key` and `label` exist
    so the base's `__init_subclass__` check passes at import and so a bare
    `traditional` still names something. `gives_boxes = True`.
  - Both recognisers are imported inside the method that uses them, and a
    missing one is a `RuntimeError` naming the command that fixes it —
    `pip install paddleocr paddlepaddle`, or `pip install pytesseract` plus the
    binary. Paddle readers are cached per language behind a lock: construction
    loads three models from disk and costs seconds, and two web threads must
    not build the same reader and throw one away. `lang="arabic"` when `ar` is
    in `settings.languages`, since that model reads Latin digits too.
  - `_construct_paddle` walks a kwargs cascade because PaddleOCR 2.x wants
    `use_angle_cls` and `show_log` and 3.x rejects both outright. Only that
    rejection is caught; a constructor failing because the models will not
    download stays a failure. `_paddle_lines` normalises the three result
    shapes those generations return — 3.x's parallel `rec_*` lists, 2.x's
    `[polygon, (text, score)]` per page, and the older unwrapped page — so
    nothing above it branches on a Paddle version.
  - Tesseract's per-word rows are regrouped into lines, so both recognisers
    emit the same unit and `reading_order_text` and the UI never learn which
    one ran. Words are joined in Tesseract's own word order, not by position:
    inside a line that order is already the logical one, which is what an
    Arabic line needs. Confidences are normalised to 0–1 from Paddle's 0–1 and
    Tesseract's 0–100, `conf = -1` is a box with no text in it, and anything
    under `settings.ocr_min_confidence` is dropped and counted in a log line.
  - `reading_order_text(blocks, line_tolerance=12)` undoes detection order:
    blocks are grouped into lines by vertical proximity and sorted within a
    line by left edge. The line's centre is a running mean, so a column of
    near-misses does not extend one line down the whole page. Boxes are left in
    left-to-right order for Arabic too, and the comment saying why sits on the
    sort where someone will find it before "fixing" it: the recogniser already
    emits each box's characters in logical order, so reversing boxes by script
    would swap the label and the amount on every mixed row. A block with no box
    keeps its arrival order at the end of its page rather than being guessed
    into a row.
  - `extract` skips the recogniser entirely when `embedded_text` is present:
    that text is exact, and rendering it to pixels and reading them back can
    only lose characters. Overall confidence is the block confidences weighted
    by text length — an unweighted mean lets one misread character count as
    much as the line carrying the total — and it is `None` on the text-layer
    path, where no recogniser ran and 1.0 would be a claim about the extraction
    rather than about the transcription. Failure never leaves the engine: a
    backend that will not load, a page with no text and a model call that
    raises all come back as an `ExtractionResult` with `document=None`, keeping
    whatever OCR text was already paid for. An unknown `doc_type` is the one
    exception — that is a caller bug, and it raises before the clock starts.
- `app/engines/vlm.py` — the vision backend: one `structured_vision` call reads
  the pages and fills the schema, with no recogniser and no seam in the middle.
  - `VLMExtractor(provider="gemini" | "openai")` sets `key` and `label` per
    instance, so `vlm:gemini` and `vlm:openai` are two registry entries for the
    same reason the two recognisers are: asking both and comparing is the point
    of having both. `PROVIDERS` is the vision-capable subset of
    `app.llm.PROVIDERS` — DeepSeek serves no vision model, and `llm` refuses it
    by name anyway. `gives_boxes = False`.
  - What the path gives up is stated, not papered over: `text_blocks=[]` and
    `confidence=None`, both with the comment saying `app.validate` is what
    compensates. There is nothing in a vision answer that says where on the page
    a value was read, and a fabricated 0.95 is a claim the review gate would act
    on. Determinism goes too — temperature is 0 and the same page can still come
    back different.
  - `VISION_SYSTEM_PROMPT` is `SYSTEM_PROMPT` plus a page-condition addendum,
    and the split is the point: the base prompt is *policy*, which both engines
    must share, and the addendum is *condition* — skew, glare, creases, cropped
    edges, a hand in frame — which only the model looking at pixels can use. The
    traditional engine's model reads a transcription and is warned about that
    instead, by `build_prompt`'s OCR framing. Neither warning helps the other.
    The addendum's rule is that an unreadable character nulls the whole field:
    no rebuilding around the gap, no recovering a digit from its column, no
    finishing a word from its first half.
  - `_transcript` is one extra top-level key holding the model's own plain-text
    reading of every page. It is declared in the schema by
    `_schema_with_transcript`, not merely requested in prose, because Gemini is
    handed that schema as `response_schema` and answers with the keys it names
    and no others. `_pop_transcript` removes it before `coerce_fields` and it
    becomes `raw_text` — a few hundred tokens that give the reviewer something
    to check the fields against, and the one place in the answer where a partly
    legible line still belongs while the field it would have filled stays null.
  - `embedded_text` is ignored, with the reason on it: reading a PDF's text
    layer instead of its pixels would make this a text engine wearing a vision
    engine's key, and compare mode would then be running two text extractions
    against each other. Preferring the text layer is already the traditional
    engine's answer. Failure never leaves the engine either — no pages, an
    encode failure and a model call that raises all come back as an
    `ExtractionResult` with `document=None`; an unknown `doc_type` raises before
    the clock starts, as on the other engine.
- `app/engines/__init__.py` — the registry, and the only place an engine module
  is imported. Four keys, each mapped to a builder that does its import inside
  itself, so importing `app.engines` costs nothing and a machine without
  PaddleOCR can still list, choose and run the other three. `get_engine(key)`
  returns a fresh instance and raises `ValueError` listing the valid keys rather
  than falling back to a default — a request that named an engine and quietly
  got another produces a result nobody can account for later. A bare
  `traditional` or `vlm` resolves to whichever family member this deployment is
  configured for, which is what keeps `settings.default_engine="traditional"`
  meaning something; the lookup stays in the engine, which already reads
  `settings.ocr_backend` and the configured provider, rather than being repeated
  here. `available_engines()` returns `EngineInfo` (key, label, family,
  `gives_boxes`, note) for the UI, reading `label` and `gives_boxes` off the
  built engine so the list shown and the engine then run cannot drift apart. The
  `note` is the one thing that lives only here, and it is written for whoever
  has to pick: what the engine is good and bad at, never a restatement of its
  name.
- `app/validate.py` — the arithmetic gate, and the one layer that can catch a
  misread number. `validate(doc_type, fields) -> list[Issue]` takes the parsed
  document or the mapping behind it, because the pipeline holds a model and a
  correction coming back from the review UI is a dict; an unknown `doc_type`
  raises through `schema_for`, since a caller bug must not come back as a clean
  sheet.
  - The reason the module is arithmetic and not confidence is written at the
    top of it: a recogniser reporting 0.97 on a digit it read wrong is stating
    how sure it is that those pixels are the glyph it chose, which is a claim
    about ink and not about the number. A 3 read as an 8 comes back confident
    and every field stays individually plausible; what it cannot survive is the
    document checking itself.
  - Three arithmetic rules. Line totals sum to `subtotal` and
    `subtotal + charges − discount_total` equals `total`, both errors;
    `quantity × unit_price − discount` against a line's own total, a warning,
    because a line that rounds does not make the figure the document is filed
    for wrong. `charges` is `tax_amount` plus `service_charge`, `tip` and
    `shipping` — a tip printed on a receipt is part of what was paid, and
    leaving it out would fire the rule on every correct receipt that carries
    one. Tolerance is `settings.amount_tolerance`, absolute currency and not a
    ratio: per-item tax and rounding put a correct document a cent or two out,
    and no proportion makes that scale with the size of the bill.
  - Every rule declines rather than guesses. Missing inputs run no rule at all,
    because a null the extractor was right to return is not a failed
    reconciliation, and `_line_total_sum` returns `None` if any one line total
    is null: short by an unknown amount is not the same as disagreeing by a
    known one, and reporting it would point the reviewer at the subtotal
    instead of at the line `coerce_fields` already warned about.
  - Dates must be `YYYY-MM-DD`, parse as a real day — the shape check alone
    passes `2024-02-31` straight into a `DATE` column — and fall between 1990
    and today. `due_date` and `service_period_end` are exempt from the future
    half, since both are supposed to be ahead of today, and a day of slack
    covers a till in a timezone ahead of ours. A date that would not parse is
    not then also reported as an ordering fault. Currency is three uppercase
    letters, a warning: the amounts are still right, we just cannot name the
    unit. Required fields are the receipt's `merchant_name` and `total` and the
    invoice's `invoice_number`, `vendor_name` and `total`, each satisfied by
    its `_ar` spelling too.
  - `decide_status(issues, ocr_confidence, warnings) -> bool` is the gate:
    any issue of any severity, any pipeline warning, or a confidence below
    `settings.review_confidence_threshold` routes to review. Conservative on
    purpose, and the asymmetry is the argument: a wrong number reaching the
    database silently is found months later by someone reconciling accounts,
    with every downstream report already built on it; a reviewer glancing at a
    document that was fine costs seconds.
- `tests/test_preprocess.py` — both triage directions, the threshold boundary,
  the page cap, images and failure modes, on PDFs built in memory. Loading
  only; the correction half is not covered yet.
- `tests/test_llm.py` — 35 tests, no network: `settings` is replaced wholesale
  so a developer's real `.env` cannot change an outcome, and the provider call
  is stubbed at `_call_gemini`, the seam against the SDK. The converted schema
  is asserted against `google.genai.types.Schema` as well as for the absence of
  `$ref`, `$defs` and `anyOf`, because the string check alone would pass on a
  schema Gemini still rejects.
- `tests/test_traditional.py` — 34 tests, no recogniser imported and no
  network. Every reading-order test feeds its blocks scrambled, because
  pre-sorted input is passed by the identity function and proves nothing: a
  receipt fed bottom-up and right-to-left, twenty seeded shuffles of the same
  four blocks, an amount detected before its label, and a mixed Arabic/Latin
  row that pins the box-order decision. Also the tolerance boundary and the
  running mean, both `extract` paths and each of its four failure modes, the
  confidence weighting, the three Paddle result shapes and the Tesseract
  word-to-line regrouping. `settings` is replaced wholesale where it is read,
  so a developer's real `.env` cannot decide an outcome.
- `tests/test_validate.py` — 44 tests, no network and no database. The one the
  module exists for comes first: a receipt whose merchant, date, currency,
  three lines and subtotal are all individually plausible and whose total reads
  130.00 on a 115.00 document. One `total_mismatch` error naming `+15.00`, and
  `decide_status` sends it to review at confidence 0.99 — the case no score
  catches. The corrected receipt then returns no issues and auto-approves at
  the same confidence, and the same clean receipt at 0.42 still routes to
  review. Around those: the tolerance boundary at 115.02 and 115.03, each rule
  declining on missing inputs, the date exemptions, an Arabic-only merchant
  name satisfying the requirement, a zero total counting as present, amounts
  arriving as strings, and both non-document inputs. `settings` is replaced
  wholesale, so the tolerance and the threshold under test are the documented
  ones and not a developer's `.env`.

- `sql/schema.sql` — `documents`, `corrections`, and the two decisions in here
  that cannot be added later.
  - `documents` carries both `fields` and `raw_extraction`, and the split is
    the point. `fields` is what the business queries and what
    `approve_document` replaces, so it always holds the corrected truth.
    `raw_extraction` is the whole `ExtractionResult` the engine returned,
    written once on insert and updated by nothing — not an approval, not a
    correction, not a migration. The argument is in the SQL: this schema will
    change, and re-deriving the new shape from stored raw output is one
    `UPDATE` that costs nothing, while re-running inference over the archive
    costs the archive again and assumes the original uploads still exist. Raw
    output that was never written down cannot be recovered afterwards at any
    price, which is why it is not a "when we need it".
  - `corrections` is one row per field a reviewer changed — dotted path, old
    value, new value — and its reason is in the SQL too. It is the eval set and
    later the fine-tuning set, and it is produced for free by work somebody is
    doing anyway. It cannot be reconstructed after the fact: the instant
    `documents.fields` is overwritten, nothing anywhere records that a human
    looked at *that* field and disagreed. `raw_extraction` still holds the old
    answer, but it holds the old answer for every field, including the ones the
    reviewer read and accepted, and the difference between those two sets is
    the entire label.
  - `doc_type`, `engine` and `model` are copied onto each correction rather
    than joined from the document. Re-process a document with another engine
    and `documents.engine` names the backend behind the *current* fields, not
    the one whose answer this reviewer rejected; per-engine accuracy is the
    first question anyone asks of this table, and the join would answer it
    wrongly and silently.
  - The foreign key is `ON DELETE RESTRICT`. A correction is only interpretable
    beside the page it came from, so deleting a document really does destroy
    the pair — which is the argument for making that deletion a statement
    somebody has to write, rather than a side effect of tidying up `documents`.
    `CHECK (old_value IS DISTINCT FROM new_value)` refuses a row that records
    no change, because a correction that corrected nothing enters the eval set
    as a page the engine got wrong.
  - `old_value` and `new_value` are `jsonb` and not `text`, so `12.5` and
    `"12.50"` stay distinguishable: a training pair that cannot tell a number
    from its printed spelling is not much of a training pair.
  - Indexes. `(status, created_at DESC)` is the review queue and
    `(doc_type, created_at DESC)` the browse list, both carrying the order the
    rows are wanted in so neither listing sorts. `content_hash` is not unique —
    the same page legitimately arrives twice, re-scanned or re-filed, and
    refusing the insert would lose the second filing rather than flag it. GIN
    `jsonb_path_ops` on `fields` indexes one hash per path-and-value instead of
    an entry per key and per value, several times smaller and faster on `@>`,
    and gives up only the key-existence operators this table cannot use anyway:
    every extracted field is optional and therefore present-as-null, so "does
    this document have a merchant_name" is a question about the value. GIN
    `gin_trgm_ops` on `raw_text`, below. Two on `corrections`: `(document_id)`
    for the review UI and for the FK's own lookup on delete, and
    `(engine, field_path)` for the eval query.
  - Trigram and not tsvector, with the reason on the index. Postgres ships no
    Arabic text search configuration — no stemmer, no stop words, no lexeme
    normalisation — so `to_tsvector('simple', raw_text)` would index
    whitespace-delimited tokens verbatim, and Arabic writes the article, the
    conjunctions and the attached pronouns joined to the word: one noun becomes
    several different tokens and a search for one of them matches none of the
    others. Trigrams do not know what language they are looking at, which is
    the point, and they survive OCR — a word with one character misread keeps
    most of its trigrams, where token equality misses the row outright. What
    they buy is fuzzy substring matching and only that: no ranking, no phrase
    search, no field weighting. So that index is the line. If Arabic search
    ever has to be a feature rather than a convenience — ranked results,
    morphological matching, snippets — the answer is OpenSearch with an Arabic
    analyser beside Postgres, not a cleverer index here; Postgres cannot be
    taken there and pretending it can costs a migration to find out.
  - `comparison` and `agreement` are stored for the same reason
    `raw_extraction` is: two engines disagreeing on a page is the cheapest
    signal this system produces about which to trust, and it is produced once,
    at the moment both were run.
- `app/db.py` — Postgres, and the only place SQL is written. `enabled`,
  `get_pool`, `close_pool`, `init_schema`, `save_document`, `list_documents`,
  `get_document`, `approve_document`.
  - With `settings.database_url` empty every one of them returns its empty
    answer — None, `[]`, False — immediately, without importing a driver,
    opening a socket or raising. That is the hard constraint, and
    `tests/test_db.py` checks it by making any `import psycopg*` raise for the
    duration of a test and then calling every entry point that could want one,
    rather than by reading the code and agreeing with it.
  - An empty URL is the only thing that gets that treatment. A URL that *is*
    set and does not work — no driver installed, host unreachable, credentials
    wrong — raises, because a database somebody configured and expects to be
    written to is a different situation from one that was never configured, and
    answering both with a quiet None makes the first indistinguishable from the
    second until somebody goes looking for a month of documents that were never
    stored. A missing driver names the pip command, the way the missing
    recognisers in `traditional.py` name theirs.
  - psycopg is imported inside the functions that use it, so the app boots
    without it, and the pool is cached behind a lock for the reason the Paddle
    readers are: two web threads arriving together must not each open
    `db_pool_min` connections and throw one set away. Rows come back as dicts,
    because everything above this module is JSON-shaped.
  - `approve_document` is the one that matters. It flattens the stored and the
    corrected fields to dotted paths, diffs them, inserts one `corrections` row
    per changed value, and only then overwrites `documents.fields` — all in one
    transaction, with the row taken `FOR UPDATE`. Both properties are
    load-bearing. The UPDATE destroys the only thing that says which fields a
    human disagreed with, so a commit that landed it without the corrections
    rows would silently cost a training pair; and two reviewers on one document
    would otherwise both diff against the pre-review values, with the second
    UPDATE discarding the first one's edits. A reviewer who changed nothing
    still approves — that is a judgement about the document — and simply writes
    no rows.
  - `flatten_fields` spells its paths the way `app.validate` spells
    `Issue.field` — `total`, `items.2.unit_price` — so a correction and the
    rule that flagged it name the same cell and the review UI can put them side
    by side without translating between two path languages. An empty list or
    dict is a leaf: a reviewer who deletes every line leaves no `items.N.*`
    paths behind, and without `items: []` the diff would record only
    disappearances, which read the same as a document that never had lines. The
    document itself is the exception — an empty one has no fields, not one
    field named `""`.
  - `diff_fields` refuses to invent a change, and that is not fussiness. A row
    recording a difference nobody made enters the eval set as a page the engine
    got wrong and the fine-tuning set as the answer it should have given
    instead. So a missing key and an explicit null are one statement, which is
    what every optional field returns when the document does not show it; a
    blank box is that same statement typed into a form, and is *recorded* as
    null rather than as `""`, so the training pair does not ask the extractor
    for a value it is forbidden to produce; and `12` and `12.0` are one number
    that went through JSON. `True` and `1` are not, which is why the bool test
    comes before the numeric one. Ordering is numeric per path segment, so
    `items.2` sorts before `items.10` and a parent before its children.
  - `list_documents` returns everything except `raw_extraction`, `comparison`
    and `raw_text` — the three columns that make a row worth keeping are the
    three that would not fit down the wire a page at a time — and filters on
    `status` and `doc_type`, which are exactly the two composite indexes.
    `get_document` returns those columns plus that document's corrections,
    because it is the call the review UI makes when somebody opens a page and
    the transcription is what they check the fields against.
  - `init_schema` runs the file whole on every boot, every statement being
    `IF NOT EXISTS`, and then checks one thing at runtime — see the seventh
    decision below.
- `app/pipeline.py` — orchestration, and the first module that runs the whole
  thing. `process`, `compare`, `field_diff`, `agreement`, `render_pages`.
  - `process(data, filename, doc_type, engine_key, clean_images=True,
    persist=True)` is load → truncate → clean → engine → validate → gate →
    save, and it never raises. Every stage is caught and a failure comes back
    as `ProcessResult(status="failed", error=...)`, because the callers are a
    web handler and a Streamlit form and an exception in either is a stack
    trace where an answer should be. The stages are not handled equally, and
    the differences are the design: a file that will not load is fatal to the
    run because there is nothing to return; a save that fails is logged and
    swallowed because returning the extraction beats losing it; a page whose
    cleanup raises is read as it arrived, because a correction is an
    improvement to a page we can already read and failing over it would throw
    away one the engine would have managed. Both of the last two leave a note
    the reviewer sees.
  - Nothing in it asks which engine ran. `get_engine` turns the request's
    string into an `Extractor` and everything after that is the interface,
    which is what makes `compare` a thread pool rather than a mode.
  - The two kinds of finding are kept apart on the way in and joined on the way
    out. `validate` produces *issues* — the document disagreeing with itself.
    The pipeline produces *notes* — pages truncated, a page not cleaned — which
    are statements about the run and not faults found on the paper. Notes go to
    `decide_status` as its `warnings` argument, which is what that argument was
    left there for, and both lists land in `ProcessResult.issues`, which is
    what the reviewer reads and what `save_document` stores. A validation rule
    that raises becomes a `validation_failed` issue rather than a lost
    extraction: an unchecked document deserves review anyway.
  - Truncation stays `load_document`'s — it reads `settings.max_pages` and is
    the only place the cap is applied. What the pipeline adds is saying so out
    loud: an extraction off the first 20 pages of a 400-page file is not wrong
    so much as partial, and a reviewer who is not told cannot know the totals
    came off a page nobody read.
  - Cleaning runs only when `clean_images` *and* there is no trusted text
    layer. A PDF whose text we trust was rendered from vectors: no skew to
    remove, no page edge to find in a scene, no lighting gradient — and
    `crop_to_document` mistaking a printed border for the edge of the paper is
    a failure this repo has already had once.
  - `content_hash` is computed here, `sha256` over the uploaded bytes, because
    this is the last place the bytes exist. That closes the column
    `documents_content_hash_idx` was indexing nothing for.
  - `compare(data, filename, doc_type, engine_keys, ...)` loads and cleans
    once, then runs each engine in a `ThreadPoolExecutor` over the same arrays
    — one load is cheaper, and more importantly it is the only thing that makes
    the disagreement mean anything: two engines that were shown different
    pixels have not been compared. Threads because every engine is waiting on a
    network call; the pages are read and never written, so there is nothing to
    copy or lock. Results come back one per key in the order asked, including
    when the load failed and all N get the same failure, so a caller can zip
    them against what it requested. `persist` is not a parameter: a comparison
    is an experiment, and writing N rows for one document would put N answers
    in the table where the archive expects the one that was accepted. A
    repeated key is honoured rather than collapsed — the vision engines are not
    deterministic, so asking one twice is a real question.
  - `field_diff(results)` is one row per field, one column per engine, plus
    `agree`. Paths are `app.db.flatten_fields` paths, so a disagreement, the
    rule that flagged it and the correction row a reviewer eventually writes
    all name the same cell, and equality is `app.db._same`, so `12` and `12.0`
    are one number and two engines that both declined have not disagreed. The
    one rule that is *not* inherited: an engine that returned no document gets
    an empty column that disagrees on every row. Reading its absence as a page
    full of nulls would make agreement rise on exactly the runs where one
    engine fell over, since most fields on most documents are null.
    `agreement(rows)` is the fraction, computed outside `compare` because
    `compare` returns N results and the pairing is the caller's to choose.
  - `render_pages` returns PNG of exactly the pages an engine was given, same
    `clean_images` switch and same chain, so a reviewer looking at page 2 is
    looking at what the extraction was read from — crop and lighting included,
    which change what is legible. PNG and not the engine's JPEG: that one is a
    wire format sized down for a model. It returns `[]` rather than raising on
    a file that will not load, because a display helper has no error channel
    and `process` has already reported the same failure in
    `ProcessResult.error`.
- `tests/test_db.py` — 35 tests, no database and no driver. The flatten/diff
  pair carries most of them, because both of its failure modes cost something
  real and neither is visible: a change it misses is a label thrown away at the
  moment it was created, and a change it invents is a correct page entered into
  the eval set as a wrong one. So: the 130.00 receipt from
  `tests/test_validate.py` corrected to 115.00 producing exactly one
  `FieldChange`, a correction inside a line item arriving as
  `items.1.unit_price`, a line added and every line deleted, the
  same-value pairs that must produce nothing and the boolean that must not join
  them, a cleared box recorded as null, and `items.2` before `items.10`. Then
  the no-op contract for every entry point, one test of which makes any
  `import psycopg*` an `AssertionError` before calling them. `settings` is replaced
  wholesale, so a developer's real `.env` cannot decide an outcome.
- `tests/test_pipeline.py` — 33 tests, no network, no database, no recogniser.
  `app.llm.structured_vision` is stubbed with a fixed answer, which is the seam
  the vision engine reaches through, so what is under test is everything on
  either side of it: a real PNG through `load_document`, the correction chain,
  the engine, `app.validate`, the gate and the result the caller gets. The two
  the task named lead — a synthetic receipt arriving with its fields parsed and
  a status that is not `failed`, and a corrupt byte string coming back
  `status="failed"` without raising. Then the 130.00-on-a-115.00-document
  receipt run end to end for the first time, producing one `total_mismatch` and
  routing to review at no confidence at all; each failure path separately (an
  unknown engine key, an unknown `doc_type`, an empty upload, a raising engine,
  a raising `preprocess_page`, a raising `validate`); the difference between a
  failed *run* and an engine that returned no *document*, which keeps its
  `raw_text` and is reviewed rather than reported as failed; the no-database
  contract and the swallowed save failure; that the sha256 reaching
  `save_document` is the sha256 of the uploaded bytes; that `compare` loads
  once, never persists, and returns one result per key even when the load
  failed; and `field_diff`'s number equality, its numeric path sort and its
  empty column. `settings` is replaced wholesale in every module that reads it
  on this path, so a developer's real `.env` cannot decide an outcome.
- `api/main.py` — FastAPI, and deliberately a translator and a doorman rather
  than a layer with opinions. Seven routes: `GET /health`, `GET /v1/engines`,
  `POST /v1/documents/extract`, `POST /v1/documents/compare`,
  `GET /v1/documents`, `GET /v1/documents/{id}`,
  `POST /v1/documents/{id}/approve`.
  - Nothing in it asks which engine ran. `engine` arrives as a string and is
    passed to `pipeline.process` as the string it arrived as; the only thing
    the API does with it is ask `get_engine` whether the registry knows it, so
    the family aliases and the empty-means-default rule stay defined in
    `app/engines/__init__.py` and are not restated in an HTTP handler.
  - The status codes draw one line, and it is the line the pipeline already
    drew. `process` never raises and reports a document it could not read as
    `ProcessResult(status="failed", error=...)` — a value a caller can render —
    so that comes back **200** with `status="failed"` in the body, and a handler
    catches nothing. The codes are reserved for requests that were malformed
    before any document was read: empty upload 400, over the size cap 413,
    unknown `doc_type` 400 listing the valid ones, fewer than two engines on
    compare 400, unknown engine 400 carrying the registry's own message. A 5xx
    from this API therefore means the API itself broke, which is a different
    bug from a document that could not be read, and the distinction stays
    legible in a log of status codes.
  - Every refusal has the same shape — a stable `code`, a human `message`, and
    whatever list makes the message actionable (`valid`, `limit_mb`) — which is
    how `app.schemas.Issue` spells its findings and for the same reason: the
    string is for a person, the code is what anything else keys on.
  - Persistence stays optional here too. The three database-backed routes hang
    `Depends(_require_database)` and answer 503 naming `DATABASE_URL`;
    `/extract` and `/compare` never ask. Lifespan calls `db.init_schema()` and
    logs rather than crashes: a configured database that cannot be reached must
    not take down the two endpoints that never wanted one, and the endpoints
    that do want it fail their own request, loudly, when somebody asks.
  - `/health` does not open a connection. A health check that touches Postgres
    becomes a load generator the moment a balancer polls it, and would report a
    merely busy database as a dead API — so `persistence` there means
    *configured*, not reachable.
  - `POST /{id}/approve` with no `fields` means "approve what the engine
    returned", and it is resolved in the handler by reading the stored fields
    and handing them back. Passing `None` down would be the one way to lose a
    document through this API: `approve_document` flattens what it is given and
    diffs it against what is stored, so `None` flattens to `{}`, records every
    field on the page as deleted, and writes that into the eval set as the
    answer a human wanted. The extra read has a race the comment names, and the
    reviewer path — a UI submitting its own `fields` — does not go through it.
  - The size cap is `MAX_UPLOAD_MB = 25`, a module constant, and it is checked
    twice: `UploadFile.size` first, because refusing on the parser's count is
    free, then the length actually read, so a misdeclared part cannot get past
    it. An oversized body still reaches the process either way; the wall is
    `client_max_body_size` in front, and this is the backstop.
  - Extraction runs inline, on the request thread, and the upgrade path is
    commented on the handler because the point of it is that it stops at the
    handler: enqueue `(bytes, filename, doc_type, engine, clean_images,
    persist)` to Redis, return 202 with a job id and a `Location`, have a
    worker call `pipeline.process` with exactly those arguments and write the
    result where the status endpoint reads it. Nothing in `app/` changes —
    `process` is synchronous, takes bytes, never raises and returns one value,
    which is the shape a queue wants. A 20-page PaddleOCR scan otherwise holds
    a worker for the better part of a minute.
  - An empty `api/__init__.py` was added alongside it, which was outside the
    task's named file: `api.main` is imported as `api.main:app` and a namespace
    package would have worked by accident on this interpreter and not on the
    next one. It has no contents and is not expected to grow any.

`sql/schema.sql` and `app/db.py` were run against a real PostgreSQL 16.2 —
there is no Docker on this machine, so a user-mode cluster on port 55432 rather
than the container in the task — and all 41 checks pass. `init_schema` twice
for idempotency; `\d documents` and `\d corrections` rendered from the same
catalogs psql reads, since this build ships no `psql.exe`; then the receipt
`tests/test_validate.py` is built around, stored with its 130.00 misread and
approved down to 115.00, producing exactly one `corrections` row naming
`total`, `130.0`, `115.0` and `traditional:paddle`. Also pinned in that run:
`raw_extraction` byte-identical after two separate approvals, a blank box
writing no row, a second review arriving as `items.1.unit_price`, all three
`CHECK`s and the `RESTRICT` refusing what their comments say they refuse,
`EXPLAIN` choosing each of the six indexes by name, and Arabic matching through
the trigram index. That script lives outside the repo; the checks belong in
`tests/`, behind a skip when `DATABASE_URL` is unset.

Two decisions in the correction half came out of watching it fail, and should
not be quietly reverted:

- `crop_to_document` declines when the tone outside its candidate quad matches
  the tone inside. A photographed page has a scene around it; a scan has more
  of the same paper, so a four-gon surrounded by paper is a border *printed on*
  the document. Without that guard a flatbed scan of a bordered invoice was
  cropped to the border box, losing letterhead and footer.
- Rotation fills the new corners with the page's own paper colour, not with
  `BORDER_REPLICATE`. Replicating smears whatever sat on the edge into long
  diagonal streaks that the next skew estimate reads as ink: on a bordered page
  it left 5.9 degrees of apparent skew behind where a paper fill left 0.4.

A third, in `_gemini_schema`, is easy to break by tidying: the keyword filter
runs on the keys of a schema node, never on the keys of a `properties` map. `Receipt`
has a field called `items` and `GenericDocument` one called `title`, so a
filter applied one level too high silently deletes real fields from the schema
the model is asked to fill. `tests/test_llm.py` pins both. `_shrink` in
`app/engines/base.py` walks the same shape and carries the same trap; it has no
test yet.

A fourth, in `reading_order_text`, looks like a bug until you read the comment
on it: boxes are sorted left to right for Arabic as well. The recogniser emits
each box's characters in logical order already, so the only thing left to order
is the boxes, and reversing them by script breaks every mixed row — the Arabic
label and the Latin amount trade places, and a run of one script inside a line
of the other lands at the wrong end. `tests/test_traditional.py` pins it.

A fifth, in `app/engines/vlm.py`, is easy to tidy away: `_transcript` is added
to the *schema* sent to the provider, not only asked for in the prompt. Gemini
is handed that schema as `response_schema` and answers with the keys it names
and no others, so moving the request into prose alone would silently lose the
transcript on the Gemini path — and `raw_text` would come back `None` on the one
engine that has no `text_blocks` to fall back on. Untested, like the rest of
that module.

A sixth, in `app/validate.py`, reads as an omission until you see what
tightening it would do. `decide_status` treats `ocr_confidence=None` as *no
score*, not as a low one: the vision engine reports `None` by design and the
PDF text-layer path reports `None` because nothing was recognised there either,
so failing a missing score would send every one of those to review on the
strength of a number nobody claimed. `validate` declines the same way — a rule
missing an input reports nothing — and the future-date rule exempts `due_date`
and `service_period_end`, which are supposed to be ahead of today. The
conservatism is in the gate, where one real issue of any severity is enough;
spreading it into the rules would only produce findings the reviewer learns to
scroll past. `tests/test_validate.py` pins all four.

A seventh, in `sql/schema.sql`, is the one that would otherwise have shipped
broken and stayed broken. `pg_trgm` asks the *database's* `LC_CTYPE` what counts
as a letter, and under `LC_CTYPE=C` nothing outside ASCII is one, so every
Arabic character is discarded before the trigrams are cut. Measured on 16.2 on
one Arabic word: `show_trgm` returns 0 trigrams and `word_similarity` 0.0 under
C, and 7 trigrams and 1.0 under `en-US`. A C-locale database therefore creates
`documents_raw_text_trgm_idx`, keeps it up to date, and matches no Arabic in it
ever — no error, no empty-index warning, just a query returning nothing and a
corpus that looks as though it contains no Arabic. It cannot be repaired in
SQL, because `LC_CTYPE` is fixed at `CREATE DATABASE`, so `init_schema` reads
`datctype` after applying the file and logs a WARNING naming it. The
measurement is in the comment on that index; the collation does not matter and
the character classification is the whole thing, so do not "simplify" the
warning away on the grounds that the index obviously works.

The pipeline was run end to end on a real 3-page PDF with no mocking: triage
found the text layer (5943 characters), `traditional:paddle` took the
text-layer path and never touched a recogniser, `vlm:gemini` rendered and
encoded the same three pages to 928 KB of JPEG, and both then failed at the
model call and came back as `ExtractionResult(document=None)` — one
`no_document` issue, `needs_review=True`, `status="ok"`, no exception, and the
traditional engine still carrying the 5943 characters it had already paid for.
That is the never-raise contract and the keep-what-was-paid-for rule holding on
real bytes, and it is *not* evidence that either provider works.

**Neither provider has ever been called.** There is no API key on this machine
and no `.env`, and a key would not be enough: `app/config.py` declares no
`gemini_api_key`, `openai_api_key` or `llm_provider`, and `Settings` is
`extra="ignore"`, so a `GEMINI_API_KEY` line in `.env` is dropped before
`app.llm._setting` ever looks for it and every call ends at
`LLMError: no API key for provider 'gemini'`. That was measured, not inferred.
Until those three fields exist in config, no real extraction can happen through
any of the four engines, and every test in the repo is a test of the code
around a provider rather than of the provider. `paddleocr` and `pytesseract`
are not installed either, so the two traditional engines can currently only run
the PDF text-layer path.

`api/main.py` was run for real — uvicorn on 127.0.0.1:8077, curl against it,
no mocking anywhere — and the error paths were the point of the exercise, since
the happy path cannot reach a provider on this machine. Every refusal below is
a pasted response:

- empty upload → **400** `{"code":"empty_upload","message":"the uploaded file
  has no bytes in it"}`
- a 26.0 MB PDF → **413** `{"code":"file_too_large","message":"the upload is
  26.0 MB and the limit is 25 MB","limit_mb":25}`
- `doc_type=payslip` → **400** `{"code":"unknown_doc_type","message":"unknown
  doc_type 'payslip'","valid":["document","invoice","receipt"]}`
- `engines=traditional:paddle` on compare → **400** `{"code":"too_few_engines",
  "message":"compare needs at least two engines and got 1; ...","valid":[the
  four keys]}`
- `engine=trocr` → **400** `{"code":"unknown_engine"}` carrying
  `get_engine`'s own message, four keys and both family aliases
- all three database-backed routes with `DATABASE_URL` unset → **503**
  `{"code":"persistence_disabled","message":"this endpoint needs Postgres and
  DATABASE_URL is not set; extraction and comparison work without it"}`
- a byte string that is not a document → **200**, `status="failed"`,
  `error="ValueError: could not open 'junk.pdf' as a PDF: Failed to open
  stream"`, `needs_review=true`. That is the line the codes draw, holding on
  real bytes: a document that could not be read is an answer, not a 5xx.

Then the same 3-page PDF from the pipeline run above, through
`POST /v1/documents/extract` with `engine=traditional:paddle` and
`doc_type=document`: **200**, `status="ok"`, `page_count=3`, `raw_text` 6035
characters off the text layer, engine time 6.8 ms against 192 ms for the whole
request — and `document: null` with one `no_document` issue and
`needs_review=true`, because the model call ended at `LLMError: no API key for
provider 'gemini'` exactly as it does everywhere else. `stored=false` and
`document_id=null`, with no database configured and no complaint about it.
`/v1/documents/compare` across `traditional:paddle,vlm:gemini` returned both
results, `diff: []` and `agreement: null` — which is `field_diff`'s empty-column
rule being right rather than a bug: neither engine returned a document, so
there was nothing to disagree about.

So the API's rejections, its never-raise contract and its no-database contract
are proven on real requests; **an extraction that actually fills a document is
still not proven through any path**, and will not be until `app/config.py`
carries a provider key. The three database-backed routes are proven only at
their 503 — there is no Postgres on this machine any more, the 16.2 cluster
below having been a user-mode one that is gone.

- `ui/streamlit_app.py` — the review UI, and the first thing in the repo that
  talks to `api/main.py` over HTTP instead of importing `app/` directly.
  Sidebar: mode (Extract / Compare engines), document type, an engine picker —
  a radio with each engine's `note` as a caption in Extract mode, a multiselect
  in Compare mode — and toggles for image cleanup and saving; Compare mode
  hides the save toggle rather than disabling it, since `app.pipeline.compare`
  takes no `persist` argument at all. Extract results are a metrics row
  (engine, pages, seconds, OCR confidence or "n/a", status), issues grouped by
  severity, then four tabs: Review fields, Recognised text (the transcript
  plus `text_blocks` sorted lowest-confidence-first when the engine gives any),
  Pages, and Raw JSON. Compare mode is one column of metrics per engine, an
  agreement score, then a table of only the fields the engines disagree on
  with an expander for the rest.
  - The one hard requirement was that the review form is built from
    `schema_for(doc_type).model_fields`, never from the keys the response
    happened to contain: a field the extractor never mentions and a field it
    returned `null` for look identical once a JSON response has been reduced
    to a dict, and only the schema knows the field exists at all to draw a box
    for it. Widget type is read off each field's annotation rather than
    hand-mapped per name — `float in typing.get_args(...)` for an `Amount` /
    `Rate` / `Quantity` field, `int` for the one `int | None` field
    (`GenericDocument.page_count`), everything else a `text_input`
    (`text_area` for `full_text` and `summary`) — so a field added to
    `app/schemas.py` later gets a box without this file changing. `items` gets
    its own `st.data_editor`, columns and types read off `LineItem.model_fields`
    the same way, seeded with one blank row when the document has none so
    there is something to type into; a row left blank is dropped again on
    submit.
  - Pages are the one thing rendered locally rather than fetched: the API
    returns no page images, so `app.pipeline.render_pages` runs again against
    the uploaded bytes with the same `clean_images` switch, cached on
    `(bytes, filename, clean_images)` so a rerun triggered by editing an
    unrelated review field does not re-run the crop/deskew/lighting chain
    against a tab nobody is looking at.
  - `API_URL` is read with a bare `os.environ.get`, not through
    `app.config.settings`, and that is deliberate rather than a lapse in the
    "settings come from `app.config.settings`" convention: it is the address
    of a *different process* this file talks to over HTTP, and `Settings` is
    what the API reads to configure itself, not what this file would read to
    find the API. Putting it in `app/config.py` would hand a frontend concern
    to a module the backend boots with.
  - `_md_safe` exists because of a real bug, not a defensive habit: Streamlit's
    markdown renderer reads a colon immediately followed by a letter as the
    start of a directive (`:red[...]`, an emoji shortcode) and silently drops
    everything from the colon onward when nothing recognisable follows it.
    Every engine key in this system is `family:backend` — `traditional:paddle`,
    `vlm:gemini` — and rendering one through `st.subheader` came back as the
    bare family name with the backend gone and no error anywhere. The same
    parser reads `help=`, so `Receipt.time`'s own description, "Purchase time
    as HH:MM", loses `:MM` the same way. A colon followed by whitespace or a
    digit is never touched by it, which is why `st.error(f"...{API_URL}...")`
    with a `host:port` string read correctly in testing and is easy to
    mistake for the whole class of strings being safe. Escaping every colon
    before a markdown-rendering call is a no-op where the colon was never at
    risk and the fix everywhere it was, so do not "simplify" the escaping back
    out for the call sites that happened not to fail in testing.
  - What did **not** get built: a banner for `used_text_layer`. That field
    already does not exist on `ExtractionResult` — see the known gap below —
    the traditional engine knows whether it read the PDF's text layer and has
    nowhere to put the fact, so it only logs it, and that is still true here.
    `app/schemas.py` was outside this task's named files, so the fix stays
    where the earlier note put it: a field on `ExtractionResult`, filled by
    the engine that knows.
  - The task's proof steps named a field, `merchant_tax_id`, that does not
    exist on `Receipt`, `Invoice` or `GenericDocument` — the closest is
    `Receipt.tax_number`. The proof below used that field rather than adding
    the named one to `app/schemas.py`, which would have been a schema change
    outside this task's file.

  Proven for real, not by reading the code. No API key and no Postgres exist
  on this machine (see above), so `app.llm.structured_text` was stubbed with a
  fixed answer — `tax_number: null`, every other receipt field filled — the
  same technique `tests/test_pipeline.py` already uses, run this time through
  a real `uvicorn` process instead of a `TestClient`. A synthetic receipt PDF
  with a genuine embedded text layer (422 characters, well past
  `MIN_CHARS_PER_PAGE`) took the text-layer path, so no OCR backend was needed
  either. Through the running Streamlit app in a real browser: `tax_number`
  rendered as an empty, editable box beside the rest of the form, filled. A
  temporary embedded Postgres (`pgserver`, torn down afterward and not added
  to the repo) took `documents.status`; its bundled build carries no contrib
  extensions, so the schema was applied by hand with `pg_trgm` and its index
  stripped — a fresh instance of the same "index that silently does nothing"
  problem `sql/schema.sql` already documents for a C-locale database, not a
  new one. With a real `document_id`, Approve enabled; editing `tax_number`
  and `merchant_address` and clicking it produced exactly two `corrections`
  rows — `tax_number` from `null` to `300123456700003`, `merchant_address`
  from `"King Fahd Road, Jeddah"` to `"King Fahd Road, Jeddah -- Corniche
  Tower"` — and no others, which is `diff_fields` confirming that every
  untouched field round-tripped through its widget, `Amount`/`Rate` fields
  through `st.number_input`'s `float` included, without registering a false
  change. Compare mode was run the same way across `traditional:paddle` and
  `traditional:tesseract` — both take the text-layer path on this PDF, so
  neither needed a recogniser installed — and came back `agreement: 1.0`
  across all 34 fields, both engines having been handed the same stub.

- `eval.py` — the field-level accuracy harness, and the thing meant to turn
  "which engine is better" from an opinion into a number. `discover_samples`
  pairs every document in the folder with a same-stem `.json` of human-checked
  fields and skips, with a logged name, anything that has none; `evaluate`
  runs each requested engine over each pair through `app.pipeline.process`
  with `persist=False` — an eval run is not a record and must not write rows a
  reviewer would later have to account for — and scores every field with
  `app.db.flatten_fields`, the same dotted paths `app.validate` and the review
  UI already use, so a row in the per-document CSV names the same cell a
  correction or an `Issue` would.
  - `normalise(value)` is what keeps the score honest: money rounds to two
    places, text is case-folded with its whitespace collapsed, and a
    numeric-looking string coerces to a number before comparison, so "12.50"
    typed into a human's JSON and 12.5 out of a parsed `Amount` field agree,
    and "Total  Mart" agrees with "total mart". Skip this and every engine's
    score is dominated by formatting differences that say nothing about
    whether the number on the page was actually read correctly.
  - Accuracy is aggregated two ways. The headline number per engine is fields
    correct over fields compared, summed across the whole folder. The "five
    weakest fields" are aggregated by `_field_key`, which strips the digit
    segments out of a dotted path first — `items.2.unit_price` on one receipt
    and `items.7.unit_price` on another are the same *field*; the line index
    is an accident of that one document's layout, not something the next
    document shares, and comparing without stripping it would report a dozen
    near-empty line-index "fields" instead of the one that is actually weak.
  - The module docstring carries the field-accuracy-not-CER argument the task
    named: an engine can be 98% correct character by character and still get
    40% of totals wrong, because the characters it misreads cluster on digits
    rather than spreading evenly across the page, and CER would credit it for
    every letter it got right regardless.
  - `main` refuses to run on a folder that is empty or has no labelled
    documents rather than inventing ground truth or falling back to a
    synthetic corpus, and says so with a nonzero exit. Under about 100
    labelled documents it still runs, but says the run proves the harness
    rather than either engine.

  This repo has no `samples/` directory at all, which is the first thing
  running the harness against it found: `python eval.py samples --doc-type
  receipt` printed `samples does not exist` and exited 1 — exactly the
  refusal the task asked for. **`eval.py` has not been run against real
  documents, and this is not that proof.** What is proven, in a scratch
  location outside the repo and not added to it: a synthetic receipt image
  built the way `tests/test_pipeline.py` builds one, `app.llm.structured_vision`
  stubbed with a fixed answer the same way, and a hand-written `.json` label
  that matched every field except `total` (82.00 against the stubbed 82.80)
  and respelled `merchant_name` in different case and spacing. `evaluate` came
  back with `merchant_name` correct — `normalise` absorbing the respelling —
  and exactly one wrong field, `total`, which is also what `weakest_fields`
  named at 0% with n=1. That is the harness scoring what it was handed
  correctly; it is not a statement about `vlm:gemini` or any other engine,
  none of which has an API key to be judged with yet.

- `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `requirements-ocr.txt`
  and `README.md` — packaging, and the first thing in the repo anyone outside
  this machine can actually build. One image serves both roles: the
  Dockerfile's default `CMD` runs uvicorn, and `docker-compose.yml`'s `ui`
  service runs the same image with its `command` overridden to `streamlit
  run` instead, because `ui/streamlit_app.py` imports `app.pipeline` and
  `app.schemas` directly and needs the same dependencies the API does, not a
  lighter subset. `requirements-ocr.txt` holds `paddlepaddle` and
  `paddleocr` on their own, opt-in and roughly +1 GB, with the Dockerfile's
  install line for it commented out; the other three engines run without it.
  `docker-compose.yml` runs `postgres:16-alpine` with a named volume, `api`
  pointed at it by a `DATABASE_URL` set only in compose (`.env`'s own stays
  empty, so the same file also drives the no-Postgres, no-Docker quick
  start), and `ui` depending on `api` healthy with `DATABASE_URL` forced
  empty for the reason `api/main.py` never asks it for one either.

  **This machine has no Docker, and none was installed to test this.**
  Neither natively nor inside the one WSL distribution present (`Ubuntu`,
  which has no Docker in it either) — the same absence `sql/schema.sql`'s
  verification already worked around with a user-mode Postgres cluster.
  There is no equivalent workaround for a container engine, so `docker
  compose up --build` itself has never been run, and that is not glossed
  over here: the Dockerfile and compose file are unverified as a build,
  only as their parts.

  What was proven instead, from a real `git clone` of this repo (not the
  working tree — see the pin fix below, which is exactly the kind of thing
  that check exists to catch) into a scratch location outside it: every path
  the Dockerfile and `docker-compose.yml` reference exists in the clone;
  `docker-compose.yml` parses as the three services and one volume it is
  meant to; a clean virtualenv installing `requirements.txt` — the same `pip
  install -r requirements.txt` line the Dockerfile runs — resolves and
  completes; the full 197-test suite passes inside that venv; and, run for
  real with no mocking, `uvicorn api.main:app` (the Dockerfile's own `CMD`)
  answers `/health` and `/v1/engines` over real HTTP, and `streamlit run
  ui/streamlit_app.py` (`docker-compose.yml`'s `ui` command, `API_URL`
  pointed at the running API) serves its page with a real `200`. That is
  every part Docker would run, run directly; it is not proof that the
  three-service network, the healthcheck ordering, or the image build
  itself work, and nothing here claims otherwise.

  The clone step is why `requirements.txt` reads `opencv-python==4.11.0.86`
  and not `4.13.0.90`: `pip show` on this machine's existing environment
  reported both that version and `numpy==1.26.4` installed side by side,
  which is not the same claim as the two being a valid joint resolution. A
  clean-venv install of the first pin hit pip's `ResolutionImpossible` —
  `opencv-python 4.13.0.90 depends on numpy>=2` — immediately. `4.11.0.86`
  is the highest version pip resolves against `numpy==1.26.4`; the 197
  tests above ran against that pin, not the original one.

  Provider keys were not part of this: the "add keys" step in extracting a
  document through the built containers is still blocked by the gap further
  down this list — no field on `Settings` for a Gemini, OpenAI or DeepSeek
  key — and packaging does not touch `app/config.py`. A real extraction
  through the compose stack is therefore unproven for two independent
  reasons, not one, and fixing Docker's absence on this machine would still
  leave the other standing.

Known gaps:

- `app/config.py` is still Anthropic-shaped and `app/llm.py` is not. Config
  carries `anthropic_api_key` and defaults `llm_model` and `vision_model` to
  `claude-opus-5`; it has no `llm_provider`, `gemini_api_key`, `openai_api_key`
  or `deepseek_api_key`. Until it does, `app/llm.py` reads all of those through
  `getattr(settings, ...)` with a default — the same fallback shape
  `load_document` uses for `max_pages`, and for the same reason: config stays
  the only source, so adding the fields is a pure deletion of the fallbacks.
  Two consequences to clear at the same time: a configured model that does not
  belong to the selected provider is logged and replaced by that provider's
  default, which is a workaround for the Anthropic defaults and not a feature;
  and `settings.llm_enabled` still reports on the Anthropic key, so it is now
  wrong. `.env.example` needs the same fields. Config was outside this task's
  scope, so none of it was touched — and it is still the gap that blocks
  everything else: with no field to carry a key, `Settings`' `extra="ignore"`
  drops `GEMINI_API_KEY` out of `.env` silently and every engine fails at the
  model call. The API is now written and demonstrably answers, so this is the
  one thing standing between it and an extraction that fills a document. Fix it
  before `ui/`.
- The page cap has no home in settings. `load_document` takes `max_pages` as a
  keyword argument and otherwise reads `settings.max_pages` if it appears,
  falling back to a module constant. Add the field to `app/config.py` and drop
  the fallback.
- The repo carries no pytest configuration, so pytest resolves its rootdir from
  ancestor directories. On the current machine it finds `C:\Users\mooda\setup.cfg`
  and aborts before collection with a `UnicodeDecodeError`, so plain `pytest -q`
  does not run at all; `pytest -c <any empty ini> --rootdir=. tests` passes 197
  tests. A `pytest.ini` at the repo root fixes it.
- `coerce_fields` returns `(document, list[Issue])` and `ExtractionResult`
  still has nowhere to put them, so both engines still only log them. The
  pipeline is written now and this is the answer it did *not* get to give: it
  can only read what an engine returns, and a field pydantic dropped is not in
  there. `ExtractionResult` has to grow an `issues` list and both engines have
  to fill it; `app/pipeline.py` then concatenates it with `validate`'s the way
  it already concatenates its own notes, and `decide_status` sees it for free.
  It matters most on the vision path, where a dropped field is the only signal
  there is, and it is invisible today: a reviewer cannot see a log line.
- `ProcessResult` grew `status` and `error`, in `app/schemas.py`, which was
  outside this task's named files. The task required `process` to return
  `status="failed"` with the message in `error` and neither field existed;
  `ProcessResult` is the pipeline's own return type, `BaseModel` ignores extra
  keys so there was no way to carry them from `app/pipeline.py`, and CLAUDE.md
  had already deferred this file's shape to "when `app/pipeline.py` is
  written". Both are optional with defaults, so nothing else changed. The one
  thing to keep straight: `ProcessResult.status` is `ok | failed` and describes
  the *run*, while `documents.status` in SQL is `pending_review | approved |
  rejected` and is the review gate's verdict on a document that was read. A
  failed run has no such verdict, and is not stored.
- `app/db.py` has no test that touches Postgres. The 41 checks that prove the
  schema live in a script outside the repo, which means they will rot: nothing
  runs them, and the next change to `sql/schema.sql` or to `approve_document`
  passes `pytest` whatever it does to the database. They belong in
  `tests/test_db.py` behind a skip when `DATABASE_URL` is unset, which is also
  the shape the missing recogniser tests want.
- Duplicate detection is still not a feature. `app/pipeline.py` now computes
  the sha256 and `documents.content_hash` is populated, so
  `documents_content_hash_idx` has something to index — but nothing ever reads
  it. Somebody has to decide what a second filing of the same bytes should do,
  and the index deliberately does not answer: it is not unique, because the
  same page legitimately arrives twice and refusing the insert would lose the
  second filing rather than flag it.
- The `rejected` status has no producer. The `CHECK` admits it and
  `list_documents(status="rejected")` would return it, but nothing writes it: a
  reviewer can approve a document or leave it, and "this page is not what it
  says it is" has nowhere to go. `reject_document` is a few lines and belongs
  beside `approve_document`, with the same question about whether a rejection
  should write `corrections` rows — it probably should not, since a rejected
  page is not a page whose fields are wrong.
- `updated_at` is set by the statements that write it rather than by a trigger,
  so a hand-written `UPDATE` in psql will leave it stale. A `BEFORE UPDATE`
  trigger is five lines and makes the column mean what its name says.
- `init_schema` is create-if-absent, not a migration tool. It adds a table or
  an index that is not there; it will not alter a column, backfill one, or
  notice that the file has changed under a database that already ran it. The
  first schema change that is not purely additive needs Alembic or a hand-rolled
  version table, and the argument for `raw_extraction` is the argument for
  making that change cheap when it comes.
- Neither pool timeouts nor a statement timeout are configurable.
  `settings.db_pool_min` and `db_pool_max` are read; a query that hangs holds
  its connection until the server drops it, and `pool.connection()` waits on
  psycopg's default rather than on anything this app chose. Both belong in
  `app/config.py` beside them.
- `validate` reads the printed line totals against the printed subtotal, and
  `app/schemas.py` describes a line total as *after* its line discount and the
  subtotal as *before* discounts. A document-level `discount_total` is handled
  where it belongs, in the total rule, but a receipt that prints line totals
  before their own line discounts will disagree by the sum of them. It has not
  been seen on a real document yet; if it is, the fix is to add the line
  discounts back before comparing, not to widen the tolerance.
- The rules the task named are all that is there. No `tax_rate × subtotal`
  against `tax_amount`, no `amount_paid − total` against `change_due`, no
  `total − amount_paid` against `balance_due`, and currency is shape-checked
  rather than looked up, so `ZZZ` passes. Each is a few lines and an `Issue`
  code; none was in scope.
- `_EARLIEST` (1990) and `_FUTURE_SLACK` (one day) are module constants in
  `app/validate.py`, not settings, in the same way the page cap is a constant
  in `load_document`. Both belong in `app/config.py` next to
  `amount_tolerance`, which is already there and already read.
- OpenAI and DeepSeek see the schema twice: `build_prompt` embeds the compact
  one and `app.llm._json_instruction` appends the full Pydantic one to the
  system message. Gemini sees the compact one in the prompt and the sanitised
  one as `response_schema`, which is fine. Harmless but wasteful for the two
  `openai`-SDK providers; the fix is for `_json_instruction` to skip the schema
  when the prompt already carries it, which is a change in `app/llm.py` and was
  out of scope here.
- `app/engines/base.py` has no tests. What is worth pinning: that
  `compact_schema` keeps the `items` and `title` *fields* while dropping the
  `title` *keyword*, that `build_prompt` produces valid JSON inside `<schema>`
  after the array-collapsing pass in `_dumps`, that the OCR framing appears
  only with `source_text`, and each `coerce_fields` path — clean, one bad
  scalar, a bad line item, a bad `doc_type`, and a non-dict.
- `used_text_layer` has nowhere to live. The traditional engine knows whether
  it read the PDF's text layer or the pixels — the two produce results of very
  different quality and the review UI wants to say which — but
  `ExtractionResult` carries no such field, so for now the engine only logs it.
  `app/pipeline.py` cannot supply it either: it knows `loaded.has_text_layer`
  but not whether the engine chose to use it, and guessing on the engine's
  behalf is exactly the branch on engine identity the architecture rule
  forbids. So this is the same missing channel as the `coerce_fields` issues
  above and wants the same fix — a field on `ExtractionResult`, filled by the
  engine that knows.
- `ExtractionResult.model` comes back `None` from both engines.
  `app.llm.structured_text` and `structured_vision` return the parsed object and
  nothing about which provider or model answered, and neither engine will write
  a guess into an audit field — `settings.llm_model` is exactly the value
  `_resolve_model` is documented to override. The fix belongs in `app/llm.py`:
  report the resolved model alongside the answer. It is worse on the vision
  path, where the model *is* the extraction and the result names only its
  provider, through `engine`.
- `app/engines/vlm.py` has no tests. Everything worth pinning is reachable with
  `app.llm.structured_vision` stubbed and no network: that `_schema_with_transcript`
  adds `_transcript` while keeping the `items` and `title` fields, and that
  `app.llm._gemini_schema` still accepts the result; that `_pop_transcript`
  handles a string, a list of pages, a missing key and a non-dict answer, and
  that the key never reaches `coerce_fields`; that `text_blocks` and
  `confidence` stay empty and `None` on a *successful* extraction, not only on a
  failed one; that the provider named at construction is the one passed to
  `llm`; that the system prompt carries both the base policy and the vision
  addendum while the prompt carries no OCR framing; and each failure path — no
  pages, an encode failure, a raising model call, a non-dict answer, a bad
  `doc_type`. The registry wants its own: four keys and no more from
  `available_engines()`, `get_engine` on an unknown key and on each family
  alias, and that importing `app.engines` imports neither engine module.
- A long `_transcript` competes with the document for `settings.llm_max_tokens`
  (8192). On a dense multi-page invoice the answer can be truncated mid-object,
  which `parse_json_object` then rejects and tenacity retries — three times, at
  full cost, with the same result. Nothing caps or budgets it yet. Either the
  transcript gets a length instruction, or `llm_max_tokens` becomes a per-call
  argument the vision path raises.
- No test runs a real recogniser. `_read_paddle` and `_read_tesseract` are
  stubbed everywhere, so the SDK calls themselves are unverified against an
  installed engine, and `_paddle_lines` is pinned against result shapes written
  from the docs rather than captured from a run. The kwargs cascade in
  `_construct_paddle` is untested for the same reason. A test marked to skip
  when the backend is absent would cover it.
- `ProcessResult.comparison` and `agreement` have no producer, and the
  `documents.comparison` and `agreement` columns are therefore always null.
  `compare` returns a list of N results and `field_diff`/`agreement` reduce it,
  which is the right shape for a UI table but not the shape those two fields
  are: they hold *one* other result and one number, i.e. a pair. Nothing
  chooses the pair, and nothing should choose it automatically — two of five
  engines is an arbitrary selection. It is the review UI or the API that knows
  which comparison a user asked to keep, so this stays open until one of them
  exists, and the columns keep their argument in `sql/schema.sql` in the
  meantime.
- `app/pipeline.py` reaches into `app.db._same` and `app.db._path_key`. Both
  encode decisions `field_diff` must not answer differently — what counts as
  the same value, and that `items.2` sorts before `items.10` — and a second
  copy in the pipeline would drift from the one that writes the corrections
  rows, so the import is deliberate. They should simply be public on `app.db`;
  that is a rename and a line in `__all__`, and `app/db.py` was outside this
  task's files.
- Nothing measures how long the stages take relative to each other.
  `ProcessResult.duration_ms` is the whole run and `ExtractionResult.duration_ms`
  is the engine, so preprocessing is the difference between them and is
  reported nowhere. On a scanned multi-page PDF the correction chain is not
  free, and `clean_images=False` is a switch nobody can currently justify
  turning off with a number.
- `compare` runs every engine at once by default — `max_workers` defaults to
  the number of keys. That is right when the engines are waiting on four
  different providers and wrong when two of them are PaddleOCR, which holds
  models in memory per language. Nobody has run it with both traditional
  engines on a large page yet.
- The correction half has no unit tests. It was verified visually instead — a
  synthetic page shot at 6 degrees on a grey desk under a lighting gradient,
  1200x1500 in, 807x1105 out against an 800x1100 original, residual skew 0.0
  degrees, and a flat scan passing through at its own size. That script lives
  outside the repo; the checks belong in `tests/test_preprocess.py`.
- ~~The repo has no dependency file at all~~ **Resolved**: `requirements.txt`
  and `requirements-ocr.txt` exist now, pinned to what this repo was actually
  built and tested against — with one pin corrected by a clean-venv install
  rather than trusted from this machine's `pip show` (see the packaging entry
  above). Still open: nothing pins `pytest`, so the version the 197-test
  suite runs against (`7.4.4` here) is undeclared anywhere in the repo.
- The upload cap is `MAX_UPLOAD_MB = 25` in `api/main.py` and
  `settings.max_upload_mb` is 20, and the two are not connected. The task named
  25 and the limit a client is refused by has to be the documented one, so the
  constant won; it is the only number in the repo read from somewhere other
  than `app.config`. Raise the config field to 25 and delete the constant —
  and note this is the third thing waiting on a config edit, after the provider
  keys and the page cap.
- `api/main.py` has no tests. Everything worth pinning runs under
  `fastapi.testclient.TestClient` with no network and no database, which is
  exactly the shape `tests/` already requires: each of the five refusals and
  its `code`, the 503 on all three database routes with `settings.database_url`
  empty, the 200-with-`status="failed"` for bytes that will not load — the one
  that guards the line the whole module draws — the size cap on both the
  declared and the actual length, and that `/v1/engines` lists four keys and no
  more. `app.llm.structured_text` stubbed the way `tests/test_pipeline.py`
  stubs it would also pin the shape of a *successful* extract response, which
  is the one shape nothing anywhere has yet seen.
- `POST /v1/documents/{id}/approve` with no body reads the document, then
  `approve_document` reads it again under `FOR UPDATE`. A correction committed
  between the two would be approved away. The fix is a `corrected_fields`
  sentinel in `app/db.py` meaning "whatever is stored", resolved inside the
  transaction — which is `app/db.py`'s to make, not the API's, and is the same
  shape as the `_same`/`_path_key` privacy problem already listed above.
- Neither `/extract` nor `/compare` trims what it returns. `text_blocks` is
  every OCR line on every page with its box, and `compare` returns N of those
  at once — a five-page PaddleOCR comparison is a response measured in
  megabytes. A `?include=` or a `text_blocks=false` form field is a few lines;
  nobody has run it against a real recogniser to find out where the threshold
  actually bites.
- `ProcessResult.comparison` and `agreement` still have no producer, and
  `/v1/documents/compare` is now the endpoint that could have one — it holds N
  results and the caller knows which pair it cared about. It deliberately does
  not: nothing stores a comparison, and picking two of five engines on the
  user's behalf is the arbitrary selection `app/pipeline.py` declined to make.
  The endpoint that resolves it is a "keep this comparison" call the review UI
  makes, which is `ui/`'s to ask for.
- Nothing authenticates. Every route is open, `reviewed_by` is a string the
  client supplies and nothing checks, and the corrections table is therefore
  attributable only as far as the caller is honest. That is fine for a local
  deployment and is not fine for a shared one; whatever answers it has to
  answer for `ui/` too, so it belongs above both.
- `eval.py` has no test in `tests/` and no `samples/` folder to run against.
  Everything worth pinning is reachable with `app.llm.structured_vision`
  stubbed and no network, the same technique `tests/test_pipeline.py` already
  uses: `normalise` on money, case and numeric strings; `_field_key` collapsing
  line indices; `discover_samples` skipping an unlabelled document rather than
  inventing one; the empty-and-unlabelled-folder refusal and its exit code;
  and that a deliberately wrong ground-truth field is the one `weakest_fields`
  names. None of that is a claim about which engine is better — it still needs
  labelled documents in `samples/` for that, and none exist on this machine.
- `docker compose up --build` has never been run — there is no Docker on this
  machine, natively or in its one WSL distribution (see the packaging entry
  above for what was verified in its place). The untested surface is Docker
  itself: the image build, the `depends_on: condition: service_healthy`
  ordering across `postgres` → `api` → `ui`, and the named volume actually
  persisting `postgres`'s data across a restart. Whoever runs this first on a
  machine that has Docker should also add the provider-key fields to
  `app/config.py` first — without them the same `LLMError: no API key`
  every prior session hit is what a real extraction through the compose
  stack will hit too.

## Conventions

- Python 3.11+, `from __future__ import annotations` at the top of every module.
- Type hints on every function signature. `X | None`, not `Optional[X]`.
- Pydantic v2 (`model_validate`, `model_dump`, `model_json_schema`).
- Settings come from `app.config.settings`. Never call `os.getenv` elsewhere.
- Never `print`. Use `logging.getLogger(__name__)`.
- Docstrings explain *why*, not *what*. Skip the docstring if it would only
  restate the signature.

## Hard constraints

- Persistence is **optional**. With `DATABASE_URL` empty, the whole pipeline
  must still run; only storage is skipped. Never make a code path require a
  database.
- Heavy OCR dependencies (`paddleocr`, `pytesseract`) are imported **lazily,
  inside the function that uses them**, so the app boots without them.
- Documents may be Arabic, English, or mixed. Never translate — transcribe.
  Arabic values go into Arabic fields in Arabic script.
- Every model response is validated against a Pydantic schema. A malformed
  response is a retry, never an exception that reaches the user.
- No secrets in code or in committed files. `.env` is gitignored.

## Testing

```bash
pytest -q
```

Tests must not require API keys, a database, or network access. Mock the LLM
layer at `app.llm.structured_text` / `app.llm.structured_vision`.

## Scope discipline

Change only the files named in the current task. If a task seems to require
editing a file outside its scope, stop and say so instead of doing it.