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
- `tests/test_preprocess.py` — both triage directions, the threshold boundary,
  the page cap, images and failure modes, on PDFs built in memory. Loading
  only; the correction half is not covered yet.
- `tests/test_llm.py` — 35 tests, no network: `settings` is replaced wholesale
  so a developer's real `.env` cannot change an outcome, and the provider call
  is stubbed at `_call_gemini`, the seam against the SDK. The converted schema
  is asserted against `google.genai.types.Schema` as well as for the absence of
  `$ref`, `$defs` and `anyOf`, because the string check alone would pass on a
  schema Gemini still rejects.

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

Not written yet: `app/validate.py`, `app/db.py`, `app/pipeline.py`,
`app/engines/traditional.py`, `app/engines/vlm.py`, the `app/engines/__init__.py`
registry, `api/`, `ui/`, `sql/`, `eval.py`.

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
  scope, so none of it was touched.
- The page cap has no home in settings. `load_document` takes `max_pages` as a
  keyword argument and otherwise reads `settings.max_pages` if it appears,
  falling back to a module constant. Add the field to `app/config.py` and drop
  the fallback.
- The repo carries no pytest configuration, so pytest resolves its rootdir from
  ancestor directories. On the current machine it finds `C:\Users\mooda\setup.cfg`
  and aborts before collection with a `UnicodeDecodeError`, so plain `pytest -q`
  does not run at all; `pytest -c <any empty ini> --rootdir=. tests` passes 51
  tests. A `pytest.ini` at the repo root fixes it.
- `coerce_fields` returns `(document, list[Issue])` and `ExtractionResult` has
  nowhere to put the issues, so an engine can only log them. `ProcessResult`
  has the `issues` list they belong in, which means either `app/pipeline.py`
  collects them from the engine somehow or `ExtractionResult` grows a field.
  Decide it when `app/pipeline.py` is written, before both engines have
  invented their own answer. The `Issue` shape is already right for either.
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
- The correction half has no unit tests. It was verified visually instead — a
  synthetic page shot at 6 degrees on a grey desk under a lighting gradient,
  1200x1500 in, 807x1105 out against an 800x1100 original, residual skew 0.0
  degrees, and a flat scan passing through at its own size. That script lives
  outside the repo; the checks belong in `tests/test_preprocess.py`.

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