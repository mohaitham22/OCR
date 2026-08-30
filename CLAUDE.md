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
- `app/preprocess.py` — `load_document` only. PDFs through PyMuPDF (imported
  lazily), images through OpenCV, pages returned as BGR arrays rendered at
  `settings.pdf_dpi` and capped, plus text-layer triage: a page must average
  `MIN_CHARS_PER_PAGE` non-whitespace characters before its embedded text is
  trusted, so a PDF of blank lines or a lone stamp still goes to OCR.
- `tests/test_preprocess.py` — both triage directions, the threshold boundary,
  the page cap, images and failure modes, on PDFs built in memory.

Not written yet: deskew, auto-crop and lighting correction in
`app/preprocess.py`; `app/llm.py`, `app/validate.py`, `app/db.py`,
`app/pipeline.py`, `app/engines/`, `api/`, `ui/`, `sql/`, `eval.py`.

Known gaps:

- The page cap has no home in settings. `load_document` takes `max_pages` as a
  keyword argument and otherwise reads `settings.max_pages` if it appears,
  falling back to a module constant. Add the field to `app/config.py` and drop
  the fallback.
- The repo carries no pytest configuration, so pytest resolves its rootdir from
  ancestor directories and can pick up an unrelated config outside the project.
  A `pytest.ini` at the repo root fixes it.

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