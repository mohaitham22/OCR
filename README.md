# OCR DMS

Extracts structured data from scanned business documents (receipts, invoices,
general documents) and stores it in Postgres. Two interchangeable extraction
backends -- a traditional OCR pipeline and a vision LLM -- behind one API and
one review UI. See [`CLAUDE.md`](CLAUDE.md) for the full architecture and
build history; this file is the operator's view.

## Quick start

```bash
cp .env.example .env
# edit .env: at minimum set a provider key -- see "Provider keys" below
docker compose up --build
```

- API: http://localhost:8000 (docs at `/docs`, health at `/health`)
- Review UI: http://localhost:8501

`docker compose` starts three services: `postgres` (with a named volume, so
data survives a restart), `api`, and `ui`. `ui` depends on `api` being
healthy and `api` depends on `postgres` being healthy, so a plain `up`
brings them up in the right order. `api`'s `DATABASE_URL` is pointed at the
`postgres` service inside the compose network; `.env`'s own `DATABASE_URL`
stays empty so the same file also works for running the two processes
directly, without Postgres or Docker at all:

```bash
pip install -r requirements.txt   # add requirements-ocr.txt too for OCR_BACKEND=paddle
uvicorn api.main:app --reload
streamlit run ui/streamlit_app.py   # in a second shell
```

### Provider keys

Every engine makes at least one model call, so extraction needs a working
key regardless of which engine is chosen. **As of this build, `app/config.py`
has no field for a Gemini, OpenAI or DeepSeek key** -- see CLAUDE.md's Known
gaps -- so setting `GEMINI_API_KEY` or `OPENAI_API_KEY` in `.env` today has no
effect: `Settings` is `extra="ignore"` and silently drops it, and every call
ends at `LLMError: no API key for provider '...'`. This is a pre-existing gap
that packaging does not fix. Confirm it is closed (grep `app/config.py` for
`gemini_api_key` / `openai_api_key` / `deepseek_api_key` / `llm_provider`)
before expecting a real extraction to succeed.

### OCR backend

`OCR_BACKEND=tesseract` works out of the box -- the Dockerfile installs the
`tesseract-ocr` binary and its Arabic and English language data via apt.
`OCR_BACKEND=paddle` needs `requirements-ocr.txt` installed too
(paddlepaddle + paddleocr, roughly +1 GB); its install line in the Dockerfile
is commented out by default. Uncomment it and rebuild to enable
`traditional:paddle`.

### A note on Arabic search in Postgres

`postgres:16-alpine`'s locale support determines whether the trigram index on
`raw_text` actually matches Arabic (see `sql/schema.sql` and CLAUDE.md's
seventh pinned decision). `app.db.init_schema()` checks this on every `api`
boot and logs a `WARNING` naming the database's `LC_CTYPE` if it is `C` --
watch the `api` container's startup logs for it after a first `up`.

## Pipeline stages

`app.pipeline.process` is one straight line, and every stage but the first
degrades instead of raising:

```
load -> truncate -> clean -> engine -> validate -> gate -> save
```

load a PDF or image and triage its pages (text layer vs. OCR) -- truncate to
`max_pages` -- clean (crop/deskew/lighting, skipped on a trusted text layer)
-- run the chosen `Extractor` -- run `app.validate`'s arithmetic and format
rules -- `decide_status` gates to review or auto-approve -- save to Postgres
if configured. `process` never raises; a failure at any stage comes back as
`ProcessResult(status="failed", error=...)` instead of a stack trace.

## Engines

| Key | Label | Boxes | Good for | Weak on |
|---|---|---|---|---|
| `traditional:paddle` | Traditional OCR (PaddleOCR) | yes | Arabic and mixed Arabic/English pages; shows where each value was read | A large one-off model download; slow first document after startup |
| `traditional:tesseract` | Traditional OCR (Tesseract) | yes | Clean English scans; quick to install and start | Weaker than PaddleOCR on Arabic; falls behind on photographs and faint print |
| `vlm:gemini` | Vision model (Google Gemini) | no | Phone photographs, creases, bad lighting, handwriting, scrambled layouts | No on-page positions or per-line confidence; answers can vary run to run |
| `vlm:openai` | Vision model (OpenAI GPT) | no | Same as Gemini's engine; the natural second opinion when Gemini looks wrong | Same loss of positions and confidence |

`GET /v1/engines` returns this list from `app.engines.available_engines()`
directly, so it cannot drift from what `POST /v1/documents/extract` will
actually run. `POST /v1/documents/compare` runs two or more engines against
the same loaded pages and reports per-field agreement -- see
`app.pipeline.field_diff`.

## Evaluation

`eval.py` scores field-level accuracy, not character error rate: an engine
can transcribe 98% of characters correctly and still get 40% of totals wrong,
because misreads cluster on digits rather than spreading evenly across a
page. It needs a folder of documents, each next to a same-stem `.json` of
human-checked fields:

```
samples/
  receipt-001.pdf
  receipt-001.json
  receipt-002.png
  receipt-002.json
```

```bash
python eval.py samples --doc-type receipt \
    --engines traditional:paddle,vlm:gemini \
    --out eval_results.csv
```

`--engines` defaults to all four registered keys; `--doc-type` is required
and must be one of `receipt`, `invoice`, `document`; `--no-clean` skips the
crop/deskew/lighting chain before extraction. The run writes a per-document,
per-field CSV to `--out` and prints, per engine, overall field accuracy and
the five weakest fields (line-item indexes collapsed, so
`items.2.unit_price` and `items.7.unit_price` count as the same field). It
runs with `persist=False` -- an eval run is not a record and never writes to
`documents` or `corrections`. Refuses to run, with a nonzero exit, on a
folder that is empty or has no labelled documents rather than inventing
ground truth.

## Adding a document type

1. In `app/schemas.py`, add a model subclassing `_Extracted` with a `Literal`
   `doc_type` discriminator, the way `Receipt`, `Invoice` and
   `GenericDocument` do. Field descriptions are prompt text -- write them for
   the model that will read them, not for a future maintainer.
2. Add it to the `Document` union and to the `DOC_TYPES` dict.

That is enough to reach every engine, the API, and the review UI: prompts are
built from `compact_schema`/`json_schema_for`, the API's `doc_type` parameter
is validated against `DOC_TYPES`, and `ui/streamlit_app.py` builds its review
form from `schema_for(doc_type).model_fields` -- none of them hardcode a
document type by name. Optionally add an entry to `_REQUIRED` in
`app/validate.py` if the new type has fields that should block auto-approval
when missing; without one, `validate` still runs its arithmetic and date/
currency rules, just no required-field check.

## Before production

- **Extraction runs on the request thread.** A 20-page PaddleOCR scan can
  hold an `api` worker for the better part of a minute. The upgrade path is
  a queue: enqueue `(bytes, filename, doc_type, engine, clean_images,
  persist)`, return `202` with a job id, have a worker call
  `app.pipeline.process` with exactly those arguments. `process` is already
  synchronous, takes bytes, never raises, and returns one value -- the shape
  a queue wants -- so nothing in `app/` has to change.
- **Original files are stored nowhere.** `documents.raw_extraction` keeps the
  full `ExtractionResult`, but the uploaded bytes themselves are never
  written anywhere after the request that processed them. A re-extraction,
  an audit, or a "show me the original PDF" request has nothing to read.
- **No auth, and CORS is open.** Every route in `api/main.py` is reachable by
  anyone who can reach the port; `reviewed_by` is a string the client
  supplies and nothing checks. Fine for a local deployment, not for a shared
  one -- and whatever answers it has to answer for both `api` and `ui`, since
  `ui` only ever talks to Postgres through `api`.

See CLAUDE.md's "Known gaps" for the rest -- most pressingly the provider-key
fields missing from `app/config.py`, without which no engine can complete a
real extraction.
