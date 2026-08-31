"""Self-contained demo UI: calls `app.pipeline.process` in-process.

Built for Streamlit Community Cloud, which runs exactly one Python file and
has nowhere to put a separate `api/main.py` process. `ui/streamlit_app.py`
stays the API-backed review UI for local and `docker-compose` use, unchanged;
this file trades that two-service architecture for a single deployable page,
in exchange for no persistence (`process(..., persist=False)`: nothing here
can be saved or approved, since that needs Postgres).

Engine choice is limited to the two `vlm:*` keys, not all four the registry
knows: `traditional:tesseract` needs the `tesseract-ocr` apt package, and
`packages.txt` has already broken this deployment's build twice on Streamlit
Cloud's own apt sources -- see the Dockerfile and requirements.txt comments
for that history. `traditional:paddle` is a ~1 GB pip install, unfit for a
free-tier build regardless of platform. Both `vlm:*` engines need nothing but
a provider API key.

Run locally with:

    streamlit run ui/cloud_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit Community Cloud's launcher only puts this file's own directory
# (ui/) on sys.path, not the repository root -- `app` lives beside `ui/`, not
# inside it, so "from app.config import settings" fails with
# ModuleNotFoundError there even though it works locally, where `streamlit
# run` is typically launched with the repo root already on sys.path. Adding
# the repo root explicitly, relative to this file, makes the import work
# regardless of how or from where the process was started.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.config import settings
from app.engines import available_engines
from app.pipeline import process
from app.schemas import DOC_TYPES, ProcessResult

_CLOUD_ENGINE_KEYS = ("vlm:gemini", "vlm:openai")
_DOC_TYPE_ORDER = ["receipt", "invoice", "document"]
_ACCEPTED_SUFFIXES = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"]

st.set_page_config(page_title="OCR DMS demo", layout="wide")


def _md_safe(text: str | None) -> str | None:
    """See ui/streamlit_app.py's own copy: Streamlit's markdown parser drops
    everything after a ':' immediately followed by a letter, which an engine
    key ("vlm:gemini"), an engine note, or an issue/error message can contain."""
    return text if text is None else text.replace(":", "\\:")


def _cloud_engines() -> list[dict[str, str]]:
    """The two vlm:* engines, described the way app.engines already describes
    them -- reused rather than restated, so this file cannot drift from what
    `available_engines()` says elsewhere in the app."""
    return [
        {"key": e.key, "label": e.label, "note": e.note}
        for e in available_engines()
        if e.key in _CLOUD_ENGINE_KEYS
    ]


_KEY_SETTING = {"vlm:gemini": "gemini_api_key", "vlm:openai": "openai_api_key"}


st.title("OCR DMS -- extraction demo")
st.caption(
    "Reads a receipt, invoice, or document and shows the structured fields it "
    "extracted. Demo only: nothing uploaded here is saved."
)

if not (settings.gemini_api_key.strip() or settings.openai_api_key.strip()):
    st.error(
        "No provider key is set. On Streamlit Community Cloud: app menu -> "
        "Settings -> Secrets, add `GEMINI_API_KEY = \"...\"` or "
        "`OPENAI_API_KEY = \"...\"`. Locally: add one to .env. Then reload."
    )
    st.stop()

doc_types = [d for d in _DOC_TYPE_ORDER if d in DOC_TYPES]
engines = _cloud_engines()

with st.sidebar:
    st.header("Run")
    doc_type = st.selectbox("Document type", doc_types)

    engine_key = st.radio(
        "Engine",
        [e["key"] for e in engines],
        format_func=lambda k: next(e["label"] for e in engines if e["key"] == k),
        captions=[_md_safe(e["note"]) for e in engines],
    )
    key_setting = _KEY_SETTING[engine_key]
    if not getattr(settings, key_setting, "").strip():
        st.caption(f"⚠️ {key_setting.upper()} is not set -- this engine will fail until it is.")

    st.divider()
    input_mode = st.radio("Input", ["Upload a file", "Use camera"], horizontal=True)
    if input_mode == "Upload a file":
        uploaded = st.file_uploader("Document", type=_ACCEPTED_SUFFIXES)
        data = uploaded.getvalue() if uploaded is not None else None
        filename = uploaded.name if uploaded is not None else None
    else:
        st.caption("Photographs work best flat, well lit, and square to the camera.")
        photo = st.camera_input("Take a photo of the document")
        data = photo.getvalue() if photo is not None else None
        filename = "camera_capture.jpg"

    run = st.button("Run extraction", type="primary", disabled=data is None)

if run and data is not None:
    engine_label = next(e["label"] for e in engines if e["key"] == engine_key)
    with st.spinner(f"Reading the document with {engine_label}..."):
        st.session_state.result = process(data, filename, doc_type, engine_key, persist=False)

result: ProcessResult | None = st.session_state.get("result")
if result is None:
    st.info("Upload a document or take a photo, then press “Run extraction” to see results here.")
else:
    extraction = result.extraction
    cols = st.columns(4)
    cols[0].metric("Pages", extraction.page_count if extraction else "n/a")
    cols[1].metric("Seconds", f"{(result.duration_ms or 0) / 1000:.1f}")
    status = result.status + (" · needs review" if result.needs_review else "")
    cols[2].metric("Status", status)
    cols[3].metric("Engine", (extraction.engine if extraction else None) or "n/a")

    if result.status == "failed":
        st.error(f"The run failed: {_md_safe(result.error) or 'no reason was given'}")

    for issue in result.issues:
        line = f"**{issue.code}** -- {_md_safe(issue.message)}"
        if issue.field:
            line += f" (`{issue.field}`)"
        renderer = st.error if issue.severity == "error" else st.warning if issue.severity == "warning" else st.info
        renderer(line)

    tab_fields, tab_text, tab_json = st.tabs(["Extracted fields", "Recognised text", "Raw JSON"])
    with tab_fields:
        if extraction is not None and extraction.document is not None:
            st.json(extraction.document.model_dump())
        else:
            st.caption("No document was extracted.")
    with tab_text:
        st.text_area(
            "Transcript",
            value=(extraction.raw_text if extraction else "") or "",
            height=300,
            disabled=True,
        )
    with tab_json:
        st.json(result.model_dump(mode="json"))
