"""Self-contained demo UI: calls `app.pipeline.process` in-process.

Built for Streamlit Community Cloud, which runs exactly one Python file and
has nowhere to put a separate `api/main.py` process. `ui/streamlit_app.py`
stays the API-backed review UI for local and `docker-compose` use, unchanged;
this file trades that two-service architecture for a single deployable page,
in exchange for no persistence (`process(..., persist=False)`: nothing here
can be saved or approved, since that needs Postgres) and one fixed engine.

The engine is pinned to `vlm:gemini` rather than offered as a choice: it is
the only one of the four that needs neither an OCR binary
(`traditional:tesseract`) nor a ~1 GB pip package
(`traditional:paddle`) to run, which matters on a free-tier build.

Run locally with:

    streamlit run ui/cloud_app.py
"""

from __future__ import annotations

import streamlit as st

from app.config import settings
from app.pipeline import process
from app.schemas import DOC_TYPES, ProcessResult

ENGINE_KEY = "vlm:gemini"
_DOC_TYPE_ORDER = ["receipt", "invoice", "document"]
_ACCEPTED_SUFFIXES = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"]

st.set_page_config(page_title="OCR DMS demo", layout="wide")


def _md_safe(text: str | None) -> str | None:
    """See ui/streamlit_app.py's own copy: Streamlit's markdown parser drops
    everything after a ':' immediately followed by a letter, which both an
    engine key ("vlm:gemini") and an issue or error message can contain."""
    return text if text is None else text.replace(":", "\\:")


st.title("OCR DMS -- extraction demo")
st.caption(
    "Reads a receipt, invoice, or document with Google Gemini and shows the "
    "structured fields it extracted. Demo only: nothing uploaded here is saved."
)

if not settings.gemini_api_key.strip():
    st.error(
        "GEMINI_API_KEY is not set. On Streamlit Community Cloud: app menu -> "
        "Settings -> Secrets, add `GEMINI_API_KEY = \"...\"`. Locally: add it "
        "to .env. Then reload."
    )
    st.stop()

doc_types = [d for d in _DOC_TYPE_ORDER if d in DOC_TYPES]

with st.sidebar:
    st.header("Run")
    doc_type = st.selectbox("Document type", doc_types)
    uploaded = st.file_uploader("Document", type=_ACCEPTED_SUFFIXES)
    run = st.button("Run extraction", type="primary", disabled=uploaded is None)

if run and uploaded is not None:
    data = uploaded.getvalue()
    with st.spinner("Reading the document with Gemini..."):
        st.session_state.result = process(data, uploaded.name, doc_type, ENGINE_KEY, persist=False)

result: ProcessResult | None = st.session_state.get("result")
if result is None:
    st.info("Upload a document and press “Run extraction” to see results here.")
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
