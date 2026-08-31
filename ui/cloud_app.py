"""Self-contained demo UI: calls `app.pipeline.process` in-process.

Built for Streamlit Community Cloud, which runs exactly one Python file and
has nowhere to put a separate `api/main.py` process. `ui/streamlit_app.py`
stays the API-backed review UI for local and `docker-compose` use, unchanged;
this file trades that two-service architecture for a single deployable page,
in exchange for no persistence (`process(..., persist=False)`: nothing here
can be saved or approved, since that needs Postgres).

Engine choice is limited to three of the four the registry knows:
`traditional:paddle` is a ~1 GB pip install, unfit for a free-tier build on
any platform, so it never appears here. The other three split into two
methods, which the sidebar offers as its own top-level radio before either
sub-choice: Vision LLM (`vlm:gemini` / `vlm:openai`, needs a provider key)
and Traditional OCR (`traditional:tesseract`, the `tesseract-ocr` apt
package via packages.txt and nothing else -- stage two is keyword and
pattern matching in `app.engines.traditional`, not a model call, so this
path needs no API key and keeps working when every provider is out of
quota). packages.txt has broken this deployment's build before on Streamlit
Cloud's own apt sources -- see the Dockerfile and requirements.txt comments
for that history -- but the earlier failure was specifically
`libglib2.0-0`, dropped when opencv-python-headless replaced opencv-python;
tesseract-ocr's own dependencies do not touch it.

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

_METHOD_ENGINES = {
    "Vision LLM": ("vlm:gemini", "vlm:openai"),
    "Traditional OCR": ("traditional:tesseract",),
}
_DOC_TYPE_ORDER = ["receipt", "invoice", "document"]
_ACCEPTED_SUFFIXES = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"]

st.set_page_config(page_title="OCR DMS demo", layout="wide")


def _md_safe(text: str | None) -> str | None:
    """See ui/streamlit_app.py's own copy: Streamlit's markdown parser drops
    everything after a ':' immediately followed by a letter, which an engine
    key ("vlm:gemini"), an engine note, or an issue/error message can contain."""
    return text if text is None else text.replace(":", "\\:")


def _cloud_engines() -> dict[str, dict[str, str]]:
    """Every engine offered anywhere on this page, keyed by its registry key
    and described the way app.engines already describes it -- reused rather
    than restated, so this file cannot drift from what `available_engines()`
    says elsewhere in the app."""
    offered = {key for keys in _METHOD_ENGINES.values() for key in keys}
    return {e.key: {"label": e.label, "note": e.note} for e in available_engines() if e.key in offered}


def _key_setting_for(engine_key: str) -> str | None:
    """Which Settings field a missing-key warning should point at, or None if the
    engine needs no key at all.

    vlm:* pins its own provider explicitly (VLMExtractor(provider=...)), so the
    key is always the one matching its name. traditional:tesseract needs
    nothing: both its stages -- OCR, then keyword/pattern matching in
    app.engines.traditional -- run locally.
    """
    if engine_key == "vlm:openai":
        return "openai_api_key"
    if engine_key == "vlm:gemini":
        return "gemini_api_key"
    return None


st.title("OCR DMS -- extraction demo")
st.caption(
    "Reads a receipt, invoice, or document and shows the structured fields it "
    "extracted. Demo only: nothing uploaded here is saved."
)

if not (settings.gemini_api_key.strip() or settings.openai_api_key.strip()):
    st.info(
        "No provider key is set, so **Traditional OCR** is the only method that will "
        "work right now (it needs no key). To also use **Vision LLM**: app menu -> "
        "Settings -> Secrets, add `GEMINI_API_KEY = \"...\"` or "
        "`OPENAI_API_KEY = \"...\"`. Locally: add one to .env. Then reload."
    )

doc_types = [d for d in _DOC_TYPE_ORDER if d in DOC_TYPES]
engines = _cloud_engines()

with st.sidebar:
    st.header("Run")
    doc_type = st.selectbox("Document type", doc_types)

    method = st.radio("Method", list(_METHOD_ENGINES), captions=[
        "Reads the page images directly, one model call.",
        "Recognises text with Tesseract, then fills the fields by pattern "
        "matching -- no API key needed.",
    ])
    method_keys = _METHOD_ENGINES[method]
    if len(method_keys) > 1:
        engine_key = st.radio(
            "Provider",
            method_keys,
            format_func=lambda k: engines[k]["label"],
            captions=[_md_safe(engines[k]["note"]) for k in method_keys],
        )
    else:
        engine_key = method_keys[0]
        st.caption(_md_safe(engines[engine_key]["note"]))

    key_setting = _key_setting_for(engine_key)
    if key_setting is not None and not getattr(settings, key_setting, "").strip():
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
    with st.spinner(f"Reading the document with {engines[engine_key]['label']}..."):
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
