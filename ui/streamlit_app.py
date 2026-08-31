"""Streamlit review UI: talks to `api/main.py` over HTTP, renders nothing itself.

Every decision about how a document is read stays behind the API -- this file
turns sidebar choices into a multipart request and a `ProcessResult` back into
widgets. The one thing it does without the API is put pixels on screen for the
Pages tab: the API never returns page images, so the pages are re-loaded and
re-cleaned locally through the same `app.pipeline.render_pages` chain the
engine itself was given, with the same clean-images switch, so what a reviewer
sees is what the extraction was read from.

Run with the API already up:

    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import logging
import os
import typing
import uuid
from typing import Any

import pandas as pd
import requests
import streamlit as st

from app.pipeline import render_pages
from app.schemas import DOC_TYPES, LineItem, ProcessResult, schema_for

logger = logging.getLogger(__name__)

# Not `app.config.settings`: this is the address of a separate process this
# file talks to over HTTP, not a setting of the extraction pipeline itself --
# the API reads `Settings` to configure itself, not to find itself.
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

_DOC_TYPE_ORDER = ["receipt", "invoice", "document"]
_LONG_TEXT_FIELDS = {"full_text", "summary"}
_ACCEPTED_SUFFIXES = ["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"]

st.set_page_config(page_title="OCR DMS", layout="wide")


def _md_safe(text: str | None) -> str | None:
    """Escape ':' before handing dynamic text to a markdown-rendering call.

    Streamlit's markdown reads a colon immediately followed by a letter as
    the start of a directive (":red[...]", an emoji shortcode) and drops
    everything from the colon onward when nothing recognisable follows --
    confirmed on engine keys like "traditional:paddle", which render as a
    bare "traditional". A colon followed by whitespace or a digit is never
    touched by the parser, so escaping every colon is a no-op there and a fix
    everywhere else, rather than something that has to be judged per string.
    """
    return text if text is None else text.replace(":", "\\:")


# --- Talking to the API ----------------------------------------------------


def _error_detail(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"the service returned HTTP {response.status_code} with no details"
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        message = str(detail.get("message", "unknown error"))
        valid = detail.get("valid")
        return f"{message} (valid: {', '.join(str(v) for v in valid)})" if valid else message
    return str(detail if detail is not None else body)


def _call(method: str, path: str, **kwargs: Any) -> Any:
    """One HTTP round trip, or a rendered error and a stopped script.

    Every call site wants the same three outcomes -- the JSON body, a plain
    "start the service" message when it is not running, and the API's own
    `code`/`message` when it refused the request -- so all three are handled
    here instead of at each call site.
    """
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=300, **kwargs)
    except requests.exceptions.ConnectionError:
        st.error(f"The extraction service is not reachable at {_md_safe(API_URL)}. Start it, then reload.")
        st.stop()
    except requests.exceptions.RequestException as exc:
        st.error(
            f"The request to {_md_safe(API_URL)} failed: {_md_safe(str(exc))}. "
            "Check the service, then try again."
        )
        st.stop()
    if response.status_code >= 400:
        st.error(_md_safe(_error_detail(response)))
        st.stop()
    return response.json()


@st.cache_data(ttl=30, show_spinner=False)
def _engines() -> list[dict[str, Any]]:
    return _call("GET", "/v1/engines")["engines"]


@st.cache_data(show_spinner="Rendering pages...")
def _cached_pages(data: bytes, filename: str, clean_images: bool) -> list[bytes]:
    """Cached so switching tabs does not re-run the crop/deskew/lighting chain."""
    return render_pages(data, filename, clean_images=clean_images)


# --- Sidebar -----------------------------------------------------------------

st.title("OCR DMS")

engines = _engines()
if not engines:
    st.error("The extraction service reports no engines. Check its configuration.")
    st.stop()
engine_labels = {engine["key"]: engine["label"] for engine in engines}

doc_types = [d for d in _DOC_TYPE_ORDER if d in DOC_TYPES] + sorted(
    set(DOC_TYPES) - set(_DOC_TYPE_ORDER)
)

with st.sidebar:
    st.header("Run")
    mode = st.radio("Mode", ["Extract", "Compare engines"])
    doc_type = st.selectbox("Document type", doc_types)

    engine_key: str | None = None
    engine_keys: list[str] = []
    if mode == "Extract":
        engine_key = st.radio(
            "Engine",
            [engine["key"] for engine in engines],
            format_func=lambda key: engine_labels[key],
            captions=[_md_safe(engine["note"]) for engine in engines],
        )
        engine_keys = [engine_key] if engine_key else []
    else:
        engine_keys = st.multiselect(
            "Engines to compare",
            [engine["key"] for engine in engines],
            default=[engine["key"] for engine in engines[:2]],
            format_func=lambda key: engine_labels[key],
        )

    clean_images = st.checkbox("Clean images (crop, deskew, lighting)", value=True)
    if mode == "Extract":
        persist = st.checkbox("Save to database", value=True)
    else:
        persist = False
        st.caption("Compare mode never saves. Run the engine you keep through Extract.")

    uploaded = st.file_uploader("Document", type=_ACCEPTED_SUFFIXES)
    run_label = "Run extraction" if mode == "Extract" else "Compare"
    run = st.button(run_label, type="primary", disabled=uploaded is None)

if run and uploaded is not None:
    if mode == "Compare engines" and len(engine_keys) < 2:
        st.sidebar.warning("Pick at least two engines to compare.")
    else:
        data = uploaded.getvalue()
        with st.spinner("Reading the document..."):
            if mode == "Extract":
                payload = _call(
                    "POST",
                    "/v1/documents/extract",
                    files={"file": (uploaded.name, data)},
                    data={
                        "doc_type": doc_type,
                        "engine": engine_key or "",
                        "clean_images": str(clean_images).lower(),
                        "persist": str(persist).lower(),
                    },
                )
                st.session_state.result = {"mode": "extract", "payload": payload}
            else:
                payload = _call(
                    "POST",
                    "/v1/documents/compare",
                    files={"file": (uploaded.name, data)},
                    data={
                        "doc_type": doc_type,
                        "engines": ",".join(engine_keys),
                        "clean_images": str(clean_images).lower(),
                    },
                )
                st.session_state.result = {"mode": "compare", "payload": payload}
        st.session_state.file_bytes = data
        st.session_state.file_name = uploaded.name
        st.session_state.clean_images = clean_images
        st.session_state.doc_type = doc_type
        # A fresh token per run, not per document id: an unsaved extraction has
        # no id at all, and reusing "draft" as a widget key would let a second
        # unsaved run inherit the first one's edited field values.
        st.session_state.run_token = uuid.uuid4().hex


# --- Issues ------------------------------------------------------------------


def _render_issues(issues: list[Any]) -> None:
    if not issues:
        return
    grouped: dict[str, list[str]] = {"error": [], "warning": [], "info": []}
    for issue in issues:
        line = f"**{issue.code}** -- {_md_safe(issue.message)}"
        if issue.field:
            line += f" (`{issue.field}`)"
        grouped.setdefault(issue.severity, []).append(line)
    if grouped["error"]:
        st.error("\n\n".join(grouped["error"]))
    if grouped["warning"]:
        st.warning("\n\n".join(grouped["warning"]))
    if grouped["info"]:
        st.info("\n\n".join(grouped["info"]))


# --- Review form ---------------------------------------------------------
# Built from the schema's own field list, never from the keys the response
# happened to contain: a null field and a field the extractor never mentioned
# look identical once the response has been reduced to a dict, but only the
# schema knows the field exists at all. Read a response dict's keys instead
# and a field the extractor dropped entirely would have no box to fix.


def _leaf_widget(container: Any, key: str, name: str, info: Any, value: Any) -> Any:
    label = name.replace("_", " ")
    args = typing.get_args(info.annotation)
    help_text = _md_safe(info.description)
    if float in args:
        return container.number_input(
            label,
            value=float(value) if value is not None else None,
            step=0.01,
            format="%.2f",
            help=help_text,
            key=key,
        )
    if int in args:
        return container.number_input(
            label,
            value=int(value) if value is not None else None,
            step=1,
            help=help_text,
            key=key,
        )
    widget = container.text_area if name in _LONG_TEXT_FIELDS else container.text_input
    typed = widget(label, value=value or "", help=help_text, key=key)
    return typed or None


def _items_editor(token: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns = list(LineItem.model_fields.keys())
    rows = [{col: item.get(col) for col in columns} for item in items]
    if not rows:
        # An empty table renders no columns to type into; one blank row gives
        # the reviewer somewhere to start, and a row left blank is dropped
        # again by `_clean_items` on submit.
        rows = [{col: None for col in columns}]

    column_config = {}
    for col in columns:
        info = LineItem.model_fields[col]
        label = col.replace("_", " ")
        if float in typing.get_args(info.annotation):
            column_config[col] = st.column_config.NumberColumn(
                label, format="%.2f", help=_md_safe(info.description)
            )
        else:
            column_config[col] = st.column_config.TextColumn(label, help=_md_safe(info.description))

    edited = st.data_editor(
        rows,
        column_config=column_config,
        column_order=columns,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=f"items:{token}",
    )
    return _clean_items(edited, columns)


def _clean_items(rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    cleaned = []
    for row in rows:
        values = {col: (row.get(col) or None) for col in columns}
        if any(value is not None for value in values.values()):
            cleaned.append(values)
    return cleaned


def _render_review_form(
    doc_type: str, document: Any, document_id: str | None, token: str
) -> None:
    model = schema_for(doc_type)
    data = document.model_dump() if document is not None else {}

    values: dict[str, Any] = {}
    scalar_fields = [name for name in model.model_fields if name not in ("doc_type", "items")]
    columns = st.columns(2)
    for index, name in enumerate(scalar_fields):
        values[name] = _leaf_widget(
            columns[index % 2],
            f"field:{token}:{name}",
            name,
            model.model_fields[name],
            data.get(name),
        )

    if "items" in model.model_fields:
        st.markdown("**Line items**")
        values["items"] = _items_editor(token, data.get("items") or [])

    reviewed_by = st.text_input("Reviewed by", key=f"reviewer:{token}")

    if st.button("Approve", disabled=document_id is None, key=f"approve:{token}"):
        corrected = {"doc_type": doc_type, **values}
        response = _call(
            "POST",
            f"/v1/documents/{document_id}/approve",
            json={"fields": corrected, "reviewed_by": reviewed_by or None},
        )
        st.success("Approved.")
        corrections = (response.get("document") or {}).get("corrections") or []
        if corrections:
            st.markdown("**Corrections recorded**")
            st.dataframe(pd.DataFrame(corrections), use_container_width=True, hide_index=True)
        else:
            st.caption("No fields changed.")

    if document_id is None:
        st.caption(
            "Approve is disabled: this run was not saved. Turn on \"Save to database\" and "
            "make sure the API has DATABASE_URL set, then run extraction again."
        )


# --- Recognised text and pages ---------------------------------------------


def _render_recognised_text(extraction: Any) -> None:
    if extraction is None:
        st.caption("No extraction to show.")
        return
    st.text_area("Transcript", value=extraction.raw_text or "", height=300, disabled=True)
    if extraction.text_blocks:
        rows = sorted(
            (block.model_dump() for block in extraction.text_blocks),
            key=lambda b: (b["confidence"] is None, b["confidence"]),
        )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("This engine reports no per-line text blocks.")


def _render_pages_tab() -> None:
    data = st.session_state.get("file_bytes")
    filename = st.session_state.get("file_name")
    if not data:
        st.caption("No file loaded.")
        return
    pages = _cached_pages(data, filename, st.session_state.get("clean_images", True))
    if not pages:
        st.caption("The page images could not be rendered.")
        return
    for number, png in enumerate(pages, start=1):
        st.image(png, caption=f"Page {number}", use_column_width=True)


# --- Extract mode ------------------------------------------------------------


def _render_extract(payload: dict[str, Any]) -> None:
    result = ProcessResult.model_validate(payload)
    extraction = result.extraction

    cols = st.columns(5)
    cols[0].metric("Engine", (extraction.engine if extraction else None) or "n/a")
    pages = extraction.page_count if extraction else None
    cols[1].metric("Pages", pages if pages is not None else "n/a")
    cols[2].metric("Seconds", f"{(result.duration_ms or 0) / 1000:.1f}")
    confidence = extraction.confidence if extraction else None
    cols[3].metric("OCR confidence", f"{confidence:.0%}" if confidence is not None else "n/a")
    status = result.status + (" · needs review" if result.needs_review else "")
    cols[4].metric("Status", status)

    if result.status == "failed":
        st.error(f"The run failed: {_md_safe(result.error) or 'no reason was given'}")

    _render_issues(result.issues)

    doc_type = result.doc_type or st.session_state.get("doc_type") or "receipt"
    token = st.session_state.get("run_token", "draft")

    tab_review, tab_text, tab_pages, tab_json = st.tabs(
        ["Review fields", "Recognised text", "Pages", "Raw JSON"]
    )
    with tab_review:
        _render_review_form(
            doc_type, extraction.document if extraction else None, result.document_id, token
        )
    with tab_text:
        _render_recognised_text(extraction)
    with tab_pages:
        _render_pages_tab()
    with tab_json:
        st.json(payload)


# --- Compare mode --------------------------------------------------------


def _render_compare(payload: dict[str, Any]) -> None:
    requested = payload.get("engines", [])
    results = [ProcessResult.model_validate(row) for row in payload.get("results", [])]
    diff = payload.get("diff", [])
    agreement = payload.get("agreement")

    columns = st.columns(len(results)) if results else []
    for column, requested_key, result in zip(columns, requested, results):
        with column:
            extraction = result.extraction
            st.subheader(_md_safe((extraction.engine if extraction else None) or requested_key))
            pages = extraction.page_count if extraction else None
            st.metric("Pages", pages if pages is not None else "n/a")
            st.metric("Seconds", f"{(result.duration_ms or 0) / 1000:.1f}")
            confidence = extraction.confidence if extraction else None
            st.metric("OCR confidence", f"{confidence:.0%}" if confidence is not None else "n/a")
            status = result.status + (" · needs review" if result.needs_review else "")
            st.metric("Status", status)
            if result.status == "failed":
                st.caption(f"Failed: {_md_safe(result.error) or 'no reason was given'}")

    st.metric("Agreement", f"{agreement:.0%}" if agreement is not None else "n/a")

    if not diff:
        st.info("Nothing to compare: neither engine returned a document.")
        return

    frame = pd.DataFrame([{"field": row["path"], **row["values"]} for row in diff])
    mask = [not row["agree"] for row in diff]

    st.markdown("**Fields where the engines disagree**")
    disagreements = frame[mask]
    if disagreements.empty:
        st.caption("The engines agree on every field they answered.")
    else:
        st.dataframe(disagreements, use_container_width=True, hide_index=True)

    with st.expander(f"All {len(frame)} compared fields"):
        st.dataframe(frame, use_container_width=True, hide_index=True)


# --- Results -------------------------------------------------------------

result_state = st.session_state.get("result")
if result_state is None:
    st.info("Upload a document and press Run to see results here.")
elif result_state["mode"] == "extract":
    _render_extract(result_state["payload"])
else:
    _render_compare(result_state["payload"])
