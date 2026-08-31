"""app.llm tests.

Two of the three things this module does can only be checked here, because
both fail at request time rather than locally: the Gemini schema dialect
surfaces as a 400 on a live call, and a preamble in front of a JSON object
surfaces as a silent extraction of nothing. So the schema is asserted against
the SDK's own `Schema` type, not only against the absence of some strings.

No test touches the network. The seam is `_call_gemini`, which is the boundary
between our code and the SDK, and `settings` is replaced wholesale so that a
developer's real `.env` cannot change an outcome here.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app import llm
from app.llm import LLMError, _gemini_schema, parse_json_object, structured_text, structured_vision
from app.schemas import json_schema_for

DOC_TYPES = ("receipt", "invoice", "document")


@pytest.fixture
def stub_settings(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Settings with no provider key of any kind, so the guards are what is under test."""
    stub = SimpleNamespace(
        llm_provider="gemini",
        llm_model="gemini-2.5-flash",
        vision_model="gemini-2.5-flash",
        llm_max_tokens=1024,
        llm_timeout_seconds=5.0,
        llm_max_retries=3,
    )
    monkeypatch.setattr(llm, "settings", stub)
    return stub


# --- The Gemini schema dialect -------------------------------------------


@pytest.mark.parametrize("doc_type", DOC_TYPES)
def test_converted_schema_carries_no_ref_defs_or_anyof(doc_type: str) -> None:
    serialised = json.dumps(_gemini_schema(json_schema_for(doc_type)), ensure_ascii=False)

    assert "$ref" not in serialised
    assert "$defs" not in serialised
    assert "anyOf" not in serialised


def test_nested_line_item_quantity_is_nullable() -> None:
    """Proof the walk went inside $defs: quantity only exists behind a $ref."""
    converted = _gemini_schema(json_schema_for("receipt"))

    quantity = converted["properties"]["items"]["items"]["properties"]["quantity"]

    assert quantity["nullable"] is True
    assert quantity["type"] == "number"
    assert quantity["description"].startswith("Number of units")


@pytest.mark.parametrize("doc_type", DOC_TYPES)
def test_converted_schema_is_accepted_by_the_genai_schema_type(doc_type: str) -> None:
    """The string checks above are necessary, not sufficient; the SDK type is the real gate."""
    from google.genai import types

    schema = types.Schema.model_validate(_gemini_schema(json_schema_for(doc_type)))

    assert schema.properties
    assert schema.any_of is None
    assert schema.defs is None
    assert schema.ref is None


def test_const_becomes_a_single_member_enum() -> None:
    doc_type = _gemini_schema(json_schema_for("receipt"))["properties"]["doc_type"]

    assert doc_type["enum"] == ["receipt"]
    assert doc_type["type"] == "string"
    assert "const" not in doc_type


def test_title_and_default_are_dropped_but_a_field_called_title_is_not() -> None:
    """`title` is a keyword to drop and a GenericDocument field to keep."""
    converted = _gemini_schema(json_schema_for("document"))

    assert "title" not in converted
    assert "title" in converted["properties"]
    assert "default" not in converted["properties"]["title"]
    assert converted["properties"]["title"]["nullable"] is True


def test_receipt_items_survives_being_named_after_a_keyword() -> None:
    items = _gemini_schema(json_schema_for("receipt"))["properties"]["items"]

    assert items["type"] == "array"
    assert items["items"]["type"] == "object"
    assert "description_ar" in items["items"]["properties"]


def test_unresolvable_ref_raises_llm_error() -> None:
    schema = {"type": "object", "properties": {"a": {"$ref": "#/$defs/Missing"}}}

    with pytest.raises(LLMError, match="Missing"):
        _gemini_schema(schema)


def test_a_recursive_ref_terminates() -> None:
    schema = {
        "$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}},
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/Node"}},
    }

    converted = _gemini_schema(schema)

    assert converted["properties"]["root"]["properties"]["child"] == {"type": "object"}


def test_anyof_of_two_real_types_is_narrowed_rather_than_emitted() -> None:
    schema = {
        "type": "object",
        "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]}},
    }

    x = _gemini_schema(schema)["properties"]["x"]

    assert x == {"type": "string", "nullable": True}


# --- parse_json_object ----------------------------------------------------


def test_parses_a_fenced_object() -> None:
    text = '```json\n{"total": 12.5, "currency": "SAR"}\n```'

    assert parse_json_object(text) == {"total": 12.5, "currency": "SAR"}


def test_parses_an_object_behind_a_sentence() -> None:
    text = 'Here is the extracted receipt:\n{"merchant_name": "Tamimi", "total": 41.0}'

    assert parse_json_object(text) == {"merchant_name": "Tamimi", "total": 41.0}


def test_parses_a_fenced_object_with_a_sentence_on_both_sides() -> None:
    text = 'Sure. \n```\n{"a": {"b": 1}}\n```\nLet me know if you need more.'

    assert parse_json_object(text) == {"a": {"b": 1}}


def test_a_brace_inside_a_string_does_not_end_the_object() -> None:
    text = 'Result: {"note": "closes with } here", "n": 1} and that is all.'

    assert parse_json_object(text) == {"note": "closes with } here", "n": 1}


def test_arabic_values_survive_the_round_trip() -> None:
    text = '```json\n{"merchant_name_ar": "متجر التميمي"}\n```'

    assert parse_json_object(text) == {"merchant_name_ar": "متجر التميمي"}


@pytest.mark.parametrize(
    "text",
    [
        "I was unable to read this document.",
        "",
        "   \n  ",
        "[1, 2, 3]",
        '{"unclosed": ',
        "null",
    ],
)
def test_garbage_raises_llm_error(text: str) -> None:
    with pytest.raises(LLMError):
        parse_json_object(text)


def test_none_raises_llm_error() -> None:
    """A blocked or truncated Gemini response has `.text` of None, not of ''."""
    with pytest.raises(LLMError, match="empty"):
        parse_json_object(None)


# --- Guards that fire before a request is built ---------------------------


def test_missing_api_key_names_the_provider(stub_settings: SimpleNamespace) -> None:
    with pytest.raises(LLMError, match="gemini"):
        structured_text("read this", json_schema_for("receipt"))


def test_missing_api_key_names_a_requested_provider(stub_settings: SimpleNamespace) -> None:
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        structured_text("read this", json_schema_for("receipt"), provider="openai")


def test_unknown_provider_raises(stub_settings: SimpleNamespace) -> None:
    with pytest.raises(LLMError, match="unknown provider"):
        structured_text("read this", json_schema_for("receipt"), provider="bedrock")


def test_deepseek_serves_no_vision_model(stub_settings: SimpleNamespace) -> None:
    stub_settings.deepseek_api_key = "test-key"

    with pytest.raises(LLMError, match="vision"):
        structured_vision("read this", b"jpeg-bytes", json_schema_for("receipt"), provider="deepseek")


def test_structured_vision_needs_a_page(stub_settings: SimpleNamespace) -> None:
    with pytest.raises(LLMError, match="at least one page"):
        structured_vision("read this", [], json_schema_for("receipt"))


def test_a_model_from_another_provider_is_replaced(stub_settings: SimpleNamespace) -> None:
    stub_settings.llm_model = "claude-opus-5"

    assert llm._resolve_model(llm.PROVIDERS["gemini"], None, vision=False) == "gemini-2.5-flash"
    assert llm._resolve_model(llm.PROVIDERS["gemini"], "gemini-2.5-pro", vision=False) == "gemini-2.5-pro"


# --- Wiring and retry -----------------------------------------------------


def test_structured_text_returns_the_parsed_object(
    stub_settings: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_settings.gemini_api_key = "test-key"
    seen: dict[str, Any] = {}

    def fake_call(target: Any, prompt: str, schema: dict[str, Any], *rest: Any) -> str:
        seen["model"] = target.model
        seen["prompt"] = prompt
        seen["schema"] = schema
        return '```json\n{"doc_type": "receipt", "total": 9.0}\n```'

    monkeypatch.setattr(llm, "_call_gemini", fake_call)

    result = structured_text("read this", json_schema_for("receipt"))

    assert result == {"doc_type": "receipt", "total": 9.0}
    assert seen["model"] == "gemini-2.5-flash"
    assert seen["prompt"] == "read this"


def test_a_malformed_response_is_retried_not_raised(
    stub_settings: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_settings.gemini_api_key = "test-key"
    stub_settings.llm_max_retries = 2
    calls: list[int] = []

    def flaky(*_: Any) -> str:
        calls.append(1)
        return "I could not read it." if len(calls) == 1 else '{"doc_type": "receipt"}'

    monkeypatch.setattr(llm, "_call_gemini", flaky)

    assert structured_text("read this", json_schema_for("receipt")) == {"doc_type": "receipt"}
    assert len(calls) == 2


def test_the_last_failure_is_reraised(
    stub_settings: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_settings.gemini_api_key = "test-key"
    stub_settings.llm_max_retries = 1  # one attempt, so the test does not sleep
    monkeypatch.setattr(llm, "_call_gemini", lambda *_: "no json here at all")

    with pytest.raises(LLMError, match="no JSON object"):
        structured_text("read this", json_schema_for("receipt"))


def test_retry_policy_is_read_from_settings(stub_settings: SimpleNamespace) -> None:
    stub_settings.llm_max_retries = 5

    policy = llm._retrying()

    assert policy.stop.max_attempt_number == 5
    assert (policy.wait.min, policy.wait.max) == (1, 12)
    assert set(policy.retry.exception_types) == {LLMError, TimeoutError, ConnectionError}
    assert policy.reraise is True
