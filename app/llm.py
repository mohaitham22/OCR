"""Every model call in the app.

One module so that three things stay in one place: which provider is being
talked to, the fact that a model's answer is text until proven otherwise, and
the retry policy that turns a flaky call into a slow one instead of a failure.

The awkward part is `_gemini_schema`. Gemini will not accept the JSON Schema
Pydantic emits -- `$ref`, `$defs`, `anyOf`, `title`, `default`,
`additionalProperties` and `const` are all rejected -- and it rejects them with
a 400 at request time, so a schema change that breaks extraction looks fine
locally and fails in production. Sanitising here, on the way out, is the only
place that catch is cheap.

OpenAI and DeepSeek get the schema as prose instead: their portable JSON mode
(`response_format={"type": "json_object"}`) constrains the syntax but not the
shape, and DeepSeek does not implement the schema-typed variant at all. That is
why `parse_json_object` tolerates a preamble -- a model told to describe its
output in words often does.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Anything that stopped a model call from producing a usable JSON object."""


# --- Providers -----------------------------------------------------------
# app.config.settings still carries the Anthropic-shaped fields this module
# replaced and has no provider, per-provider key or per-provider model yet.
# Until it does, every read here goes through getattr on settings -- never
# os.getenv -- so config remains the single source and adding the fields is a
# pure deletion of the fallbacks. Same pattern as _MAX_PAGES_FALLBACK in
# app.preprocess.


@dataclass(frozen=True, slots=True)
class _Provider:
    name: str
    key_setting: str
    sdk: str
    default_text_model: str
    default_vision_model: str
    model_prefixes: tuple[str, ...]
    base_url: str | None = None
    supports_vision: bool = True


PROVIDERS: dict[str, _Provider] = {
    "gemini": _Provider(
        name="gemini",
        key_setting="gemini_api_key",
        sdk="genai",
        default_text_model="gemini-2.5-flash",
        default_vision_model="gemini-2.5-flash",
        model_prefixes=("gemini", "models/gemini"),
    ),
    "openai": _Provider(
        name="openai",
        key_setting="openai_api_key",
        sdk="openai",
        default_text_model="gpt-4.1-mini",
        default_vision_model="gpt-4.1-mini",
        model_prefixes=("gpt", "chatgpt", "o1", "o3", "o4"),
    ),
    "deepseek": _Provider(
        name="deepseek",
        key_setting="deepseek_api_key",
        sdk="openai",
        default_text_model="deepseek-chat",
        default_vision_model="deepseek-chat",
        model_prefixes=("deepseek",),
        base_url="https://api.deepseek.com/v1",
        supports_vision=False,
    ),
}

_DEFAULT_PROVIDER = "gemini"
_SHARED_KEY_SETTING = "llm_api_key"


@dataclass(frozen=True, slots=True)
class _Target:
    provider: _Provider
    model: str
    api_key: str


def _setting(name: str, default: Any) -> Any:
    return getattr(settings, name, default)


def _resolve(provider: str | None, model: str | None, *, vision: bool) -> _Target:
    """Pick provider, model and key, and fail here rather than at request time.

    Called outside the retry wrapper on purpose: a missing key is not a
    transient fault, and retrying it only delays the message by three seconds.
    """
    name = (provider or _setting("llm_provider", _DEFAULT_PROVIDER) or _DEFAULT_PROVIDER).strip().lower()
    if name not in PROVIDERS:
        raise LLMError(f"unknown provider {name!r}; expected one of {sorted(PROVIDERS)}")
    chosen = PROVIDERS[name]

    if vision and not chosen.supports_vision:
        raise LLMError(f"provider {chosen.name!r} serves no vision model; use gemini or openai")

    key = str(_setting(chosen.key_setting, "") or _setting(_SHARED_KEY_SETTING, "") or "").strip()
    if not key:
        raise LLMError(
            f"no API key for provider {chosen.name!r}: set {chosen.key_setting.upper()} in the environment"
        )

    return _Target(provider=chosen, model=_resolve_model(chosen, model, vision=vision), api_key=key)


def _resolve_model(provider: _Provider, requested: str | None, *, vision: bool) -> str:
    if requested:
        return requested
    configured = str(_setting("vision_model" if vision else "llm_model", "") or "").strip()
    if configured.lower().startswith(provider.model_prefixes):
        return configured
    fallback = provider.default_vision_model if vision else provider.default_text_model
    if configured:
        # settings still defaults these to an Anthropic id, which every provider
        # here answers with a 404. Prefer a model the provider actually serves.
        logger.warning(
            "configured model %r does not belong to provider %r; using %s",
            configured,
            provider.name,
            fallback,
        )
    return fallback


# --- Retry ---------------------------------------------------------------
# LLMError covers both halves of what is worth retrying: a transport failure
# (mapped at the SDK boundary, where the SDK is imported) and a response that
# did not parse. Per the schema contract, a malformed answer is a retry, not an
# exception the caller has to think about.
_RETRYABLE: tuple[type[BaseException], ...] = (LLMError, TimeoutError, ConnectionError)


def _retrying() -> Retrying:
    return Retrying(
        stop=stop_after_attempt(max(1, int(_setting("llm_max_retries", 3)))),
        wait=wait_exponential(multiplier=1, min=1, max=12),
        retry=retry_if_exception_type(_RETRYABLE),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# --- Public API ----------------------------------------------------------


def structured_text(
    prompt: str,
    json_schema: dict[str, Any],
    *,
    system: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    target = _resolve(provider, model, vision=False)
    return _retrying()(_call, target, prompt, json_schema, system, max_tokens, (), "image/jpeg")


def structured_vision(
    prompt: str,
    images: bytes | Sequence[bytes],
    json_schema: dict[str, Any],
    *,
    system: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    mime_type: str = "image/jpeg",
) -> dict[str, Any]:
    pages: tuple[bytes, ...] = (images,) if isinstance(images, bytes) else tuple(images)
    if not pages:
        raise LLMError("structured_vision needs at least one page image")
    target = _resolve(provider, model, vision=True)
    return _retrying()(_call, target, prompt, json_schema, system, max_tokens, pages, mime_type)


def parse_json_object(text: str | None) -> dict[str, Any]:
    """Recover the JSON object from whatever the model wrapped around it.

    A code fence and a sentence of preamble are the two things models add even
    when told not to, and both are recoverable; anything else is a retry.
    """
    if not text or not text.strip():
        raise LLMError("model returned an empty response")

    for candidate in [match.group(1) for match in _FENCE.finditer(text)] + [text]:
        found = _first_json_object(candidate)
        if found is not None:
            return found

    raise LLMError(f"no JSON object in model response: {_excerpt(text)}")


# --- Dispatch ------------------------------------------------------------


def _call(
    target: _Target,
    prompt: str,
    json_schema: dict[str, Any],
    system: str | None,
    max_tokens: int | None,
    images: tuple[bytes, ...],
    mime_type: str,
) -> dict[str, Any]:
    tokens = max_tokens or int(_setting("llm_max_tokens", 8192))
    if target.provider.sdk == "genai":
        raw = _call_gemini(target, prompt, json_schema, system, tokens, images, mime_type)
    else:
        raw = _call_openai(target, prompt, json_schema, system, tokens, images, mime_type)
    return parse_json_object(raw)


def _call_gemini(
    target: _Target,
    prompt: str,
    json_schema: dict[str, Any],
    system: str | None,
    max_tokens: int,
    images: tuple[bytes, ...],
    mime_type: str,
) -> str | None:
    from google import genai  # heavy and optional: the app boots without it
    from google.genai import types

    parts = [types.Part.from_text(text=prompt)]
    parts += [types.Part.from_bytes(data=page, mime_type=mime_type) for page in images]

    try:
        client = genai.Client(
            api_key=target.api_key,
            http_options=types.HttpOptions(
                timeout=int(float(_setting("llm_timeout_seconds", 120.0)) * 1000)
            ),
        )
        response = client.models.generate_content(
            model=target.model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=max_tokens,
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=_gemini_schema(json_schema),
            ),
        )
    except LLMError:
        raise
    except Exception as exc:  # every SDK failure becomes one retryable class
        raise LLMError(f"gemini call failed ({target.model}): {exc}") from exc

    return response.text


def _call_openai(
    target: _Target,
    prompt: str,
    json_schema: dict[str, Any],
    system: str | None,
    max_tokens: int,
    images: tuple[bytes, ...],
    mime_type: str,
) -> str | None:
    from openai import OpenAI  # heavy and optional: the app boots without it

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for page in images:
        encoded = base64.b64encode(page).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}})

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _json_instruction(system, json_schema)},
        {"role": "user", "content": content},
    ]

    try:
        client = OpenAI(
            api_key=target.api_key,
            base_url=target.provider.base_url,
            timeout=float(_setting("llm_timeout_seconds", 120.0)),
            max_retries=0,  # tenacity owns the retry policy; two of them compound badly
        )
        response = client.chat.completions.create(
            model=target.model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"{target.provider.name} call failed ({target.model}): {exc}") from exc

    return response.choices[0].message.content


def _json_instruction(system: str | None, json_schema: dict[str, Any]) -> str:
    """JSON mode fixes the syntax, not the shape, so the shape goes in the prompt."""
    return "\n\n".join(
        part
        for part in (
            system,
            "Reply with a single JSON object and nothing else -- no prose, no code fence. "
            "It must conform to this JSON Schema. Use null for anything the document does "
            "not carry; never invent a value.",
            json.dumps(json_schema, ensure_ascii=False, indent=2),
        )
        if part
    )


# --- Gemini schema dialect -----------------------------------------------

# Keywords Gemini rejects outright and that cost nothing to lose: `title` and
# `default` are noise, and `additionalProperties` is already implied by our
# models' extra="ignore". `$ref`, `$defs`, `anyOf` and `const` are absent from
# this set because they are translated rather than dropped.
_DROPPED = frozenset(
    {"title", "default", "additionalProperties", "$schema", "$comment", "examples", "discriminator"}
)

# Keywords whose values are themselves schemas, and so have to be walked.
_NESTED = ("items", "contains", "not")


def _gemini_schema(json_schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a Pydantic JSON Schema into the subset Gemini accepts.

    Inlines `$ref` against `$defs`, collapses an `anyOf` of a type and null into
    that type plus `nullable: True`, turns `const` into a single-member `enum`,
    and drops the rest. The walk recurses because the fields most likely to
    break -- the line items -- live behind a `$ref` into `$defs`, so cleaning
    only the top level would still 400.
    """
    return _convert(json_schema, json_schema.get("$defs", {}), ())


def _convert(node: Any, defs: dict[str, Any], seen: tuple[str, ...]) -> Any:
    if not isinstance(node, dict):
        return node

    if "$ref" in node:
        return _inline_ref(node, defs, seen)

    if "anyOf" in node:
        return _collapse_any_of(node, defs, seen)

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROPPED or key == "$defs":
            continue
        if key == "const":
            out["enum"] = [value]
        elif key == "properties":
            # Property *names* are data, never keywords: filter the schemas
            # under this map, never its keys. Receipt has a field called
            # `items` and GenericDocument one called `title`.
            out["properties"] = {name: _convert(sub, defs, seen) for name, sub in value.items()}
        elif key in _NESTED:
            out[key] = _convert(value, defs, seen)
        else:
            out[key] = value
    return out


def _inline_ref(node: dict[str, Any], defs: dict[str, Any], seen: tuple[str, ...]) -> dict[str, Any]:
    ref = node["$ref"]
    name = str(ref).rsplit("/", 1)[-1]
    if name in seen:
        # A self-referencing model would inline forever; an untyped object is
        # the most Gemini can be told about it.
        logger.warning("recursive $ref %r left untyped for gemini", ref)
        return {"type": "object"}
    target = defs.get(name)
    if not isinstance(target, dict):
        raise LLMError(f"unresolvable $ref {ref!r} in schema")
    # Siblings of the $ref win: Pydantic puts the field's description there.
    merged = {**target, **{key: value for key, value in node.items() if key != "$ref"}}
    return _convert(merged, defs, (*seen, name))


def _collapse_any_of(node: dict[str, Any], defs: dict[str, Any], seen: tuple[str, ...]) -> dict[str, Any]:
    branches = [branch for branch in node["anyOf"] if isinstance(branch, dict)]
    concrete = [branch for branch in branches if branch.get("type") != "null"]
    nullable = len(concrete) != len(branches)
    siblings = {key: value for key, value in node.items() if key != "anyOf"}

    if not concrete:
        chosen: dict[str, Any] = {"type": "string"}
    else:
        chosen = concrete[0]
        if len(concrete) > 1:
            # Gemini has no union. Narrowing loses a branch; saying so in the
            # log beats a 400 nobody can reproduce locally.
            logger.warning("narrowing a %d-branch anyOf to its first branch for gemini", len(concrete))

    collapsed = _convert({**chosen, **siblings}, defs, seen)
    if nullable and isinstance(collapsed, dict):
        collapsed["nullable"] = True
    return collapsed


# --- JSON recovery -------------------------------------------------------

_FENCE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)```", re.DOTALL)


def _first_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        whole = json.loads(stripped)
    except ValueError:
        pass
    else:
        return whole if isinstance(whole, dict) else None

    start = stripped.find("{")
    while start != -1:
        end = _matching_brace(stripped, start)
        if end is not None:
            try:
                value = json.loads(stripped[start : end + 1])
            except ValueError:
                pass
            else:
                if isinstance(value, dict):
                    return value
        start = stripped.find("{", start + 1)
    return None


def _matching_brace(text: str, start: int) -> int | None:
    """Index of the `}` closing the `{` at `start`, ignoring braces inside strings."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def _excerpt(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[:limit]}..."
