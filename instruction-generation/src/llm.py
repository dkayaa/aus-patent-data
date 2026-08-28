"""OpenAI-compatible chat client (local Llama server or OpenRouter)."""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key_env: str
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout_s: float = 120.0
    max_retries: int = 3
    json_object: bool = False


def llm_config_from_dict(raw: dict[str, Any], *, overrides: dict[str, Any] | None = None) -> LLMConfig:
    """Build LLM config.

    When ``provider`` is switched via CLI (e.g. YAML ``local`` → ``--provider
    openrouter``), do not keep the YAML local ``base_url`` / ``api_key_env`` —
    fall back to that provider's defaults unless those fields were also
    overridden explicitly.
    """
    ov = {k: v for k, v in (overrides or {}).items() if v is not None}
    raw_provider = str(raw.get("provider") or "local").strip().lower()
    provider = str(ov.get("provider") or raw_provider).strip().lower()
    if provider != "openrouter":
        provider = "local"

    if provider == "openrouter":
        default_base = OPENROUTER_BASE_URL
        default_key_env = OPENROUTER_API_KEY_ENV
        default_model = "anthropic/claude-sonnet-4.6"
    else:
        default_base = "http://127.0.0.1:11434/v1"
        default_key_env = "OPENAI_API_KEY"
        default_model = "llama3.1:8b"

    provider_unchanged = raw_provider == provider
    if "base_url" in ov:
        base_url = str(ov["base_url"])
    elif provider_unchanged and raw.get("base_url"):
        base_url = str(raw["base_url"])
    else:
        base_url = default_base

    if "api_key_env" in ov:
        api_key_env = str(ov["api_key_env"])
    elif provider_unchanged and raw.get("api_key_env"):
        api_key_env = str(raw["api_key_env"])
    else:
        api_key_env = default_key_env

    model = str(ov.get("model") or raw.get("model") or default_model)
    # If switching provider and model still looks like the other side's default, replace.
    if not ov.get("model") and not provider_unchanged:
        model = default_model

    merged = dict(raw)
    merged.update(ov)
    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url.rstrip("/"),
        api_key_env=api_key_env,
        temperature=float(merged.get("temperature", 0.7)),
        max_tokens=int(merged.get("max_tokens", 1024)),
        timeout_s=float(merged.get("timeout_s", 120)),
        max_retries=int(merged.get("max_retries", 3)),
        json_object=bool(merged.get("json_object", False)),
    )


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._local = threading.local()

    def _openai(self) -> OpenAI:
        client = getattr(self._local, "client", None)
        if client is None:
            api_key = os.environ.get(self.config.api_key_env) or "none"
            client = OpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
                timeout=self.config.timeout_s,
                max_retries=self.config.max_retries,
            )
            self._local.client = client
        return client

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,  # type: ignore[arg-type]
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        if self.config.json_object:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._openai().chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        content = choice.content if choice else None
        if not content:
            raise RuntimeError("LLM returned empty content")
        return content.strip()


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def extract_json_value(text: str) -> Any:
    """Parse JSON from model output, tolerating markdown fences / leading prose."""
    cleaned = text.strip()
    fence = _FENCE_RE.search(cleaned)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    for opener, closer in (("[", "]"), ("{", "}")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from LLM output: {text[:240]!r}")


def chat_json(
    client: LLMClient,
    messages: list[dict[str, str]],
    *,
    expect: type | tuple[type, ...] = (dict, list),
    retries: int = 2,
) -> Any:
    """Call chat and parse JSON, with short repair retries."""
    last_err: Exception | None = None
    convo = list(messages)
    for attempt in range(retries + 1):
        raw = client.chat(convo)
        try:
            value = extract_json_value(raw)
            if not isinstance(value, expect):
                raise TypeError(
                    f"Expected {expect}, got {type(value).__name__}"
                )
            return value
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            last_err = exc
            convo = list(messages) + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON matching the "
                        "requested schema. Reply again with JSON only."
                    ),
                },
            ]
    raise RuntimeError(f"Failed to obtain valid JSON from LLM: {last_err}")
