"""OpenAI-compatible chat client (local Llama server or OpenRouter)."""

from __future__ import annotations

import json
import os
import re
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


def llm_config_from_dict(raw: dict[str, Any], *, overrides: dict[str, Any] | None = None) -> LLMConfig:
    data = dict(raw)
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})

    provider = str(data.get("provider") or "local").strip().lower()
    if provider == "openrouter":
        base_url = str(data.get("base_url") or OPENROUTER_BASE_URL)
        api_key_env = str(data.get("api_key_env") or OPENROUTER_API_KEY_ENV)
    else:
        provider = "local"
        base_url = str(data.get("base_url") or "http://127.0.0.1:8080/v1")
        api_key_env = str(data.get("api_key_env") or "OPENAI_API_KEY")

    return LLMConfig(
        provider=provider,
        model=str(data.get("model") or "llama3.1-8b"),
        base_url=base_url.rstrip("/"),
        api_key_env=api_key_env,
        temperature=float(data.get("temperature", 0.7)),
        max_tokens=int(data.get("max_tokens", 1024)),
        timeout_s=float(data.get("timeout_s", 120)),
        max_retries=int(data.get("max_retries", 3)),
    )


class LLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        api_key = os.environ.get(config.api_key_env) or "none"
        self._client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout_s,
            max_retries=config.max_retries,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
        )
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
