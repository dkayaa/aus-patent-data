"""OpenAI-compatible chat client (local Ollama native API or OpenRouter).

Local provider uses Ollama's native ``/api/chat`` rather than the OpenAI
``/v1`` shim: empirically (2026-08-20) ``extra_body.options.num_ctx`` on the
OpenAI path does not honour 8192 (effective context stayed 4096), while
native ``/api/chat`` with ``options.num_ctx`` reports the configured value
via ``/api/ps``.

Ollama truncation behaviour (scripts/probe_ollama_truncation.py, 2026-08-20):
when a prompt exceeds num_ctx, Ollama drops the BEGINNING and keeps the end
(most recent tokens). The prompt_fit module must guarantee we never send an
over-limit prompt.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

DEFAULT_NUM_CTX = 8192


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key_env: str
    temperature: float = 0.7
    max_tokens: int = 1024
    max_output_tokens: int = 1024
    num_ctx: int = DEFAULT_NUM_CTX
    safety_margin: int = 64
    repeat_instruction: bool = True
    tokenizer_id: str = "NousResearch/Meta-Llama-3-8B"
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
    if not ov.get("model") and not provider_unchanged:
        model = default_model

    merged = dict(raw)
    merged.update(ov)
    max_tokens = int(merged.get("max_tokens", 1024))
    max_output = int(merged.get("max_output_tokens", max_tokens))
    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url.rstrip("/"),
        api_key_env=api_key_env,
        temperature=float(merged.get("temperature", 0.7)),
        max_tokens=max_tokens,
        max_output_tokens=max_output,
        num_ctx=int(merged.get("num_ctx", DEFAULT_NUM_CTX)),
        safety_margin=int(merged.get("safety_margin", 64)),
        repeat_instruction=bool(merged.get("repeat_instruction", True)),
        tokenizer_id=str(merged.get("tokenizer_id") or "NousResearch/Meta-Llama-3-8B"),
        timeout_s=float(merged.get("timeout_s", 120)),
        max_retries=int(merged.get("max_retries", 3)),
        json_object=bool(merged.get("json_object", False)),
    )


def ollama_origin(base_url: str) -> str:
    origin = base_url.rstrip("/")
    if origin.endswith("/v1"):
        origin = origin[: -len("/v1")]
    return origin


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
        temp = self.config.temperature if temperature is None else temperature
        max_out = self.config.max_output_tokens if max_tokens is None else max_tokens

        if self.config.provider == "local":
            return self._chat_ollama_native(
                messages, temperature=temp, max_tokens=max_out
            )

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,  # type: ignore[arg-type]
            "temperature": temp,
            "max_tokens": max_out,
        }
        if self.config.json_object:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._openai().chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        content = choice.content if choice else None
        if not content:
            raise RuntimeError("LLM returned empty content")
        return content.strip()

    def _chat_ollama_native(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Native /api/chat — the only path that reliably applies num_ctx."""
        origin = ollama_origin(self.config.base_url)
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": int(self.config.num_ctx),
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{origin}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_err: Exception | None = None
        for _ in range(max(1, self.config.max_retries)):
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                content = (body.get("message") or {}).get("content")
                if not content:
                    raise RuntimeError(f"Ollama returned empty content: {body!r}")
                return str(content).strip()
            except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                last_err = exc
        raise RuntimeError(f"Ollama /api/chat failed: {last_err}")


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
