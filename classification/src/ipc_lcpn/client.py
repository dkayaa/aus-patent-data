# Pick which model backend to use from config (local, openrouter, or hf).

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Protocol

from .config import IPCLCPNConfig, REPO_ROOT, resolve_path
from .hf_client import HFClient, HFClientConfig

# Reuse the teacher-stage chat client for local Ollama and OpenRouter.
_IG_SRC = REPO_ROOT / "instruction-generation" / "src"
if str(_IG_SRC) not in sys.path:
    sys.path.insert(0, str(_IG_SRC))

from llm import LLMClient, llm_config_from_dict  # noqa: E402


# Anything with a .chat() method that takes a list of messages and returns a string.
class ChatClient(Protocol):
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...


def build_client(cfg: IPCLCPNConfig) -> ChatClient:
    provider = cfg.provider
    if provider in {"local", "openrouter"}:
        return LLMClient(llm_config_from_dict(cfg.llm))

    if provider == "hf":
        hf = cfg.hf
        model = str(hf.get("model") or "meta-llama/Llama-3.1-8B-Instruct").strip()
        adapter_raw = hf.get("adapter_path")
        adapter: Path | None
        if adapter_raw is None or str(adapter_raw).strip() in {"", "null", "None"}:
            adapter = None
        else:
            adapter = resolve_path(str(adapter_raw))
        return HFClient(
            HFClientConfig(
                model=model,
                adapter_path=adapter,
                max_new_tokens=int(hf.get("max_new_tokens") or 512),
                temperature=float(hf.get("temperature") or 0.0),
            )
        )

    raise ValueError(f"unsupported llm.provider: {provider!r}")

# Short string for logs, e.g. local:llama3.1:8b.
def client_label(client: Any) -> str:
    if isinstance(client, HFClient):
        adapter = client.config.adapter_path
        suffix = f"+adapter:{adapter}" if adapter else ""
        return f"hf:{client.config.model}{suffix}"
    config = getattr(client, "config", None)
    if config is not None:
        return f"{getattr(config, 'provider', '?')}:{getattr(config, 'model', '?')}"
    return type(client).__name__
