"""Shared task interface for instruction generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from assemble import make_record
from ipc_lookup import IPCLookup
from llm import LLMClient
from patents import PatentText
from prompt_fit import PromptBudget


class Task(ABC):
    task_id: str

    def __init__(
        self,
        client: LLMClient,
        *,
        ipc_lookup: IPCLookup | None = None,
        evol_cfg: dict[str, Any] | None = None,
        pools_dir: Any | None = None,
        prompt_budget: PromptBudget | None = None,
    ) -> None:
        self.client = client
        self.ipc_lookup = ipc_lookup
        self.evol_cfg = evol_cfg or {}
        self.pools_dir = pools_dir
        self.prompt_budget = prompt_budget

    def setup(self) -> None:
        """Optional one-time setup (e.g. Evol-Instruct instruction pool)."""

    def eligible(self, patent: PatentText) -> bool:
        """Cheap pre-filter so the worker pool is not filled with local skips."""
        return True

    @abstractmethod
    def generate(self, patent: PatentText) -> dict[str, Any] | None:
        """Return an Alpaca-style record, or None to skip."""

    def _meta(self, patent: PatentText, **extra: Any) -> dict[str, Any]:
        meta = {
            "primary_ipc": patent.primary_ipc or None,
            "document_type": patent.document_type or None,
            "model": self.client.config.model,
            "provider": self.client.config.provider,
        }
        meta.update(extra)
        return meta

    def _record(
        self,
        patent: PatentText,
        *,
        instruction: str,
        input_text: str,
        output_text: str,
        **meta_extra: Any,
    ) -> dict[str, Any]:
        return make_record(
            task=self.task_id,
            application_number=patent.application_number,
            instruction=instruction,
            input_text=input_text,
            output_text=output_text,
            meta=self._meta(patent, **meta_extra),
        )

    def _fit_meta(self, fitted: Any) -> dict[str, Any]:
        return {
            "claims_trimmed": bool(fitted.claims_trimmed),
            "claims_dropped": list(fitted.claims_dropped),
            "n_claims_original": int(fitted.n_claims_original),
            "n_claims_sent": int(fitted.n_claims_sent),
            "prompt_tokens": int(fitted.prompt_tokens),
            "input_budget": int(fitted.input_budget),
            "num_ctx": int(self.client.config.num_ctx),
            "max_output_tokens": int(self.client.config.max_output_tokens),
            "safety_margin": int(self.client.config.safety_margin),
            "repeat_instruction": bool(
                self.prompt_budget.repeat_instruction
                if self.prompt_budget
                else self.client.config.repeat_instruction
            ),
            "tokenizer_id": (
                self.prompt_budget.tokenizer_id
                if self.prompt_budget
                else self.client.config.tokenizer_id
            ),
        }
