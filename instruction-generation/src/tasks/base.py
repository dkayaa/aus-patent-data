"""Shared task interface for instruction generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from assemble import make_record
from ipc_lookup import IPCLookup
from llm import LLMClient
from patents import PatentText


class Task(ABC):
    task_id: str

    def __init__(
        self,
        client: LLMClient,
        *,
        ipc_lookup: IPCLookup | None = None,
        evol_cfg: dict[str, Any] | None = None,
        pools_dir: Any | None = None,
    ) -> None:
        self.client = client
        self.ipc_lookup = ipc_lookup
        self.evol_cfg = evol_cfg or {}
        self.pools_dir = pools_dir

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
