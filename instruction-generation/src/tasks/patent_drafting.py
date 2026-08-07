"""Task 3: Patent drafting (abstract → claim 1) with Evol-Instruct pool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from patents import PatentText

from .base import Task
from .evol_pool import load_or_build_pool, sample_instruction

EVOL_PROMPT = (
    "Generate 5 diverse instructions asking a patent drafting assistant to write a "
    "first independent method/apparatus claim based on an abstract. "
    "(e.g., 'Draft claim 1...', 'Based on this abstract, write a legally robust "
    "independent claim...'). Output only a JSON list of strings."
)


class PatentDraftingTask(Task):
    task_id = "patent_drafting"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pool: list[str] = []

    def setup(self) -> None:
        pools_dir = Path(self.pools_dir) if self.pools_dir else Path(".")
        self._pool = load_or_build_pool(
            client=self.client,
            pools_dir=pools_dir,
            pool_name="patent_drafting",
            user_prompt=EVOL_PROMPT,
            pool_size=int(self.evol_cfg.get("pool_size", 40)),
            batch_size=int(self.evol_cfg.get("batch_size", 5)),
        )

    def generate(self, patent: PatentText) -> dict[str, Any] | None:
        if not self._pool:
            self.setup()
        instruction = sample_instruction(self._pool)
        return self._record(
            patent,
            instruction=instruction,
            input_text=patent.abstract,
            output_text=patent.claim_1,
        )
