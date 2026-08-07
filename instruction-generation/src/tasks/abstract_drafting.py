"""Task 2: Abstract drafting (claims → abstract) with Evol-Instruct pool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from patents import PatentText

from .base import Task
from .evol_pool import load_or_build_pool, sample_instruction

EVOL_PROMPT = (
    "I am building an instruction-tuning dataset. Generate 5 diverse, professional "
    "instructions asking a patent attorney to summarize a set of claims into an "
    "abstract. Vary the length and tone (e.g., 'Draft an abstract...', "
    "'Distill the following claims...', 'Provide a technical summary...'). "
    "Output only a JSON list of strings."
)


class AbstractDraftingTask(Task):
    task_id = "abstract_drafting"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pool: list[str] = []

    def setup(self) -> None:
        pools_dir = Path(self.pools_dir) if self.pools_dir else Path(".")
        self._pool = load_or_build_pool(
            client=self.client,
            pools_dir=pools_dir,
            pool_name="abstract_drafting",
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
            input_text=patent.claims_text,
            output_text=patent.abstract,
        )
