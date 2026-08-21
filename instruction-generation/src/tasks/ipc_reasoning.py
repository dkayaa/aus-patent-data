"""Task 1: IPC reasoning & classification justification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from patents import PatentText

from .base import Task
from .evol_pool import load_or_build_pool, sample_instruction

SYSTEM_PROMPT = "You are an expert Australian Patent Examiner."

# Teacher prompt: used only to synthesize the justification (not the SFT instruction).
JUSTIFICATION_USER_TEMPLATE = """Here is an Abstract, the Claims, and the assigned IPC Code with its WIPO definition. Write a 2-sentence technical justification explaining why this code is correct by mapping the claims to the definition.

{ipc_block}

Abstract:
{abstract}

Claims:
{claims}

Respond with the justification only (two sentences). Do not repeat the IPC code or use section headers."""

# Instruction diversification (same pattern as abstract drafting pools).
INSTRUCTION_POOL_PROMPT = (
    "I am building an instruction-tuning dataset for patent IPC classification "
    "reasoning. Generate 5 diverse, professional instructions that ask a model to "
    "justify why a patent's abstract and claims belong under an assigned IPC code, "
    "using technical mapping to the classification place. Vary length, tone, and "
    "framing (e.g. examiner memo, attorney note, brief rationale, classification "
    "review). Do not mention specific IPC codes or invent patent facts. "
    "Output only a JSON list of strings."
)


class IPCReasoningTask(Task):
    task_id = "ipc_reasoning"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pool: list[str] = []

    def setup(self) -> None:
        pools_dir = Path(self.pools_dir) if self.pools_dir else Path(".")
        self._pool = load_or_build_pool(
            client=self.client,
            pools_dir=pools_dir,
            pool_name="ipc_reasoning",
            user_prompt=INSTRUCTION_POOL_PROMPT,
            pool_size=int(self.evol_cfg.get("pool_size", 40)),
            batch_size=int(self.evol_cfg.get("batch_size", 5)),
        )

    def eligible(self, patent: PatentText) -> bool:
        if not patent.primary_ipc or self.ipc_lookup is None:
            return False
        entry = self.ipc_lookup.get(patent.primary_ipc)
        # Skip codes with only a short title / no WIPO definition text — the
        # generator otherwise invents grounding and produces low-quality justifications.
        return entry is not None and bool(entry.definition_statement)

    def generate(self, patent: PatentText) -> dict[str, Any] | None:
        if not self.eligible(patent):
            return None
        lookup = self.ipc_lookup
        if lookup is None:
            return None
        entry = lookup.get(patent.primary_ipc)
        if entry is None or not entry.definition_statement:
            return None

        if not self._pool:
            self.setup()
        instruction = sample_instruction(self._pool)

        user = JUSTIFICATION_USER_TEMPLATE.format(
            ipc_block=entry.grounding_text(),
            abstract=patent.abstract,
            claims=patent.claims_text,
        )
        justification = self.client.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ]
        )
        output = (
            f"Classification: {patent.primary_ipc}\n"
            f"Justification: {justification.strip()}"
        )
        input_text = f"Abstract:\n{patent.abstract}\n\nClaims:\n{patent.claims_text}"
        return self._record(
            patent,
            instruction=instruction,
            input_text=input_text,
            output_text=output,
            ipc_title=entry.title,
            has_definition_entry=entry.has_definition_entry,
        )
