"""Task 1: IPC reasoning & classification justification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from patents import PatentText

from .base import Task
from .evol_pool import load_or_build_pool, sample_instruction

SYSTEM_PROMPT = "You are an expert Australian Patent Examiner."

# Teacher prompt: used only to synthesize the justification (not the SFT instruction).
JUSTIFICATION_USER_TEMPLATE = """The assigned IPC code is GOLD. Do not propose a different code.

Here is the official WIPO catalog text for that code, plus the patent abstract and claims.

{ipc_block}

Abstract:
{abstract}

Claims:
{claims}

Write a short technical justification (about 120–220 words) that maps this invention’s claimed subject matter onto the WIPO definition.

- Ground the mapping in concrete claim features (parts, steps, materials), not the IPC title. Quote or tightly paraphrase the catalog text; do not invent definitional scope.
- Use only the provided definition. Do not mention other IPC codes.
- Prose only: no Classification line, no headings, no bullet list. Do not follow a fixed outline. Opening, sentence count, and order may vary from example to example."""

# Instruction diversification (same pattern as abstract drafting pools).
# Pool wordings must match the teacher target: one paragraph, claim→definition
# mapping, no re-classification, no multi-section examiner memos.
INSTRUCTION_POOL_PROMPT = (
    "I am building an instruction-tuning dataset for patent IPC classification "
    "reasoning. Generate 5 diverse, professional instructions that ask a model to "
    "write a short paragraph justifying an already-assigned "
    "IPC code by mapping claim features to that classification place. "
    "Vary length, tone, and framing (examiner note, attorney file note, brief "
    "rationale, classification log). All variants must: treat the assigned code as "
    "given (do not ask to re-classify, pick a better code, or discuss adjacent "
    "places); ask for claim-feature to definition mapping; and must NOT request "
    "multi-section memoranda, hierarchical walk-downs, confidence scores, or "
    "labeled subsections. Do not mention specific IPC codes or invent patent facts. "
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
            force_rebuild=bool(self.evol_cfg.get("rebuild_pool", False)),
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
