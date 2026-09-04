"""Task 1: IPC reasoning & classification justification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from patents import PatentText
from prompt_fit import (
    OversizedPromptError,
    append_oversized_record,
    fit_teacher_prompt,
    prompt_budget_from_config,
)

from .base import Task
from .evol_pool import load_or_build_pool, sample_instruction

SYSTEM_PROMPT = "You are an expert Australian Patent Examiner."

# Teacher prompt: used only to synthesize the justification (not the SFT instruction).
# Claims are injected separately by prompt_fit so they can be trimmed to num_ctx.
JUSTIFICATION_INSTRUCTION = """The assigned IPC code is GOLD. Do not propose a different code.

Here is the official WIPO catalog text for that code, plus the patent abstract and claims.

{ipc_block}

Abstract:
{abstract}"""

JUSTIFICATION_TRAILER = """Write a short technical justification (about 120–220 words) that maps this invention’s claimed subject matter onto the WIPO definition.

Ground the mapping in concrete claim features (parts, steps, materials), not the IPC title. Quote or tightly paraphrase the catalog text; do not invent definitional scope. Use only the provided definition and do not mention other IPC codes.

Respond with the justification only, as free prose: no Classification line, headings, or bullet list. Do not follow a fixed outline. Opening, sentence count, and order may vary from example to example."""

# Instruction diversification (same pattern as abstract drafting pools).
# Student-facing task: assign an IPC code and justify it from abstract+claims.
# (Teacher generation still treats the office code as gold; that is separate.)
INSTRUCTION_POOL_PROMPT = (
    "I am building an instruction-tuning dataset for patent IPC classification "
    "reasoning. Generate 5 diverse, professional instructions that ask a model to "
    "assign an IPC code to a patent and write a short paragraph justifying that "
    "code by mapping claim features to the classification place. "
    "Vary length, tone, and framing (examiner note, attorney file note, brief "
    "rationale, classification log). All variants must: ask the model to both "
    "select/assign an IPC symbol and justify it; ask for claim-feature to "
    "definition mapping in the justification; and must NOT request "
    "multi-section memoranda, hierarchical walk-downs, confidence scores, "
    "labeled subsections, or comparisons of multiple alternative codes. "
    "Do not mention specific IPC codes or invent patent facts. "
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
        sft_instruction = sample_instruction(self._pool)

        budget = self.prompt_budget or prompt_budget_from_config(
            {
                "num_ctx": self.client.config.num_ctx,
                "max_output_tokens": self.client.config.max_output_tokens,
                "safety_margin": self.client.config.safety_margin,
                "repeat_instruction": self.client.config.repeat_instruction,
                "tokenizer_id": self.client.config.tokenizer_id,
            }
        )
        teacher_instruction = JUSTIFICATION_INSTRUCTION.format(
            ipc_block=entry.grounding_text(),
            abstract=patent.abstract,
        )
        try:
            fitted = fit_teacher_prompt(
                system=SYSTEM_PROMPT,
                instruction=teacher_instruction,
                claims=patent.claims,
                trailer=JUSTIFICATION_TRAILER,
                budget=budget,
                application_number=patent.application_number,
                task=self.task_id,
            )
        except OversizedPromptError as exc:
            append_oversized_record(
                application_number=exc.application_number,
                task=exc.task,
                prompt_tokens=exc.prompt_tokens,
                input_budget=exc.input_budget,
                n_claims_original=exc.n_claims_original,
            )
            return None

        justification = self.client.chat(fitted.messages)
        output = (
            f"Classification: {patent.primary_ipc}\n"
            f"Justification: {justification.strip()}"
        )
        # SFT input keeps the full (untrimmed) patent text; trim metadata is in meta.
        input_text = f"Abstract:\n{patent.abstract}\n\nClaims:\n{patent.claims_text}"
        return self._record(
            patent,
            instruction=sft_instruction,
            input_text=input_text,
            output_text=output,
            ipc_title=entry.title,
            has_definition_entry=entry.has_definition_entry,
            **self._fit_meta(fitted),
        )
