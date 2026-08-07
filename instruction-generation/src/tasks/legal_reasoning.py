"""Task 1: Legal reasoning & IPC classification justification."""

from __future__ import annotations

from typing import Any

from patents import PatentText

from .base import Task

SYSTEM_PROMPT = "You are an expert Australian Patent Examiner."

USER_TEMPLATE = """Here is an Abstract, the Claims, and the assigned IPC Code with its WIPO definition. Write a 2-sentence technical justification explaining why this code is correct by mapping the claims to the definition.

{ipc_block}

Abstract:
{abstract}

Claims:
{claims}

Respond with the justification only (two sentences)."""

INSTRUCTION = (
    "Given the patent abstract and claims, justify the assigned IPC classification "
    "with a brief technical explanation grounded in the WIPO definition."
)


class LegalReasoningTask(Task):
    task_id = "legal_reasoning"

    def generate(self, patent: PatentText) -> dict[str, Any] | None:
        if not patent.primary_ipc:
            return None
        if self.ipc_lookup is None:
            return None
        entry = self.ipc_lookup.get(patent.primary_ipc)
        # Skip codes with only a short title / no WIPO definition text — the
        # generator otherwise invents grounding and produces low-quality justifications.
        if entry is None or not entry.definition_statement:
            return None

        user = USER_TEMPLATE.format(
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
            instruction=INSTRUCTION,
            input_text=input_text,
            output_text=output,
            ipc_title=entry.title,
            has_definition_entry=entry.has_definition_entry,
        )
