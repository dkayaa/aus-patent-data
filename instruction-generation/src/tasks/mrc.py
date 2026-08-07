"""Task 4: Machine reading comprehension (extractive QA over claims)."""

from __future__ import annotations

from typing import Any

from llm import chat_json
from patents import PatentText

from .base import Task

SYSTEM_PROMPT = "You are a patent attorney analyzing claims for infringement."

USER_TEMPLATE = """Read the following claims. Generate one highly specific, technical question regarding a numerical limit, chemical composition, or structural dependency found *explicitly* in the text. Then, provide the exact, concise answer. Format your response as a JSON object with 'question' and 'answer' keys.

Claims:
{claims}"""


class MRCTask(Task):
    task_id = "mrc"

    def generate(self, patent: PatentText) -> dict[str, Any] | None:
        payload = chat_json(
            self.client,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(claims=patent.claims_text),
                },
            ],
            expect=dict,
        )
        question = str(payload.get("question") or "").strip()
        answer = str(payload.get("answer") or "").strip()
        if not question or not answer:
            return None
        return self._record(
            patent,
            instruction=question,
            input_text=patent.claims_text,
            output_text=answer,
        )
