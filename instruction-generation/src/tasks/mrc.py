"""Task 3: Machine reading comprehension (extractive QA over claims)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm import chat_json
from patents import PatentText
from prompt_fit import (
    OversizedPromptError,
    append_oversized_record,
    fit_teacher_prompt,
    prompt_budget_from_config,
)

from .base import Task
from .evol_pool import load_or_build_pool, sample_instruction

SYSTEM_PROMPT = "You are a patent attorney analyzing claims for infringement."

# Teacher instruction (everything except Claims). Claims injected by prompt_fit.
QA_INSTRUCTION = (
    "Read the following claims. Generate one highly specific, technical question "
    "regarding a numerical limit, chemical composition, or structural dependency "
    "found *explicitly* in the text. Then, provide the exact, concise answer. "
    "Format your response as a JSON object with 'question' and 'answer' keys."
)

# Instruction diversification (same pattern as abstract drafting / IPC reasoning pools).
INSTRUCTION_POOL_PROMPT = (
    "I am building an instruction-tuning dataset for extractive reading "
    "comprehension over patent claims. Generate 5 diverse, professional "
    "instructions that ask a model to answer a question using only the provided "
    "claims text, without outside knowledge or speculation. Vary length, tone, "
    "and framing (e.g. brief directive, examiner checklist, attorney note). "
    "Do not include a specific question, claim text, or invented patent facts. "
    "Output only a JSON list of strings."
)


def format_mrc_input(*, question: str, claims: str) -> str:
    return f"Question: {question.strip()}\n\nClaims:\n{claims.strip()}"


class MRCTask(Task):
    task_id = "mrc"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pool: list[str] = []

    def setup(self) -> None:
        pools_dir = Path(self.pools_dir) if self.pools_dir else Path(".")
        self._pool = load_or_build_pool(
            client=self.client,
            pools_dir=pools_dir,
            pool_name="mrc",
            user_prompt=INSTRUCTION_POOL_PROMPT,
            pool_size=int(self.evol_cfg.get("pool_size", 40)),
            batch_size=int(self.evol_cfg.get("batch_size", 5)),
        )

    def generate(self, patent: PatentText) -> dict[str, Any] | None:
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
        try:
            fitted = fit_teacher_prompt(
                system=SYSTEM_PROMPT,
                instruction=QA_INSTRUCTION,
                claims=patent.claims,
                trailer="",
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

        payload = chat_json(self.client, fitted.messages, expect=dict)
        question = str(payload.get("question") or "").strip()
        answer = str(payload.get("answer") or "").strip()
        if not question or not answer:
            return None

        return self._record(
            patent,
            instruction=sft_instruction,
            input_text=format_mrc_input(
                question=question,
                # Persist the claims actually sent to the teacher when trimmed.
                claims=fitted.claims_text_sent,
            ),
            output_text=answer,
            **self._fit_meta(fitted),
        )
