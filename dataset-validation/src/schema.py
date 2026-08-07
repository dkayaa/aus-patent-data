"""Shared Alpaca schema checks."""

from __future__ import annotations

import re
from typing import Any

REQUIRED_KEYS = (
    "task",
    "application_number",
    "instruction",
    "input",
    "output",
    "meta",
)


def check_schema(record: dict[str, Any], *, expected_task: str | None = None) -> list[str]:
    failures: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in record:
            failures.append(f"missing_key:{key}")
    if failures:
        return failures

    for key in ("instruction", "input", "output", "application_number"):
        val = record.get(key)
        if not isinstance(val, str) or not val.strip():
            failures.append(f"empty:{key}")

    task = record.get("task")
    if expected_task is not None and task != expected_task:
        failures.append(f"task_mismatch:{task}")

    meta = record.get("meta")
    if not isinstance(meta, dict):
        failures.append("meta_not_object")

    return failures


def check_task_light(record: dict[str, Any]) -> list[str]:
    """Light structural checks beyond shared schema."""
    failures: list[str] = []
    task = str(record.get("task") or "")
    output = str(record.get("output") or "")
    instruction = str(record.get("instruction") or "")
    input_text = str(record.get("input") or "")

    if task == "abstract_drafting":
        if len(output) > max(len(input_text) * 2, 50_000):
            failures.append("output_too_long_vs_input")

    if task == "mrc":
        if "?" not in instruction:
            failures.append("mrc_instruction_not_question")
        if len(output.strip()) >= len(input_text.strip()):
            failures.append("mrc_answer_not_shorter_than_claims")

    return failures


def simple_tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text.strip().lower()) if t]
