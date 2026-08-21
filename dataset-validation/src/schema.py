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
    input_text = str(record.get("input") or "")

    if task == "abstract_drafting":
        if len(output) > max(len(input_text) * 2, 50_000):
            failures.append("output_too_long_vs_input")
        if not looks_like_claim_start(input_text):
            failures.append("abstract_input_not_claims")
        if looks_like_numbered_claims(output):
            failures.append("abstract_output_looks_like_claims")
        n_in = len(simple_tokenize(input_text))
        n_out = len(simple_tokenize(output))
        if n_in and n_out >= n_in:
            failures.append("abstract_not_shorter_than_claims")

    if task == "ipc_reasoning":
        abstract, claims = parse_ipc_input(input_text)
        if abstract is None or claims is None:
            failures.append("ipc_input_format_invalid")

    if task == "mrc":
        question, claims = parse_mrc_input(input_text)
        if question is None or claims is None:
            failures.append("mrc_input_format_invalid")
        else:
            if "?" not in question:
                failures.append("mrc_question_missing")
            if len(output.strip()) >= len(claims.strip()):
                failures.append("mrc_answer_not_shorter_than_claims")

    return failures


_MRC_INPUT_RE = re.compile(
    r"^Question:\s*(?P<question>.+?)\n\nClaims:\s*\n(?P<claims>.+)$",
    re.DOTALL | re.IGNORECASE,
)
_IPC_INPUT_RE = re.compile(
    r"^Abstract:\s*(?P<abstract>.+?)\n\nClaims:\s*\n?(?P<claims>.+)$",
    re.DOTALL | re.IGNORECASE,
)


def parse_mrc_input(input_text: str) -> tuple[str | None, str | None]:
    """Return (question, claims) from MRC ``input``, or (None, None)."""
    m = _MRC_INPUT_RE.match((input_text or "").strip())
    if not m:
        return None, None
    question = m.group("question").strip()
    claims = m.group("claims").strip()
    if not question or not claims:
        return None, None
    return question, claims


def parse_ipc_input(input_text: str) -> tuple[str | None, str | None]:
    """Return (abstract, claims) from ipc_reasoning ``input``, or (None, None)."""
    m = _IPC_INPUT_RE.match((input_text or "").strip())
    if not m:
        return None, None
    abstract = m.group("abstract").strip()
    claims = m.group("claims").strip()
    if not abstract or not claims:
        return None, None
    return abstract, claims


def simple_tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text.strip().lower()) if t]


_CLAIM_START_RE = re.compile(r"^\s*1\s*[\.\)]")
_CLAIM_SECOND_RE = re.compile(r"\n\s*2\s*[\.\)]")


def looks_like_claim_start(text: str) -> bool:
    """True if ``text`` begins like claim 1 (``1.`` / ``1)``)."""
    return bool(_CLAIM_START_RE.match(text or ""))


def looks_like_numbered_claims(text: str) -> bool:
    """True if ``text`` looks like a multi-claim list (swap detection)."""
    if not looks_like_claim_start(text):
        return False
    return bool(_CLAIM_SECOND_RE.search(text or ""))
