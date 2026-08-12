"""Per-task LLM-as-a-judge rubrics (pointwise, G-Eval-style)."""

from __future__ import annotations

import json
from typing import Any

TASKS = (
    "ipc_reasoning",
    "abstract_drafting",
    "mrc",
)

JUDGE_SYSTEM = (
    "You are an expert evaluator of patent instruction-tuning examples. "
    "Score the example using the task rubric. "
    "Do not rewrite the example. Reply with JSON only."
)

_JSON_SCHEMA = (
    'Respond with a single JSON object only, no markdown:\n'
    '{"score": <int 1-5>, "pass": <bool>, "rationale": "<short explanation>", '
    '"failure_tags": ["tag", ...]}'
)

_RUBRICS: dict[str, str] = {
    "ipc_reasoning": """Task: ipc_reasoning (IPC justification).
The Classification / meta.primary_ipc is GOLD office label. Do NOT re-classify the patent or fail the example because a different IPC might fit better.
Grade only the Justification quality relative to that fixed code.
High score (4-5): Classification equals primary_ipc; Justification maps claim/abstract subject matter to that assigned place; no invented IPC codes or fabricated WIPO definitions; reasoning engages the claims (not empty/boilerplate).
Low score (1-2): Classification ≠ primary_ipc; hallucinated codes/definitions; Justification ignores or contradicts the claims; empty/boilerplate that does not support the assigned code.
Do NOT use failure tags like wrong_ipc, better_ipc_exists, or suboptimal_ipc_selection when Classification matches primary_ipc — those are out of scope.
Set pass=true only if the example is usable for SFT (typically score >= 4).""",
    "abstract_drafting": """Task: abstract_drafting (claims → official abstract).
The output is gold patent text. Judge *triple coherence*, not abstract writing quality:
High score: instruction asks to summarize/draft an abstract; input looks like claims; output looks like a matching abstract for those claims (no obvious topic mismatch or swapped fields).
Low score: instruction/input/output misaligned, truncated nonsense, or abstract clearly about a different invention than the claims.
Set pass=true only if usable for SFT (typically score >= 4).""",
    "mrc": """Task: mrc (extractive QA over claims).
The instruction is a task directive (answer from claims only). The input embeds Question + Claims.
High score: the question is answerable from the claims alone; the answer is supported by an explicit span or clear paraphrase of claim text; no speculation outside the claims; fields are correctly arranged (directive vs question vs claims).
Low score: unanswerable from claims, hallucinated numbers/entities, answer contradicts claims, or question wrongly placed in instruction instead of input.
Set pass=true only if usable for SFT (typically score >= 4).""",
}


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n…[truncated]…"


def build_judge_payload(
    record: dict[str, Any],
    *,
    truncate_chars: int,
) -> dict[str, Any]:
    """Fields shown to the judge (no generator model/provider identity)."""
    meta_in = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    meta_out: dict[str, Any] = {}
    for key in ("primary_ipc", "ipc_title", "has_definition_entry", "document_type"):
        if key in meta_in and meta_in[key] is not None:
            meta_out[key] = meta_in[key]

    return {
        "task": record.get("task"),
        "application_number": record.get("application_number"),
        "instruction": _truncate(str(record.get("instruction") or ""), truncate_chars),
        "input": _truncate(str(record.get("input") or ""), truncate_chars),
        "output": _truncate(str(record.get("output") or ""), truncate_chars),
        "meta": meta_out,
    }


def build_judge_messages(
    record: dict[str, Any],
    *,
    truncate_chars: int = 12000,
) -> list[dict[str, str]]:
    task = str(record.get("task") or "")
    rubric = _RUBRICS.get(task)
    if rubric is None:
        raise ValueError(f"No judge rubric for task: {task}")

    payload = build_judge_payload(record, truncate_chars=truncate_chars)
    user = (
        f"{rubric}\n\n"
        f"{_JSON_SCHEMA}\n\n"
        f"Example to evaluate:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def normalize_judge_result(
    raw: dict[str, Any],
    *,
    pass_score_min: int = 4,
) -> dict[str, Any]:
    score = int(raw.get("score"))
    if score < 1 or score > 5:
        raise ValueError(f"score out of range: {score}")
    if "pass" in raw and isinstance(raw["pass"], bool):
        passed = raw["pass"]
    else:
        passed = score >= pass_score_min
    tags = raw.get("failure_tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    rationale = str(raw.get("rationale") or "").strip()
    return {
        "score": score,
        "pass": passed,
        "rationale": rationale,
        "failure_tags": [str(t) for t in tags],
    }
