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
    "Write the rationale first, then choose the score from that rationale. "
    "Do not rewrite the example. Reply with JSON only."
)

_JSON_SCHEMA = (
    "Respond with a single JSON object only, no markdown:\n"
    '{"rationale": "<short explanation>", "score": <int 1-5>, '
    '"failure_tags": ["tag", ...]}'
)

FAILURE_TAGS: dict[str, frozenset[str]] = {
    "ipc_reasoning": frozenset(
        {
            "unfaithful_to_claims",
            "fabricated_definition",
            "boilerplate",
            "ignores_assigned_ipc",
            "invented_ipc_code",
            "corrupted_text",
            "other",
        }
    ),
    "abstract_drafting": frozenset(
        {
            "topic_mismatch",
            "fields_swapped",
            "corrupted_text",
            "verbatim_claim_copy",
            "instruction_mismatch",
            "other",
        }
    ),
    "mrc": frozenset(
        {
            "unanswerable_from_claims",
            "unsupported_answer",
            "hallucinated_entity",
            "speculation",
            "fields_swapped",
            "other",
        }
    ),
}

FORBIDDEN_TAGS: dict[str, frozenset[str]] = {
    "ipc_reasoning": frozenset(
        {
            "wrong_ipc",
            "obsolete_ipc_code",
            "better_ipc_exists",
            "suboptimal_ipc_selection",
            "classification_mismatch",
        }
    ),
}

_TAG_LISTS = {task: ", ".join(sorted(tags)) for task, tags in FAILURE_TAGS.items()}

_RUBRICS: dict[str, str] = {
    "ipc_reasoning": f"""Task: ipc_reasoning (IPC justification).
The Classification / meta.primary_ipc is the GOLD office label. Do NOT re-classify the patent or fail the example because a different IPC might fit better.
Grade only Justification quality relative to that fixed code.
meta.ipc_title and meta.definition_statement are the official WIPO catalog text for this code. Treat them as the definition source. fabricated_definition means the justification contradicts or invents a definition different from that catalog text.
High score (4-5): Classification equals primary_ipc; Justification maps claim/abstract subject matter to that assigned place; no invented IPC codes; definitions match the provided WIPO text; reasoning engages the claims (not empty/boilerplate).
Low score (1-2): Justification ignores or contradicts the claims; invented codes; definition contradicts the provided WIPO text; empty/boilerplate that does not support the assigned code; scrape corruption that makes the triple unusable.
Do NOT use failure tags wrong_ipc, better_ipc_exists, suboptimal_ipc_selection, obsolete_ipc_code, or classification_mismatch — those are out of scope when Classification matches primary_ipc.
Allowed failure_tags only: {_TAG_LISTS["ipc_reasoning"]}.
Do not include a pass field. Score 1-5 only; a score >= 4 means usable for SFT.""",
    "abstract_drafting": f"""Task: abstract_drafting (claims → official abstract).
The output is gold patent text. Judge pair bugs only, not abstract writing quality.
FAIL on: topic mismatch (abstract about a different invention), swapped/truncated/corrupted fields, instruction not asking for an abstract, or claim 1 pasted verbatim as the abstract (bad SFT target).
Do NOT fail brief or generic official abstracts that still match the invention. Do NOT score USPTO style, coverage of every dependent claim, or "insufficient detail."
High score (4-5): instruction asks to summarize/draft an abstract; input looks like claims; output is a matching abstract for those claims (same invention); not a verbatim claim-1 paste; no scrape/corruption junk that makes the triple unusable.
Low score (1-2): instruction/input/output misaligned, truncated/corrupted nonsense, abstract clearly about a different invention, or output is claim 1 copied as the abstract.
Allowed failure_tags only: {_TAG_LISTS["abstract_drafting"]}.
Do not include a pass field. Score 1-5 only; a score >= 4 means usable for SFT.""",
    "mrc": f"""Task: mrc (extractive QA over claims).
The instruction is a task directive (answer from claims only). The input embeds Question + Claims.
High score (4-5): the question is answerable from the claims alone; the answer is supported by an explicit span or clear paraphrase of claim text; no speculation outside the claims; fields are correctly arranged (directive vs question vs claims).
Low score (1-2): unanswerable from claims, hallucinated numbers/entities, answer contradicts claims, speculation, or question wrongly placed in instruction instead of input.
Allowed failure_tags only: {_TAG_LISTS["mrc"]}.
Do not include a pass field. Score 1-5 only; a score >= 4 means usable for SFT.""",
}

_WIPO_DEF_TRUNCATE = 4000
_IDENTITY_META_KEYS = frozenset({"model", "provider", "judge_model", "judge_provider"})


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n…[truncated]…"


def _normalize_tag(tag: str) -> str:
    return str(tag).strip().lower().replace(" ", "_").replace("-", "_")


def allowed_failure_tags(task: str) -> frozenset[str]:
    tags = FAILURE_TAGS.get(task)
    if tags is None:
        raise ValueError(f"No judge rubric for task: {task}")
    return tags


def build_judge_payload(
    record: dict[str, Any],
    *,
    truncate_chars: int,
    wipo: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fields shown to the judge (no generator model/provider identity)."""
    meta_in = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    meta_out: dict[str, Any] = {}
    for key in ("primary_ipc", "ipc_title", "has_definition_entry", "document_type"):
        if key in meta_in and meta_in[key] is not None and key not in _IDENTITY_META_KEYS:
            meta_out[key] = meta_in[key]

    if wipo:
        title = wipo.get("ipc_title")
        if title:
            meta_out["ipc_title"] = str(title)
        definition = wipo.get("definition_statement")
        if definition:
            meta_out["definition_statement"] = _truncate(str(definition), _WIPO_DEF_TRUNCATE)
            meta_out["definition_source"] = "wipo_catalog"

    return {
        "task": record.get("task"),
        "application_number": record.get("application_number"),
        "instruction": _truncate(str(record.get("instruction") or ""), truncate_chars),
        "input": _truncate(str(record.get("input") or ""), truncate_chars),
        "output": _truncate(str(record.get("output") or ""), truncate_chars),
        "meta": meta_out,
    }


def wipo_fields_for_record(
    record: dict[str, Any],
    ipc_lookup: Any | None,
) -> dict[str, Any] | None:
    """Catalog title + definition for the record's primary IPC, if lookup is available."""
    if ipc_lookup is None:
        return None
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    code = str(meta.get("primary_ipc") or "").strip()
    if not code:
        return None
    getter = getattr(ipc_lookup, "get", None)
    if getter is None:
        return None
    entry = getter(code)
    if entry is None:
        return None
    title = str(getattr(entry, "title", "") or "").strip()
    definition = str(
        getattr(entry, "definition_statement", None)
        or getattr(entry, "scheme_note", None)
        or ""
    ).strip()
    out: dict[str, Any] = {}
    if title:
        out["ipc_title"] = title
    if definition:
        out["definition_statement"] = definition
    return out or None


def build_judge_messages(
    record: dict[str, Any],
    *,
    truncate_chars: int = 12000,
    ipc_lookup: Any | None = None,
) -> list[dict[str, str]]:
    task = str(record.get("task") or "")
    rubric = _RUBRICS.get(task)
    if rubric is None:
        raise ValueError(f"No judge rubric for task: {task}")

    wipo = wipo_fields_for_record(record, ipc_lookup)
    payload = build_judge_payload(
        record, truncate_chars=truncate_chars, wipo=wipo
    )
    user = (
        f"{rubric}\n\n"
        f"{_JSON_SCHEMA}\n\n"
        f"Example to evaluate:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def normalize_failure_tags(tags: Any, *, task: str) -> list[str]:
    allowed = allowed_failure_tags(task)
    forbidden = FORBIDDEN_TAGS.get(task, frozenset())
    if not isinstance(tags, list):
        tags = [tags] if tags else []
    out: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        tag = _normalize_tag(raw)
        if not tag or tag in forbidden:
            continue
        if tag not in allowed:
            tag = "other"
        if tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
    return out


def normalize_judge_result(
    raw: dict[str, Any],
    *,
    task: str,
    pass_score_min: int = 4,
) -> dict[str, Any]:
    score = int(raw.get("score"))
    if score < 1 or score > 5:
        raise ValueError(f"score out of range: {score}")
    # pass is derived in code; ignore any model-supplied pass field
    passed = score >= pass_score_min
    rationale = str(raw.get("rationale") or "").strip()
    return {
        "score": score,
        "pass": passed,
        "rationale": rationale,
        "failure_tags": normalize_failure_tags(raw.get("failure_tags"), task=task),
    }
