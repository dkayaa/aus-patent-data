"""Per-task text pairs, lexical/semantic scores, and soft floors."""

from __future__ import annotations

from typing import Any

from ipc_checks import check_ipc_reasoning, parse_ipc_output
from lexical import answer_contained, best_span_f1, rouge_l_f1
from schema import check_schema, check_task_light, parse_mrc_input, simple_tokenize
from semantic import SemanticScorer

TASKS = (
    "ipc_reasoning",
    "abstract_drafting",
    "mrc",
)


def text_pair(record: dict[str, Any]) -> tuple[str, str] | None:
    """Return (side_a, side_b) used for ROUGE/cosine, or None if unavailable."""
    task = str(record.get("task") or "")
    input_text = str(record.get("input") or "")
    output = str(record.get("output") or "")

    if task == "ipc_reasoning":
        _, body = parse_ipc_output(output)
        if not body:
            return None
        return input_text, body

    if task == "mrc":
        _, claims = parse_mrc_input(input_text)
        # Score answer against claims only (ignore the embedded question).
        return (claims or input_text), output

    if task == "abstract_drafting":
        return input_text, output

    return None


def length_features(input_text: str, output_text: str) -> dict[str, float | int]:
    n_in = len(simple_tokenize(input_text))
    n_out = len(simple_tokenize(output_text))
    ratio = float(n_out) / float(n_in) if n_in else 0.0
    return {
        "len_input_tokens": n_in,
        "len_output_tokens": n_out,
        "compression_ratio": ratio,
    }


def structural_failures(record: dict[str, Any], *, expected_task: str) -> list[str]:
    fails = check_schema(record, expected_task=expected_task)
    fails.extend(check_task_light(record))
    if expected_task == "ipc_reasoning":
        ipc_fails, _ = check_ipc_reasoning(record)
        fails.extend(ipc_fails)
    return fails


def score_record(
    record: dict[str, Any],
    *,
    semantic: SemanticScorer | None,
    floors: dict[str, float],
) -> dict[str, Any]:
    """Compute validation block (scores + failed_rules)."""
    task = str(record.get("task") or "")
    failed = structural_failures(record, expected_task=task)

    pair = text_pair(record)
    input_text = str(record.get("input") or "")
    output_text = str(record.get("output") or "")
    if task == "ipc_reasoning":
        _, body = parse_ipc_output(output_text)
        scored_out = body or ""
    else:
        scored_out = output_text

    metrics: dict[str, Any] = {
        **length_features(input_text, scored_out or output_text),
        "rouge_l_f1": None,
        "best_span_f1": None,
        "answer_contained": None,
        "semantic_cosine": None,
    }

    if pair is not None:
        side_a, side_b = pair
        if task == "mrc":
            metrics["answer_contained"] = answer_contained(side_b, side_a)
            metrics["best_span_f1"] = best_span_f1(side_a, side_b)
            metrics["rouge_l_f1"] = rouge_l_f1(side_a, side_b)
        else:
            metrics["rouge_l_f1"] = rouge_l_f1(side_a, side_b)
            if semantic is not None:
                metrics["semantic_cosine"] = semantic.cosine_pair(side_a, side_b)

    # Soft floors
    cos_min = float(floors.get("semantic_cosine_min", 0.15))
    rouge_min = float(floors.get("rouge_l_f1_min", 0.02))
    mrc_span_min = float(floors.get("mrc_best_span_f1_min", 0.5))

    cos = metrics["semantic_cosine"]
    if task != "mrc" and cos is not None and cos < cos_min:
        failed.append("semantic_cosine_below_floor")

    if task == "mrc":
        span_f1 = metrics["best_span_f1"]
        if span_f1 is not None and span_f1 < mrc_span_min:
            failed.append("mrc_best_span_f1_below_floor")
    else:
        rl = metrics["rouge_l_f1"]
        if rl is not None and rl < rouge_min:
            failed.append("rouge_l_below_floor")

    return {
        "passed": len(failed) == 0,
        "failed_rules": failed,
        "scores": metrics,
    }
