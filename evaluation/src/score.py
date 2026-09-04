#!/usr/bin/env python3
"""Score baseline predictions against frozen gold (automatic metrics only)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_CONFIG,
    PROMPTINGS,
    TASKS,
    load_config,
    resolve_path,
)
from holdings import model_slug, resolve_generator_dir  # noqa: E402
from io_local import iter_task_records
from ipc_checks import (
    IPC_RE,
    find_ipc_mentions,
    normalize_ipc,
    parse_ipc_output,
    parse_ipc_symbol,
)
from lexical import answer_contained, rouge_l_f1, token_f1
from schema import parse_mrc_input, simple_tokenize
from semantic import SemanticScorer

HIERARCHY_LEVELS = ("section", "class", "subclass", "group", "subgroup")


def _mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _rate(n_true: int, n: int) -> float | None:
    if n <= 0:
        return None
    return n_true / n


def hierarchical_match(pred: str | None, gold: str | None) -> dict[str, bool]:
    empty = {level: False for level in HIERARCHY_LEVELS}
    if not pred or not gold:
        return empty
    p = parse_ipc_symbol(pred)
    g = parse_ipc_symbol(gold)
    if p is None or g is None:
        return empty
    section = p[0] == g[0]
    cls = section and p[1] == g[1]
    subclass = cls and p[2] == g[2]
    group = subclass and p[3] is not None and p[3] == g[3]
    subgroup = group and (p[4] or "") == (g[4] or "")
    return {
        "section": section,
        "class": cls,
        "subclass": subclass,
        "group": group,
        "subgroup": subgroup,
    }


def _empty(text: str) -> bool:
    return not (text or "").strip()


def score_ipc(
    rows: list[dict[str, Any]],
    scorer: SemanticScorer | None,
) -> dict[str, Any]:
    n = len(rows)
    n_empty = 0
    n_format = 0
    n_exact = 0
    n_lenient = 0
    hier_hits = {level: 0 for level in HIERARCHY_LEVELS}
    rouge_vals: list[float] = []
    just_pairs: list[tuple[str, str]] = []

    for rec in rows:
        pred = str(rec.get("output") or "")
        gold = str(rec.get("gold_output") or "")
        meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
        primary = normalize_ipc(str(meta.get("primary_ipc") or ""))
        if _empty(pred):
            n_empty += 1
            rouge_vals.append(0.0)
            continue
        code, body = parse_ipc_output(pred)
        _, gold_body = parse_ipc_output(gold)
        if code and IPC_RE.match(code):
            n_format += 1
        if code and primary and code == primary:
            n_exact += 1
        if primary and primary in find_ipc_mentions(pred):
            n_lenient += 1
        for level, hit in hierarchical_match(code, primary or None).items():
            if hit:
                hier_hits[level] += 1
        if body and gold_body:
            rouge_vals.append(rouge_l_f1(gold_body, body))
            if scorer is not None:
                just_pairs.append((gold_body, body))
        else:
            rouge_vals.append(0.0)

    cosine_vals: list[float] = []
    if scorer is not None and just_pairs:
        cosine_vals = scorer.cosine_pairs(
            [a for a, _ in just_pairs], [b for _, b in just_pairs]
        )

    return {
        "n": n,
        "empty_rate": _rate(n_empty, n),
        "headline": {
            "exact_code": _rate(n_exact, n),
            "lenient_exact_code": _rate(n_lenient, n),
        },
        "format_valid_rate": _rate(n_format, n),
        "hierarchical": {level: _rate(hier_hits[level], n) for level in HIERARCHY_LEVELS},
        "justification": {
            "rouge_l_f1": _mean(rouge_vals),
            "nomic_cosine": _mean(cosine_vals),
            "n_scored": len(rouge_vals),
        },
        "notes": (
            "IPC code is scored against office primary_ipc (unbiased). "
            "exact_code requires the Classification:/Justification: schema "
            "with code == primary_ipc. lenient_exact_code is a hit if that "
            "normalized symbol appears anywhere in the output (markdown / "
            "memos / spaced symbols like H04N 19/593). Format-valid is the "
            "schema parse rate, reported separately. Justification "
            "ROUGE/Nomic is vs Llama 3.3 seed gold — teacher self-agreement, "
            "not independent quality."
        ),
    }


def score_abstract(
    rows: list[dict[str, Any]],
    scorer: SemanticScorer | None,
) -> dict[str, Any]:
    n = len(rows)
    n_empty = 0
    rouge_vals: list[float] = []
    compression: list[float] = []
    gold_pairs: list[tuple[str, str]] = []
    claim_pairs: list[tuple[str, str]] = []

    for rec in rows:
        pred = str(rec.get("output") or "").strip()
        gold = str(rec.get("gold_output") or "").strip()
        claims = str(rec.get("input") or "").strip()
        if _empty(pred):
            n_empty += 1
            rouge_vals.append(0.0)
            continue
        rouge_vals.append(rouge_l_f1(gold, pred))
        if claims:
            compression.append(len(pred) / len(claims))
        if scorer is not None:
            if gold:
                gold_pairs.append((gold, pred))
            if claims:
                claim_pairs.append((claims, pred))

    gold_cos = (
        scorer.cosine_pairs([a for a, _ in gold_pairs], [b for _, b in gold_pairs])
        if scorer is not None and gold_pairs
        else []
    )
    claim_cos = (
        scorer.cosine_pairs([a for a, _ in claim_pairs], [b for _, b in claim_pairs])
        if scorer is not None and claim_pairs
        else []
    )
    return {
        "n": n,
        "empty_rate": _rate(n_empty, n),
        "headline": {"rouge_l_f1": _mean(rouge_vals)},
        "nomic_cosine_vs_gold": _mean(gold_cos),
        "nomic_cosine_vs_claims": _mean(claim_cos),
        "compression_chars": _mean(compression),
        "notes": (
            "Abstract is scored against the office gold abstract (real ceiling "
            "for teacher and GPT)."
        ),
    }


def score_mrc(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    n_empty = 0
    n_em = 0
    n_in_claims = 0
    f1_vals: list[float] = []

    for rec in rows:
        pred = str(rec.get("output") or "").strip()
        gold = str(rec.get("gold_output") or "").strip()
        _, claims = parse_mrc_input(str(rec.get("input") or ""))
        claims = claims or ""
        if _empty(pred):
            n_empty += 1
            f1_vals.append(0.0)
            continue
        f1_vals.append(token_f1(gold, pred))
        if simple_tokenize(gold) == simple_tokenize(pred):
            n_em += 1
        if answer_contained(pred, claims):
            n_in_claims += 1

    return {
        "n": n,
        "empty_rate": _rate(n_empty, n),
        "headline": {"token_f1": _mean(f1_vals)},
        "exact_match": _rate(n_em, n),
        "answer_in_claims": _rate(n_in_claims, n),
        "notes": (
            "MRC gold answers are Llama 3.3 seed outputs. Teacher vs gold is "
            "self-agreement; few-shot exemplars leak 3.3 wording/format."
        ),
    }


def load_rows(pred_dir: Path) -> list[dict[str, Any]]:
    if not pred_dir.is_dir():
        return []
    return list(iter_task_records(pred_dir))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Score OpenRouter baseline predictions.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--generator", default=None)
    p.add_argument("--system", default=None)
    p.add_argument(
        "--prompting",
        choices=("zeroshot", "fewshot", "fewshot_k3", "all"),
        default="all",
    )
    p.add_argument("--task", choices=TASKS, default=None)
    p.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Skip Nomic cosine (lexical metrics only).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = resolve_path(args.config)
    if not cfg_path.is_file():
        print(f"error: config not found: {cfg_path}", file=sys.stderr)
        return 1
    cfg = load_config(cfg_path)
    paths = cfg.get("paths") or {}
    sem_cfg = cfg.get("semantic") or {}
    passed_root = resolve_path(
        Path(paths.get("passed_root") or "data/derived/instruction_generation_validation")
    )
    output_root = resolve_path(Path(paths.get("output_root") or "data/derived/evaluation"))
    try:
        gen_dir = resolve_generator_dir(passed_root, generator=args.generator)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    gen_slug = gen_dir.name
    pred_root = output_root / "predictions"
    scores_root = output_root / "scores"
    if not pred_root.is_dir():
        print(f"error: no predictions at {pred_root}", file=sys.stderr)
        return 1

    specs = [s for s in (cfg.get("systems") or []) if isinstance(s, dict)]
    if args.system:
        needle = args.system.strip().lower()
        specs = [
            s
            for s in specs
            if needle
            in (
                str(s.get("id") or "").lower(),
                str(s.get("model") or "").lower(),
                model_slug(str(s.get("model") or "")),
            )
        ]
        if not specs:
            print(f"error: unknown system: {args.system}", file=sys.stderr)
            return 1
    prompting_arg = "fewshot_k3" if args.prompting == "fewshot" else args.prompting
    promptings = list(PROMPTINGS) if prompting_arg == "all" else [prompting_arg]
    tasks = [args.task] if args.task else list(TASKS)

    scorer: SemanticScorer | None = None
    need_semantic = (not args.skip_semantic) and any(
        t in ("ipc_reasoning", "abstract_drafting") for t in tasks
    )
    if need_semantic:
        print("Loading Nomic embedder…", flush=True)
        scorer = SemanticScorer(
            str(sem_cfg.get("model_name") or "nomic-ai/nomic-embed-text-v1.5"),
            max_seq_length=int(sem_cfg.get("max_seq_length") or 8192),
            batch_size=int(sem_cfg.get("batch_size") or 8),
            prefix_a=str(sem_cfg.get("prefix_a") or "search_document: "),
            prefix_b=str(sem_cfg.get("prefix_b") or "search_document: "),
            trust_remote_code=bool(sem_cfg.get("trust_remote_code", True)),
        )

    summary: dict[str, Any] = {
        "generator": gen_slug,
        "systems": {},
    }
    # Merge existing summary so partial --system/--task runs do not wipe others.
    existing_path = scores_root / "summary.json"
    if existing_path.is_file():
        try:
            loaded = json.loads(existing_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("systems"), dict):
                summary["systems"] = loaded["systems"]
        except json.JSONDecodeError:
            pass

    n_scored = 0
    for spec in specs:
        model = str(spec.get("model") or "")
        sys_slug = model_slug(model)
        sys_block = summary["systems"].setdefault(sys_slug, {})
        for prompting in promptings:
            prompt_block = sys_block.setdefault(prompting, {})
            for task_id in tasks:
                pred_dir = pred_root / sys_slug / prompting / task_id
                rows = load_rows(pred_dir)
                if not rows:
                    print(f"[{sys_slug} {prompting} {task_id}] no predictions; skip", flush=True)
                    continue
                if task_id == "ipc_reasoning":
                    report = score_ipc(rows, scorer)
                elif task_id == "abstract_drafting":
                    report = score_abstract(rows, scorer)
                else:
                    print(
                        f"[{sys_slug} {prompting} {task_id}] unknown task; skip",
                        flush=True,
                    )
                    continue
                report.update(
                    {
                        "system": sys_slug,
                        "model": model,
                        "role": str(spec.get("role") or ""),
                        "prompting": prompting,
                        "task": task_id,
                        "generator": gen_slug,
                    }
                )
                out = scores_root / sys_slug / prompting / task_id / "report.json"
                write_json(out, report)
                prompt_block[task_id] = {
                    "n": report.get("n"),
                    "empty_rate": report.get("empty_rate"),
                    "headline": report.get("headline"),
                    "report": str(out),
                }
                n_scored += 1
                headline = report.get("headline") or {}
                print(
                    f"[{sys_slug} {prompting} {task_id}] n={report.get('n')} "
                    f"headline={headline} → {out}",
                    flush=True,
                )

    write_json(scores_root / "summary.json", summary)
    print(f"Wrote {n_scored} reports → {scores_root / 'summary.json'}", flush=True)
    return 0 if n_scored else 1


if __name__ == "__main__":
    raise SystemExit(main())
