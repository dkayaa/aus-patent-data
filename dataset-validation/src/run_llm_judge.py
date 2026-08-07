#!/usr/bin/env python3
"""CLI: Mode 2 LLM-as-a-judge over a sample of Mode 1 passed rows."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

IG_SRC = REPO_ROOT / "instruction-generation" / "src"
if str(IG_SRC) not in sys.path:
    sys.path.insert(0, str(IG_SRC))

from io_util import ShardWriter, iter_task_records  # noqa: E402
from judge_prompts import (  # noqa: E402
    TASKS,
    build_judge_messages,
    normalize_judge_result,
)
from judge_sample import append_done_id, load_done_ids, sample_records  # noqa: E402
from llm import LLMClient, chat_json, llm_config_from_dict  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "dataset-validation" / "config" / "llm_judge.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("llm_judge")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Mode 2 LLM-as-a-judge: sample Mode 1 passed rows and grade "
            "pointwise with a frontier OpenRouter model (not a full-corpus pass)."
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config (default: {DEFAULT_CONFIG})",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", choices=list(TASKS), help="Judge one task")
    group.add_argument("--all", action="store_true", help="Judge all tasks")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override sample_size per task (default from YAML)",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Override Mode 1 passed/ dir (implies single --task)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override llm_judge output dir (implies single --task)",
    )
    p.add_argument(
        "--provider",
        choices=("local", "openrouter"),
        default=None,
        help="Override LLM provider (default: openrouter from config)",
    )
    p.add_argument("--model", type=str, default=None, help="Override judge model")
    return p


def _next_shard_index(out_dir: Path) -> int:
    existing = sorted(out_dir.glob("part-*.jsonl.gz"))
    if not existing:
        return 0
    last = existing[-1].stem  # part-00000.jsonl
    # stem of part-00000.jsonl.gz via Path: name=part-00000.jsonl.gz, stem=part-00000.jsonl
    name = existing[-1].name
    # part-NNNNN.jsonl.gz
    try:
        num = int(name.split("-")[1].split(".")[0])
        return num + 1
    except (IndexError, ValueError):
        return len(existing)


class ResumableShardWriter(ShardWriter):
    """ShardWriter that continues numbering after existing shards."""

    def __init__(self, out_dir: Path, *, shard_size: int = 100) -> None:
        super().__init__(out_dir, shard_size=shard_size)
        self._index = _next_shard_index(out_dir)


def judge_task(
    task_id: str,
    *,
    input_dir: Path,
    output_dir: Path,
    client: LLMClient,
    sample_size: int,
    seed: int,
    pass_score_min: int,
    truncate_chars: int,
    shard_size: int,
) -> dict[str, Any]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Mode 1 passed dir missing: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output_dir)

    all_records = list(iter_task_records(input_dir))
    to_judge = sample_records(
        all_records,
        sample_size=sample_size,
        seed=seed,
        skip_ids=done_ids,
    )

    log.info(
        "%s: %d Mode1 passed, %d already judged, sampling %d new (target %d)",
        task_id,
        len(all_records),
        len(done_ids),
        len(to_judge),
        sample_size,
    )

    passed_writer = ResumableShardWriter(output_dir / "passed", shard_size=shard_size)
    rejected_writer = ResumableShardWriter(output_dir / "rejected", shard_size=shard_size)

    scores: list[int] = []
    n_pass = 0
    n_fail = 0
    n_errors = 0
    tag_counts: Counter[str] = Counter()

    # Reload previously judged scores from report if present (for aggregate)
    prev_report_path = output_dir / "report.json"
    prev: dict[str, Any] = {}
    if prev_report_path.is_file():
        try:
            prev = json.loads(prev_report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}

    for rec in to_judge:
        app = str(rec.get("application_number") or "").strip()
        try:
            messages = build_judge_messages(rec, truncate_chars=truncate_chars)
            raw = chat_json(client, messages, expect=dict)
            if not isinstance(raw, dict):
                raise ValueError(f"expected dict, got {type(raw)}")
            result = normalize_judge_result(raw, pass_score_min=pass_score_min)
        except Exception as exc:  # noqa: BLE001 — continue like generation
            n_errors += 1
            log.warning("%s %s: judge error: %s", task_id, app, exc)
            continue

        out = deepcopy(rec)
        meta = out.get("meta") if isinstance(out.get("meta"), dict) else {}
        meta = dict(meta)
        meta["llm_judge"] = {
            **result,
            "judge_model": client.config.model,
            "judge_provider": client.config.provider,
        }
        out["meta"] = meta

        scores.append(result["score"])
        for tag in result["failure_tags"]:
            tag_counts[tag] += 1

        if result["pass"]:
            n_pass += 1
            passed_writer.add(out)
        else:
            n_fail += 1
            rejected_writer.add(out)

        append_done_id(output_dir, app)

    passed_writer.flush()
    rejected_writer.flush()

    # Merge with previous aggregates for resume-friendly report
    prev_n = int(prev.get("n_judged") or 0)
    prev_scores = prev.get("_scores") or []
    if not isinstance(prev_scores, list):
        prev_scores = []
    all_scores = [int(s) for s in prev_scores] + scores
    prev_tags = prev.get("failure_tag_counts") or {}
    merged_tags: Counter[str] = Counter({str(k): int(v) for k, v in prev_tags.items()})
    merged_tags.update(tag_counts)

    n_judged = prev_n + n_pass + n_fail
    n_pass_total = int(prev.get("n_pass") or 0) + n_pass
    n_fail_total = int(prev.get("n_fail") or 0) + n_fail
    n_errors_total = int(prev.get("n_errors") or 0) + n_errors

    report = {
        "task": task_id,
        "n_mode1_passed": len(all_records),
        "sample_size_target": sample_size,
        "seed": seed,
        "n_judged": n_judged,
        "n_pass": n_pass_total,
        "n_fail": n_fail_total,
        "n_errors": n_errors_total,
        "n_this_run": n_pass + n_fail,
        "mean_score": float(statistics.mean(all_scores)) if all_scores else None,
        "pass_rate": (n_pass_total / n_judged) if n_judged else None,
        "pass_score_min": pass_score_min,
        "judge_model": client.config.model,
        "judge_provider": client.config.provider,
        "failure_tag_counts": dict(merged_tags.most_common()),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "note": "Sample-based Mode 2 judge; not a full-corpus pass.",
        "_scores": all_scores,
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.info(
        "%s: judged %d this run (total %d); mean=%.2f pass_rate=%s → %s",
        task_id,
        n_pass + n_fail,
        n_judged,
        report["mean_score"] or 0.0,
        f"{report['pass_rate']:.2%}" if report["pass_rate"] is not None else "n/a",
        report_path,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(_resolve(args.config) if not args.config.is_absolute() else args.config)

    paths = cfg.get("paths") or {}
    judge_cfg = cfg.get("judge") or {}
    llm_raw = cfg.get("llm") or {}

    input_root = _resolve(Path(paths.get("input_root", "data/interim/instruction_generation_validation")))
    output_root = _resolve(Path(paths.get("output_root", "data/interim/instruction_generation_validation")))

    sample_size = int(args.limit if args.limit is not None else judge_cfg.get("sample_size", 50))
    seed = int(judge_cfg.get("seed", 42))
    pass_score_min = int(judge_cfg.get("pass_score_min", 4))
    truncate_chars = int(judge_cfg.get("truncate_chars", 12000))
    shard_size = int(judge_cfg.get("shard_size", 50))

    overrides: dict[str, Any] = {}
    if args.provider:
        overrides["provider"] = args.provider
    if args.model:
        overrides["model"] = args.model
    llm_cfg = llm_config_from_dict(llm_raw, overrides=overrides)
    client = LLMClient(llm_cfg)
    log.info("Judge provider=%s model=%s", llm_cfg.provider, llm_cfg.model)

    if args.input_dir or args.output_dir:
        if args.all or not args.task:
            log.error("--input-dir/--output-dir require a single --task")
            return 2
        tasks = [args.task]
    elif args.all:
        tasks = list(TASKS)
    else:
        tasks = [args.task]

    reports: list[dict[str, Any]] = []
    for task_id in tasks:
        in_dir = (
            _resolve(args.input_dir)
            if args.input_dir
            else input_root / task_id / "passed"
        )
        out_dir = (
            _resolve(args.output_dir)
            if args.output_dir
            else output_root / task_id / "llm_judge"
        )
        try:
            reports.append(
                judge_task(
                    task_id,
                    input_dir=in_dir,
                    output_dir=out_dir,
                    client=client,
                    sample_size=sample_size,
                    seed=seed,
                    pass_score_min=pass_score_min,
                    truncate_chars=truncate_chars,
                    shard_size=shard_size,
                )
            )
        except FileNotFoundError as exc:
            log.error("%s", exc)
            return 1

    summary_path = output_root / "llm_judge_summary.json"
    summary = {
        "tasks": [
            {
                "task": r["task"],
                "n_judged": r["n_judged"],
                "mean_score": r["mean_score"],
                "pass_rate": r["pass_rate"],
                "n_errors": r["n_errors"],
            }
            for r in reports
        ],
        "note": "Sample-based Mode 2; see per-task llm_judge/report.json",
    }
    if len(reports) > 1 or args.all:
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        log.info("Wrote summary %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
