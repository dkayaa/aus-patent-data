#!/usr/bin/env python3
"""CLI: Mode 1 programmatic validation with lexical + semantic scores."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parents[1]  # …/dataset-validation/src → repo root
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from io_util import ShardWriter, iter_task_records  # noqa: E402
from semantic import SemanticScorer  # noqa: E402
from task_metrics import TASKS, score_record  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "dataset-validation" / "config" / "programmatic.yaml"


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
            "Programmatic validation of instruction JSONL: schema/IPC + "
            "ROUGE-L / token-F1 + MiniLM cosine."
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config (default: {DEFAULT_CONFIG})",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", choices=list(TASKS), help="Validate one task")
    group.add_argument("--all", action="store_true", help="Validate all tasks")
    p.add_argument("--limit", type=int, default=None, help="Max records per task")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Override task input dir (implies single --task)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override task output dir (implies single --task)",
    )
    p.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Skip MiniLM cosine (lexical + structural only)",
    )
    return p


def _mean(vals: list[float]) -> float | None:
    return float(statistics.mean(vals)) if vals else None


def validate_task(
    task_id: str,
    *,
    input_dir: Path,
    output_dir: Path,
    floors: dict[str, float],
    semantic: SemanticScorer | None,
    shard_size: int,
    limit: int | None,
) -> dict[str, Any]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir missing: {input_dir}")

    passed_writer = ShardWriter(output_dir / "passed", shard_size=shard_size)
    rejected_writer = ShardWriter(output_dir / "rejected", shard_size=shard_size)

    fail_counts: Counter[str] = Counter()
    rouge_vals: list[float] = []
    cos_vals: list[float] = []
    token_f1_vals: list[float] = []
    n_total = 0
    n_pass = 0
    n_reject = 0

    for record in iter_task_records(input_dir):
        if limit is not None and n_total >= limit:
            break
        # Ensure task field matches folder when missing/wrong for metrics routing
        if not record.get("task"):
            record = {**record, "task": task_id}
        n_total += 1

        validation = score_record(record, semantic=semantic, floors=floors)
        meta = dict(record.get("meta") or {})
        meta["validation"] = validation
        out_rec = {**record, "meta": meta}

        scores = validation.get("scores") or {}
        if scores.get("rouge_l_f1") is not None:
            rouge_vals.append(float(scores["rouge_l_f1"]))
        if scores.get("semantic_cosine") is not None:
            cos_vals.append(float(scores["semantic_cosine"]))
        if scores.get("token_f1") is not None:
            token_f1_vals.append(float(scores["token_f1"]))

        if validation["passed"]:
            passed_writer.add(out_rec)
            n_pass += 1
        else:
            for rule in validation["failed_rules"]:
                fail_counts[rule] += 1
            rejected_writer.add(out_rec)
            n_reject += 1

        if n_total % 50 == 0:
            print(f"[{task_id}] processed {n_total}…", flush=True)

    passed_writer.flush()
    rejected_writer.flush()

    report = {
        "task": task_id,
        "n_total": n_total,
        "n_passed": n_pass,
        "n_rejected": n_reject,
        "fail_rule_counts": dict(fail_counts),
        "mean_rouge_l_f1": _mean(rouge_vals),
        "mean_semantic_cosine": _mean(cos_vals),
        "mean_token_f1": _mean(token_f1_vals),
        "passed_shards": [str(p) for p in passed_writer.paths],
        "rejected_shards": [str(p) for p in rejected_writer.paths],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(
        f"[{task_id}] done: total={n_total} passed={n_pass} rejected={n_reject} "
        f"→ {report_path}",
        flush=True,
    )
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = _resolve(args.config) if not args.config.is_absolute() else args.config
    if not cfg_path.is_file():
        print(f"error: config not found: {cfg_path}", file=sys.stderr)
        return 1
    cfg = load_config(cfg_path)

    paths = cfg.get("paths") or {}
    input_root = _resolve(Path(paths.get("input_root") or "data/derived/instruction_generation"))
    output_root = _resolve(
        Path(paths.get("output_root") or "data/derived/instruction_generation_validation")
    )
    run_cfg = cfg.get("run") or {}
    shard_size = int(run_cfg.get("shard_size") or 100)
    limit = args.limit if args.limit is not None else run_cfg.get("limit")
    if limit is not None:
        limit = int(limit)

    floors = {
        "semantic_cosine_min": float((cfg.get("floors") or {}).get("semantic_cosine_min", 0.15)),
        "rouge_l_f1_min": float((cfg.get("floors") or {}).get("rouge_l_f1_min", 0.02)),
        "mrc_token_f1_min": float((cfg.get("floors") or {}).get("mrc_token_f1_min", 0.1)),
    }

    semantic: SemanticScorer | None = None
    if not args.skip_semantic:
        sem_cfg = cfg.get("semantic") or {}
        print(
            f"Loading semantic model: {sem_cfg.get('model_name', 'all-MiniLM-L6-v2')}",
            flush=True,
        )
        semantic = SemanticScorer(
            str(sem_cfg.get("model_name") or "sentence-transformers/all-MiniLM-L6-v2"),
            max_seq_length=int(sem_cfg.get("max_seq_length") or 512),
            batch_size=int(sem_cfg.get("batch_size") or 32),
        )

    if args.input_dir is not None or args.output_dir is not None:
        if args.all or not args.task:
            print(
                "error: --input-dir/--output-dir require a single --task",
                file=sys.stderr,
            )
            return 1

    task_ids = list(TASKS) if args.all else [args.task]
    reports = []
    for task_id in task_ids:
        in_dir = (
            _resolve(args.input_dir)
            if args.input_dir is not None
            else input_root / task_id
        )
        out_dir = (
            _resolve(args.output_dir)
            if args.output_dir is not None
            else output_root / task_id
        )
        if not in_dir.is_dir():
            print(f"[{task_id}] skip: missing input {in_dir}", flush=True)
            continue
        reports.append(
            validate_task(
                task_id,
                input_dir=in_dir,
                output_dir=out_dir,
                floors=floors,
                semantic=semantic,
                shard_size=shard_size,
                limit=limit,
            )
        )

    if not reports:
        print("error: no tasks validated", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
