#!/usr/bin/env python3
"""Freeze train/val/test splits from a task run's shards.

Source shards live at ``data/derived/{task}/{run_name}/part-*.jsonl[.gz]``.
The filing date is ``meta.filed_date``. Every row with the same application
number is assigned to the same split.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_CONFIG,
    load_config,
    resolve_path,
)
from io_local import iter_task_records, write_jsonl_gz


def parse_filed_date(raw: str) -> date | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def record_meta(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("meta")
    return meta if isinstance(meta, dict) else {}


def record_application_number(record: dict[str, Any]) -> str:
    meta = record_meta(record)
    return str(
        meta.get("application_number") or record.get("application_number") or ""
    ).strip()


def record_filed_date(record: dict[str, Any]) -> date | None:
    meta = record_meta(record)
    raw = meta.get("filed_date")
    if raw is None:
        raw = meta.get("filedDate")
    return parse_filed_date("" if raw is None else str(raw))


def application_dates(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, date | None], int, int]:
    """One date per application number.

    Returns ``(dates, n_missing_application, n_date_conflicts)``. A later row
    can fill a missing date. A disagreeing date is counted and ignored so the
    application still has a single split.
    """
    dates: dict[str, date | None] = {}
    missing_app = 0
    conflicts = 0
    for record in rows:
        app = record_application_number(record)
        if not app:
            missing_app += 1
            continue
        found = record_filed_date(record)
        if app not in dates:
            dates[app] = found
            continue
        current = dates[app]
        if current is None and found is not None:
            dates[app] = found
        elif current is not None and found is not None and found != current:
            conflicts += 1
    return dates, missing_app, conflicts


def assign_splits(
    dates: dict[str, date | None],
    *,
    cutoff: date,
    test_fraction: float,
    train_val_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Assign each application number to train, val, or test.

    Test is drawn only from filings on or after ``cutoff``. Undated filings
    stay in the train/val remainder.
    """
    apps = set(dates)
    test_pool = sorted(
        app for app, filed in dates.items() if filed is not None and filed >= cutoff
    )
    rng = random.Random(seed)
    n_all = len(apps)
    n_target = max(1, int(round(test_fraction * n_all))) if n_all else 0
    if len(test_pool) <= n_target:
        test_ids = set(test_pool)
    else:
        test_ids = set(rng.sample(test_pool, n_target))
    rest = sorted(app for app in apps if app not in test_ids)
    rng.shuffle(rest)
    n_train = int(round(train_val_ratio * len(rest)))
    train_ids = set(rest[:n_train])
    val_ids = set(rest[n_train:])

    split_of: dict[str, str] = {}
    for app in train_ids:
        split_of[app] = "train"
    for app in val_ids:
        split_of[app] = "val"
    for app in test_ids:
        split_of[app] = "test"
    return split_of


def truncate_input(text: str, limit: int) -> str:
    if limit < 1 or len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def shard_dir(derived_root: Path, task: str, run_name: str) -> Path:
    return derived_root / task / run_name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Freeze train/val/test splits from shards in "
            "data/derived/{task}/{run_name}/."
        )
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--task", required=True, help="Task folder under data/derived/")
    p.add_argument("--run", required=True, help="Run folder under the task")
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Shard directory (default: data/derived/{task}/{run}/)",
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
    split_cfg = cfg.get("split") or {}
    few_cfg = cfg.get("fewshot") or {}

    task = str(args.task).strip()
    run_name = str(args.run).strip()
    if not task or not run_name:
        print("error: --task and --run are required", file=sys.stderr)
        return 1

    derived_root = resolve_path(Path(paths.get("derived_root") or "data/derived"))
    output_root = resolve_path(Path(paths.get("output_root") or "data/derived/evaluation"))
    source = (
        resolve_path(args.input)
        if args.input is not None
        else shard_dir(derived_root, task, run_name)
    )
    if not source.is_dir():
        print(f"error: shard dir missing: {source}", file=sys.stderr)
        return 1

    seed = int(split_cfg.get("seed") or 42)
    cutoff = parse_filed_date(str(split_cfg.get("test_min_filed_date") or "2024-01-01"))
    if cutoff is None:
        print("error: invalid test_min_filed_date", file=sys.stderr)
        return 1
    test_fraction = float(split_cfg.get("test_fraction") or 0.10)
    train_val_ratio = float(split_cfg.get("train_val_ratio") or 0.80)
    k = int(few_cfg.get("k") or 3)
    exemplar_chars = int(few_cfg.get("exemplar_input_chars") or 4000)

    rows = list(iter_task_records(source))
    dates, n_missing_app, n_conflicts = application_dates(rows)
    n_missing_date = sum(1 for filed in dates.values() if filed is None)
    split_of = assign_splits(
        dates,
        cutoff=cutoff,
        test_fraction=test_fraction,
        train_val_ratio=train_val_ratio,
        seed=seed,
    )
    print(
        f"task={task} run={run_name} shards={source} rows={len(rows)}",
        flush=True,
    )

    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for record in rows:
        app = record_application_number(record)
        name = split_of.get(app)
        if name:
            buckets[name].append(record)

    dest = output_root / "splits" / run_name / task
    dest.mkdir(parents=True, exist_ok=True)
    for name, split_rows in buckets.items():
        write_jsonl_gz(dest / f"{name}.jsonl.gz", split_rows)
    counts = {name: len(split_rows) for name, split_rows in buckets.items()}

    rng = random.Random(seed)
    pool = list(buckets["train"])
    rng.shuffle(pool)
    exemplars = [
        {
            "application_number": record_application_number(record),
            "instruction": str(record.get("instruction") or ""),
            "input": truncate_input(str(record.get("input") or ""), exemplar_chars),
            "output": str(record.get("output") or ""),
        }
        for record in pool[: max(0, k)]
    ]

    n_test = sum(1 for name in split_of.values() if name == "test")
    n_train = sum(1 for name in split_of.values() if name == "train")
    n_val = sum(1 for name in split_of.values() if name == "val")
    n_test_pool = sum(
        1
        for filed in dates.values()
        if filed is not None and filed >= cutoff
    )
    manifest = {
        "task": task,
        "run": run_name,
        "source_dir": str(source),
        "seed": seed,
        "test_min_filed_date": cutoff.isoformat(),
        "date_field": "meta.filed_date",
        "n_rows": len(rows),
        "n_unique_apps": len(dates),
        "n_missing_application_number": n_missing_app,
        "n_missing_date": n_missing_date,
        "n_date_conflicts": n_conflicts,
        "n_test_pool": n_test_pool,
        "n_test": n_test,
        "test_sampled_from_pool": n_test_pool > max(1, int(round(test_fraction * len(dates))))
        if dates
        else False,
        "n_train": n_train,
        "n_val": n_val,
        "counts": counts,
        "fewshot_k": k,
        "k_effective": len(exemplars),
        "exemplar_input_chars": exemplar_chars,
    }
    (dest / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "splits" / run_name / "exemplars.json").write_text(
        json.dumps({task: exemplars}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote splits → {dest}", flush=True)
    print(
        f"apps unique={len(dates)} missing_date={n_missing_date} "
        f"missing_app={n_missing_app} test_pool={n_test_pool} "
        f"test={n_test} train={n_train} val={n_val} "
        f"rows train={counts['train']} val={counts['val']} test={counts['test']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
