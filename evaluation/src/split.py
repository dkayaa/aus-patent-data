#!/usr/bin/env python3
"""Freeze train/val/test splits and few-shot exemplars from Mode 1 passed/."""

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
    TASKS,
    load_config,
    resolve_path,
)
from holdings import resolve_generator_dir  # noqa: E402
from io_local import iter_task_records, write_jsonl_gz
from jsonl_gz import iter_jsonl_gz_shards, iter_records  # noqa: E402


def parse_filed_date(raw: str) -> date | None:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def load_filed_dates(patents_dir: Path, needed: set[str]) -> dict[str, date | None]:
    found: dict[str, date | None] = {}
    remaining = set(needed)
    if not remaining:
        return found
    for shard in iter_jsonl_gz_shards(patents_dir):
        for rec in iter_records(shard):
            app = str(rec.get("application_number") or "").strip()
            if not app or app not in remaining:
                continue
            found[app] = parse_filed_date(str(rec.get("filedDate") or ""))
            remaining.discard(app)
            if not remaining:
                return found
    for app in remaining:
        found[app] = None
    return found


def truncate_input(text: str, limit: int) -> str:
    if limit < 1 or len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Freeze eval train/val/test splits from Mode 1 passed rows."
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--generator",
        default=None,
        help="Generator model id or slug (required if several exist)",
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

    passed_root = resolve_path(Path(paths.get("passed_root") or "data/derived/instruction_generation_validation"))
    patents_dir = resolve_path(Path(paths.get("patents_dir") or "data/derived/patent_search_clean"))
    output_root = resolve_path(Path(paths.get("output_root") or "data/derived/evaluation"))
    seed = int(split_cfg.get("seed") or 42)
    cutoff = parse_filed_date(str(split_cfg.get("test_min_filed_date") or "2024-01-01"))
    if cutoff is None:
        print("error: invalid test_min_filed_date", file=sys.stderr)
        return 1
    test_fraction = float(split_cfg.get("test_fraction") or 0.10)
    train_val_ratio = float(split_cfg.get("train_val_ratio") or 0.80)
    k = int(few_cfg.get("k") or 3)
    exemplar_chars = int(few_cfg.get("exemplar_input_chars") or 4000)

    try:
        gen_dir = resolve_generator_dir(passed_root, generator=args.generator)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    gen_slug = gen_dir.name
    print(f"Generator: {gen_slug}", flush=True)

    by_task: dict[str, list[dict[str, Any]]] = {}
    all_apps: set[str] = set()
    for task_id in TASKS:
        passed = gen_dir / task_id / "passed"
        if not passed.is_dir():
            print(f"error: Mode 1 passed dir missing: {passed}", file=sys.stderr)
            return 1
        rows = list(iter_task_records(passed))
        by_task[task_id] = rows
        for rec in rows:
            app = str(rec.get("application_number") or "").strip()
            if app:
                all_apps.add(app)
        print(f"[{task_id}] {len(rows)} Mode 1 passed", flush=True)

    print(f"Loading filedDate from {patents_dir}…", flush=True)
    dates = load_filed_dates(patents_dir, all_apps)
    n_missing = sum(1 for app in all_apps if dates.get(app) is None)
    test_pool = sorted(
        app for app in all_apps if (dates.get(app) is not None and dates[app] >= cutoff)
    )
    rng = random.Random(seed)
    n_all = len(all_apps)
    n_target = max(1, int(round(test_fraction * n_all))) if n_all else 0
    if len(test_pool) <= n_target:
        test_ids = set(test_pool)
    else:
        test_ids = set(rng.sample(test_pool, n_target))
    rest = sorted(app for app in all_apps if app not in test_ids)
    rng.shuffle(rest)
    n_train = int(round(train_val_ratio * len(rest)))
    train_ids = set(rest[:n_train])
    val_ids = set(rest[n_train:])

    split_of = {}
    for app in train_ids:
        split_of[app] = "train"
    for app in val_ids:
        split_of[app] = "val"
    for app in test_ids:
        split_of[app] = "test"

    dest = output_root / "splits" / gen_slug
    dest.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    exemplars: dict[str, list[dict[str, Any]]] = {}

    for task_id in TASKS:
        buckets: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        for rec in by_task[task_id]:
            app = str(rec.get("application_number") or "").strip()
            name = split_of.get(app)
            if name:
                buckets[name].append(rec)
        task_dir = dest / task_id
        for name, rows in buckets.items():
            write_jsonl_gz(task_dir / f"{name}.jsonl.gz", rows)
        counts[task_id] = {k: len(v) for k, v in buckets.items()}
        pool = list(buckets["train"])
        rng.shuffle(pool)
        chosen = pool[: max(0, k)]
        exemplars[task_id] = [
            {
                "application_number": str(rec.get("application_number") or ""),
                "instruction": str(rec.get("instruction") or ""),
                "input": truncate_input(str(rec.get("input") or ""), exemplar_chars),
                "output": str(rec.get("output") or ""),
            }
            for rec in chosen
        ]
        print(
            f"[{task_id}] train={counts[task_id]['train']} "
            f"val={counts[task_id]['val']} test={counts[task_id]['test']} "
            f"exemplars={len(exemplars[task_id])}",
            flush=True,
        )

    manifest = {
        "generator": gen_slug,
        "seed": seed,
        "test_min_filed_date": cutoff.isoformat(),
        "date_field": "filedDate",
        "patents_dir": str(patents_dir),
        "n_unique_apps": n_all,
        "n_missing_date": n_missing,
        "n_test_pool": len(test_pool),
        "n_test": len(test_ids),
        "test_sampled_from_pool": len(test_pool) > n_target if n_all else False,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "task_counts": counts,
        "fewshot_k": k,
        "k_effective": {task_id: len(exemplars.get(task_id) or []) for task_id in TASKS},
        "exemplar_input_chars": exemplar_chars,
    }
    (dest / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (dest / "exemplars.json").write_text(
        json.dumps(exemplars, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote splits → {dest}", flush=True)
    print(
        f"apps unique={n_all} missing_date={n_missing} "
        f"test_pool={len(test_pool)} test={len(test_ids)} "
        f"train={len(train_ids)} val={len(val_ids)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
