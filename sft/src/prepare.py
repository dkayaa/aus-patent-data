"""Prepare flat SFT JSONL from frozen evaluation splits (no new split logic)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common import (
    DATASET_TO_SPLIT_TASK,
    DEFAULT_CONFIG,
    DEFAULT_GENERATOR,
    SFT_DATASETS,
    SPLITS,
    load_config,
    resolve_path,
)
from holdings import resolve_generator_dir
from ipc_checks import normalize_ipc, parse_ipc_output
from io_local import load_split_records
from io_util import write_jsonl_gz


def classification_only_output(output: str) -> str | None:
    """Return ``Classification: <code>`` or None if unparseable."""
    code, _body = parse_ipc_output(output or "")
    if code:
        return f"Classification: {code}"
    for line in (output or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("classification:"):
            rest = stripped.split(":", 1)[1].strip()
            if rest:
                return f"Classification: {normalize_ipc(rest)}"
    return None


def transform_record(
    rec: dict[str, Any],
    *,
    dataset: str,
) -> dict[str, Any] | None:
    """Copy Alpaca fields; rewrite output for classification_only."""
    instruction = str(rec.get("instruction") or "")
    input_text = str(rec.get("input") or "")
    output = str(rec.get("output") or "")
    app = str(rec.get("application_number") or "").strip()
    if not app:
        return None

    if dataset == "ipc_reasoning_classification_only":
        only = classification_only_output(output)
        if only is None:
            return None
        output = only

    meta = rec.get("meta")
    meta_out: dict[str, Any] = dict(meta) if isinstance(meta, dict) else {}
    if dataset.startswith("ipc_reasoning") and "primary_ipc" not in meta_out:
        code, _ = parse_ipc_output(str(rec.get("output") or ""))
        if code:
            meta_out["primary_ipc"] = code

    return {
        "dataset": dataset,
        "task": dataset,
        "application_number": app,
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "meta": meta_out,
    }


def prepare_dataset(
    *,
    split_task_dir: Path,
    dest_dir: Path,
    dataset: str,
    limit: int | None,
    split_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    skipped = 0
    for split in SPLITS:
        rows_in = load_split_records(split_task_dir, split)
        out: list[dict[str, Any]] = []
        for rec in rows_in:
            transformed = transform_record(rec, dataset=dataset)
            if transformed is None:
                skipped += 1
                continue
            out.append(transformed)
        if split == "train" and limit is not None and limit >= 0:
            out = out[:limit]
        write_jsonl_gz(dest_dir / f"{split}.jsonl.gz", out)
        counts[split] = len(out)

    manifest: dict[str, Any] = {
        "dataset": dataset,
        "source_split_task": DATASET_TO_SPLIT_TASK[dataset],
        "source_split_dir": str(split_task_dir),
        "counts": counts,
        "skipped_unparseable": skipped,
        "train_limit": limit,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if split_manifest:
        for key in (
            "seed",
            "test_min_filed_date",
            "n_train",
            "n_val",
            "n_test",
            "generator",
        ):
            if key in split_manifest:
                manifest[f"split_{key}"] = split_manifest[key]
    (dest_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Prepare flat SFT JSONL from frozen evaluation/splits "
            "(inherits temporal + application_number holdout; no new split)."
        )
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--generator",
        default=None,
        help=(
            "Generator model id or slug (default: config generator, "
            f"currently {DEFAULT_GENERATOR})"
        ),
    )
    p.add_argument(
        "--dataset",
        default="all",
        help=(
            "One of: "
            + ", ".join(SFT_DATASETS)
            + ", or all (default)"
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap train rows per dataset (smoke); val/test unchanged",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override paths.output_root/{generator_slug}",
    )
    p.add_argument(
        "--splits-root",
        type=Path,
        default=None,
        help="Override paths.splits_root",
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
    generator = (
        args.generator
        or str(cfg.get("generator") or "").strip()
        or DEFAULT_GENERATOR
    )

    splits_root = resolve_path(
        args.splits_root
        or Path(paths.get("splits_root") or "data/derived/evaluation/splits")
    )
    output_root = resolve_path(
        Path(paths.get("output_root") or "data/derived/sft")
    )

    try:
        gen_dir = resolve_generator_dir(splits_root, generator=generator)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            "hint: run scripts/split_eval_data.py --generator … first",
            file=sys.stderr,
        )
        return 1

    gen_slug = gen_dir.name
    split_manifest_path = gen_dir / "split_manifest.json"
    split_manifest: dict[str, Any] | None = None
    if split_manifest_path.is_file():
        loaded = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            split_manifest = loaded

    dest_root = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else (output_root / gen_slug)
    )

    ds_arg = (args.dataset or "all").strip()
    if ds_arg == "all":
        datasets = list(SFT_DATASETS)
    elif ds_arg in SFT_DATASETS:
        datasets = [ds_arg]
    else:
        print(
            f"error: unknown --dataset {ds_arg!r}; "
            f"choose from {', '.join(SFT_DATASETS)}, all",
            file=sys.stderr,
        )
        return 1

    print(f"Generator splits: {gen_dir}", flush=True)
    print(f"SFT output: {dest_root}", flush=True)

    for dataset in datasets:
        split_task = DATASET_TO_SPLIT_TASK[dataset]
        split_task_dir = gen_dir / split_task
        if not split_task_dir.is_dir():
            print(
                f"error: missing eval split dir {split_task_dir}. "
                f"Run: scripts/split_eval_data.py --generator {gen_slug}",
                file=sys.stderr,
            )
            return 1
        # Sanity: require at least train.jsonl.gz
        if not (split_task_dir / "train.jsonl.gz").is_file():
            print(
                f"error: missing {split_task_dir / 'train.jsonl.gz'}",
                file=sys.stderr,
            )
            return 1
        dest = dest_root / dataset
        man = prepare_dataset(
            split_task_dir=split_task_dir,
            dest_dir=dest,
            dataset=dataset,
            limit=args.limit,
            split_manifest=split_manifest,
        )
        print(
            f"[{dataset}] train={man['counts']['train']} "
            f"val={man['counts']['val']} test={man['counts']['test']} "
            f"skipped={man['skipped_unparseable']} → {dest}",
            flush=True,
        )

    print(f"Done. Prepared {len(datasets)} dataset(s) under {dest_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
