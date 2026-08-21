#!/usr/bin/env python3
"""CLI: generate synthetic instruction-tuning JSONL for one or all tasks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from assemble import (  # noqa: E402
    ShardWriter,
    append_done_id,
    load_done_ids,
    task_output_dir,
)
from holdings import (  # noqa: E402
    generator_dir,
    legacy_task_dirs,
    model_slug,
    pools_dir,
    write_manifest,
)
from ipc_lookup import IPCLookup  # noqa: E402
from llm import LLMClient, llm_config_from_dict  # noqa: E402
from patents import iter_patent_texts  # noqa: E402
from tasks import TASKS  # noqa: E402

DEFAULT_CONFIG = (
    REPO_ROOT / "instruction-generation" / "config" / "instruction_generation.yaml"
)


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
            "Generate Alpaca-style instruction JSONL from patent_search_clean "
            "+ IPC catalog (local Llama or OpenRouter)."
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config (default: {DEFAULT_CONFIG})",
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--task",
        choices=sorted(TASKS.keys()),
        help="Run a single task",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all tasks sequentially",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max successful records to write this run (skips do not count)",
    )
    p.add_argument(
        "--provider",
        choices=("local", "openrouter"),
        default=None,
        help="Override llm.provider",
    )
    p.add_argument("--model", default=None, help="Override llm.model")
    p.add_argument("--base-url", default=None, help="Override llm.base_url")
    p.add_argument(
        "--patents-dir",
        type=Path,
        default=None,
        help="Override paths.patents_dir",
    )
    p.add_argument(
        "--ipc-jsonl",
        type=Path,
        default=None,
        help="Override paths.ipc_jsonl",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override generator dir (default: "
            "{paths.output_dir}/{model_slug})"
        ),
    )
    return p


def run_task(
    task_id: str,
    *,
    cfg: dict[str, Any],
    client: LLMClient,
    ipc_lookup: IPCLookup,
    patents_dir: Path,
    dest: Path,
    pools: Path,
    limit: int | None,
) -> dict[str, Any]:
    evol_cfg = cfg.get("evol_instruct") or {}
    run_cfg = cfg.get("run") or {}
    shard_size = int(run_cfg.get("shard_size") or 100)

    task_cls = TASKS[task_id]
    task = task_cls(
        client,
        ipc_lookup=ipc_lookup,
        evol_cfg=evol_cfg,
        pools_dir=pools,
    )
    print(f"[{task_id}] setup…", flush=True)
    task.setup()

    task_dir = task_output_dir(dest, task_id)
    done = load_done_ids(task_dir)
    writer = ShardWriter(task_dir, shard_size=shard_size)

    n_ok = 0
    n_skip = 0
    n_err = 0
    # Iterate without a hard patent cap; --limit means successful writes.
    for patent in iter_patent_texts(patents_dir, limit=None, skip_ids=done):
        if limit is not None and n_ok >= limit:
            break
        try:
            record = task.generate(patent)
        except Exception as exc:  # noqa: BLE001 — continue batch on per-patent failures
            n_err += 1
            print(
                f"[{task_id}] error {patent.application_number}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue
        if record is None:
            n_skip += 1
            continue
        writer.add(record)
        append_done_id(task_dir, patent.application_number)
        n_ok += 1
        if n_ok % 10 == 0:
            print(f"[{task_id}] wrote {n_ok}…", flush=True)

    flushed = writer.flush()
    stats = {
        "task": task_id,
        "written": n_ok,
        "skipped": n_skip,
        "errors": n_err,
        "shards": [str(p) for p in writer.written_paths],
        "last_flush": str(flushed) if flushed else None,
    }
    print(
        f"[{task_id}] done: written={n_ok} skipped={n_skip} errors={n_err}",
        flush=True,
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = _resolve(args.config) if not args.config.is_absolute() else args.config
    if not cfg_path.is_file():
        print(f"error: config not found: {cfg_path}", file=sys.stderr)
        return 1
    cfg = load_config(cfg_path)

    paths = cfg.get("paths") or {}
    patents_dir = _resolve(
        args.patents_dir
        or Path(paths.get("patents_dir") or "data/derived/patent_search_clean")
    )
    ipc_jsonl = _resolve(
        args.ipc_jsonl
        or Path(paths.get("ipc_jsonl") or "data/ipc-codes/ipc_codes_20260101.jsonl")
    )
    holdings_root = _resolve(
        Path(paths.get("output_dir") or "data/derived/instruction_generation")
    )
    leftover = legacy_task_dirs(holdings_root)
    if leftover:
        print(
            f"error: legacy task dirs still at {holdings_root}: "
            f"{', '.join(leftover)}. Move them under "
            f"{holdings_root}/<model-slug>/.",
            file=sys.stderr,
        )
        return 1

    if not patents_dir.is_dir():
        print(f"error: patents dir missing: {patents_dir}", file=sys.stderr)
        return 1
    if not ipc_jsonl.is_file():
        print(f"error: IPC JSONL missing: {ipc_jsonl}", file=sys.stderr)
        return 1

    llm_cfg = llm_config_from_dict(
        cfg.get("llm") or {},
        overrides={
            "provider": args.provider,
            "model": args.model,
            "base_url": args.base_url,
        },
    )
    dest = (
        _resolve(args.output_dir)
        if args.output_dir is not None
        else generator_dir(holdings_root, llm_cfg.model)
    )
    shared_pools = pools_dir(holdings_root)
    write_manifest(
        dest,
        {
            "slug": model_slug(llm_cfg.model),
            "provider": llm_cfg.provider,
            "model": llm_cfg.model,
            "base_url": llm_cfg.base_url,
            "temperature": llm_cfg.temperature,
            "max_tokens": llm_cfg.max_tokens,
        },
    )
    client = LLMClient(llm_cfg)
    print(
        f"LLM provider={llm_cfg.provider} model={llm_cfg.model} "
        f"base_url={llm_cfg.base_url}",
        flush=True,
    )
    print(f"Generator dir: {dest}", flush=True)
    print(f"Shared pools: {shared_pools}", flush=True)

    print(f"Loading IPC catalog: {ipc_jsonl}", flush=True)
    ipc_lookup = IPCLookup.from_jsonl(ipc_jsonl)
    print(f"IPC entries: {len(ipc_lookup)}", flush=True)

    run_cfg = cfg.get("run") or {}
    limit = args.limit if args.limit is not None else run_cfg.get("limit")
    if limit is not None:
        limit = int(limit)

    task_ids = sorted(TASKS.keys()) if args.all else [args.task]
    all_stats = []
    for task_id in task_ids:
        stats = run_task(
            task_id,
            cfg=cfg,
            client=client,
            ipc_lookup=ipc_lookup,
            patents_dir=patents_dir,
            dest=dest,
            pools=shared_pools,
            limit=limit,
        )
        all_stats.append(stats)

    print(f"Finished {len(all_stats)} task(s) → {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
