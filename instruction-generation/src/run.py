#!/usr/bin/env python3
"""CLI: generate synthetic instruction-tuning JSONL for one or all tasks."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
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
from llm import LLMClient, llm_config_from_dict, ollama_origin  # noqa: E402
from patents import iter_patent_texts  # noqa: E402
from prompt_fit import (  # noqa: E402
    OversizedPromptError,
    assert_ollama_num_ctx,
    get_tokenizer,
    prompt_budget_from_config,
)
from tasks import TASKS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Empirically determined by scripts/probe_ollama_truncation.py (2026-08-20):
# Ollama drops the BEGINNING of prompts that exceed num_ctx.
OLLAMA_TRUNCATION_DROPS = "start"
OLLAMA_TRUNCATION_NOTE = (
    "When a prompt exceeds num_ctx, Ollama drops the BEGINNING "
    "(keeps the most recent / end tokens). Over-long prompts therefore "
    "lose the system prompt and instruction first. prompt_fit trims claims "
    "so we never send an over-limit prompt."
)

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
        "--workers",
        type=int,
        default=None,
        help=(
            "Concurrent LLM calls (default: run.workers in YAML, else 1). "
            "Use >1 for OpenRouter; keep 1 for a serial local server."
        ),
    )
    p.add_argument(
        "--provider",
        choices=("local", "openrouter"),
        default=None,
        help="Override llm.provider",
    )
    p.add_argument("--model", default=None, help="Override llm.model")
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override llm.temperature",
    )
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
    p.add_argument(
        "--rebuild-pool",
        action="store_true",
        help=(
            "Rebuild the Evol-Instruct phrasing pool for this --task "
            "(requires a single --task; overwrites _pools/<task>.json)."
        ),
    )
    p.add_argument(
        "--only-ids",
        type=Path,
        default=None,
        help=(
            "JSON/JSONL/txt list of application_numbers to restrict generation to. "
            "Already-written done_ids.txt are still skipped (use for stratified expand)."
        ),
    )
    p.add_argument(
        "--repeat-instruction",
        type=lambda s: str(s).lower() in ("1", "true", "yes", "on"),
        default=None,
        help="Override llm.repeat_instruction (true/false)",
    )
    return p


def _try_generate(task: Any, patent: Any) -> tuple[Any, dict[str, Any] | None, BaseException | None]:
    try:
        return patent, task.generate(patent), None
    except Exception as exc:  # noqa: BLE001 — continue batch on per-patent failures
        return patent, None, exc


def _write_ok(
    *,
    task_id: str,
    task_dir: Path,
    writer: ShardWriter,
    patent: Any,
    record: dict[str, Any],
    n_ok: int,
    append_done: bool,
) -> int:
    writer.add(record)
    if append_done:
        append_done_id(task_dir, patent.application_number)
    n_ok += 1
    if n_ok % 10 == 0:
        print(f"[{task_id}] wrote {n_ok}…", flush=True)
    return n_ok


def _load_only_ids(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8").strip()
    ids: set[str] = set()
    if path.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    ids.add(item.strip())
                elif isinstance(item, dict) and item.get("application_number"):
                    ids.add(str(item["application_number"]).strip())
        elif isinstance(data, dict):
            for item in data.get("ids") or data.get("application_numbers") or []:
                ids.add(str(item).strip())
        return {x for x in ids if x}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            row = json.loads(line)
            ids.add(str(row.get("application_number") or "").strip())
        else:
            ids.add(line)
    return {x for x in ids if x}


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
    workers: int,
    only_ids: set[str] | None,
    prompt_budget: Any,
) -> dict[str, Any]:
    evol_cfg = cfg.get("evol_instruct") or {}
    run_cfg = cfg.get("run") or {}
    shard_size = int(run_cfg.get("shard_size") or 100)
    workers = max(1, int(workers))

    task_cls = TASKS[task_id]
    task = task_cls(
        client,
        ipc_lookup=ipc_lookup,
        evol_cfg=evol_cfg,
        pools_dir=pools,
        prompt_budget=prompt_budget,
    )
    print(f"[{task_id}] setup…", flush=True)
    task.setup()

    task_dir = task_output_dir(dest, task_id)
    # Restrict universe with --only-ids; still skip/resume via done_ids.txt.
    done = load_done_ids(task_dir)
    writer = ShardWriter(task_dir, shard_size=shard_size)
    patents = iter_patent_texts(patents_dir, limit=None, skip_ids=done)
    if only_ids is not None:
        patents = (
            patent
            for patent in patents
            if patent.application_number in only_ids
        )

    if workers == 1:
        n_ok, n_skip, n_err, n_oversized, n_trimmed = _run_serial(
            task_id, task=task, patents=patents, task_dir=task_dir,
            writer=writer, limit=limit, append_done=True,
        )
    else:
        print(f"[{task_id}] workers={workers}", flush=True)
        n_ok, n_skip, n_err, n_oversized, n_trimmed = _run_parallel(
            task_id, task=task, patents=patents, task_dir=task_dir,
            writer=writer, limit=limit, workers=workers,
            append_done=True,
        )

    flushed = writer.flush()
    stats = {
        "task": task_id,
        "written": n_ok,
        "skipped": n_skip,
        "errors": n_err,
        "workers": workers,
        "oversized_skipped": n_oversized,
        "claims_trimmed": n_trimmed,
        "shards": [str(p) for p in writer.written_paths],
        "last_flush": str(flushed) if flushed else None,
    }
    print(
        f"[{task_id}] done: written={n_ok} skipped={n_skip} errors={n_err} "
        f"trimmed={n_trimmed} oversized={n_oversized}",
        flush=True,
    )
    return stats


def _run_serial(
    task_id: str,
    *,
    task: Any,
    patents: Any,
    task_dir: Path,
    writer: ShardWriter,
    limit: int | None,
    append_done: bool,
) -> tuple[int, int, int, int, int]:
    n_ok = 0
    n_skip = 0
    n_err = 0
    n_oversized = 0
    n_trimmed = 0
    for patent in patents:
        if limit is not None and n_ok >= limit:
            break
        if not task.eligible(patent):
            n_skip += 1
            continue
        patent, record, err = _try_generate(task, patent)
        if err is not None:
            if isinstance(err, OversizedPromptError):
                n_oversized += 1
                n_skip += 1
                logger.error("%s", err)
                continue
            n_err += 1
            print(
                f"[{task_id}] error {patent.application_number}: {err}",
                file=sys.stderr,
                flush=True,
            )
            continue
        if record is None:
            n_skip += 1
            continue
        if (record.get("meta") or {}).get("claims_trimmed"):
            n_trimmed += 1
        n_ok = _write_ok(
            task_id=task_id,
            task_dir=task_dir,
            writer=writer,
            patent=patent,
            record=record,
            n_ok=n_ok,
            append_done=append_done,
        )
    return n_ok, n_skip, n_err, n_oversized, n_trimmed


def _run_parallel(
    task_id: str,
    *,
    task: Any,
    patents: Any,
    task_dir: Path,
    writer: ShardWriter,
    limit: int | None,
    workers: int,
    append_done: bool,
) -> tuple[int, int, int, int, int]:
    n_ok = 0
    n_skip = 0
    n_err = 0
    n_oversized = 0
    n_trimmed = 0
    exhausted = False
    inflight: dict[Future[tuple[Any, dict[str, Any] | None, BaseException | None]], Any] = {}
    # Keep the pool fed. IPC skips locally; eligible() already dropped those.
    target_inflight = workers * 3

    def _submit(executor: ThreadPoolExecutor) -> bool:
        nonlocal exhausted, n_skip
        if exhausted:
            return False
        while True:
            try:
                patent = next(patents)
            except StopIteration:
                exhausted = True
                return False
            if not task.eligible(patent):
                n_skip += 1
                continue
            fut = executor.submit(_try_generate, task, patent)
            inflight[fut] = patent
            return True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            if limit is not None and n_ok >= limit:
                for fut in inflight:
                    fut.cancel()
                break
            while (
                not exhausted
                and len(inflight) < target_inflight
                and (limit is None or n_ok + len(inflight) < limit)
            ):
                if not _submit(executor):
                    break
            if not inflight:
                break
            finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in finished:
                patent = inflight.pop(fut)
                try:
                    patent, record, err = fut.result()
                except Exception as exc:  # noqa: BLE001
                    n_err += 1
                    print(
                        f"[{task_id}] error {patent.application_number}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                if err is not None:
                    if isinstance(err, OversizedPromptError):
                        n_oversized += 1
                        n_skip += 1
                        logger.error("%s", err)
                        continue
                    n_err += 1
                    print(
                        f"[{task_id}] error {patent.application_number}: {err}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                if record is None:
                    n_skip += 1
                    continue
                if limit is not None and n_ok >= limit:
                    continue
                if (record.get("meta") or {}).get("claims_trimmed"):
                    n_trimmed += 1
                n_ok = _write_ok(
                    task_id=task_id,
                    task_dir=task_dir,
                    writer=writer,
                    patent=patent,
                    record=record,
                    n_ok=n_ok,
                    append_done=append_done,
                )
    return n_ok, n_skip, n_err, n_oversized, n_trimmed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = _resolve(args.config) if not args.config.is_absolute() else args.config
    if not cfg_path.is_file():
        print(f"error: config not found: {cfg_path}", file=sys.stderr)
        return 1
    cfg = load_config(cfg_path)

    if getattr(args, "rebuild_pool", False):
        if args.all or not args.task:
            print("error: --rebuild-pool requires a single --task", file=sys.stderr)
            return 2
        evol = dict(cfg.get("evol_instruct") or {})
        evol["rebuild_pool"] = True
        cfg["evol_instruct"] = evol

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

    llm_raw = dict(cfg.get("llm") or {})
    if args.repeat_instruction is not None:
        llm_raw["repeat_instruction"] = bool(args.repeat_instruction)

    llm_cfg = llm_config_from_dict(
        llm_raw,
        overrides={
            "provider": args.provider,
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
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
    budget = prompt_budget_from_config(
        {
            "num_ctx": llm_cfg.num_ctx,
            "max_output_tokens": llm_cfg.max_output_tokens,
            "safety_margin": llm_cfg.safety_margin,
            "repeat_instruction": llm_cfg.repeat_instruction,
            "tokenizer_id": llm_cfg.tokenizer_id,
        }
    )

    print(
        f"LLM provider={llm_cfg.provider} model={llm_cfg.model} "
        f"temperature={llm_cfg.temperature} base_url={llm_cfg.base_url}",
        flush=True,
    )
    print(f"Generator dir: {dest}", flush=True)
    print(f"Shared pools: {shared_pools}", flush=True)
    print(
        f"Budget num_ctx={budget.num_ctx} max_output_tokens={budget.max_output_tokens} "
        f"safety_margin={budget.safety_margin} input_budget={budget.input_budget} "
        f"repeat_instruction={budget.repeat_instruction} "
        f"tokenizer_id={budget.tokenizer_id}",
        flush=True,
    )

    # Preload tokenizer once (documented HF stand-in for llama3.1:8b).
    get_tokenizer(budget.tokenizer_id)

    asserted_num_ctx: int | None = None
    if llm_cfg.provider == "local":
        def _warmup() -> None:
            client.chat(
                [{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0,
            )

        asserted_num_ctx = assert_ollama_num_ctx(
            model=llm_cfg.model,
            base_url=llm_cfg.base_url,
            expected_num_ctx=llm_cfg.num_ctx,
            chat_fn=_warmup,
        )
        print(
            f"Asserted Ollama effective num_ctx={asserted_num_ctx} "
            f"(origin={ollama_origin(llm_cfg.base_url)})",
            flush=True,
        )

    print(f"Loading IPC catalog: {ipc_jsonl}", flush=True)
    ipc_lookup = IPCLookup.from_jsonl(ipc_jsonl)
    print(f"IPC entries: {len(ipc_lookup)}", flush=True)

    only_ids = None
    if args.only_ids is not None:
        only_path = _resolve(args.only_ids)
        only_ids = _load_only_ids(only_path)
        print(f"Regenerating only {len(only_ids)} ids from {only_path}", flush=True)

    run_cfg = cfg.get("run") or {}
    limit = args.limit if args.limit is not None else run_cfg.get("limit")
    if getattr(args, "rebuild_pool", False) and args.limit is None:
        # Rebuild the pool in setup() without starting a generation pass.
        limit = 0
    if limit is not None:
        limit = int(limit)
    workers = args.workers if args.workers is not None else run_cfg.get("workers")
    workers = max(1, int(workers or 1))
    print(f"Workers: {workers}", flush=True)

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
            workers=workers,
            only_ids=only_ids,
            prompt_budget=budget,
        )
        all_stats.append(stats)

        task_dir = task_output_dir(dest, task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        run_meta = {
            "task": task_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "provider": llm_cfg.provider,
            "model": llm_cfg.model,
            "base_url": llm_cfg.base_url,
            "num_ctx": llm_cfg.num_ctx,
            "asserted_num_ctx": asserted_num_ctx,
            "max_output_tokens": llm_cfg.max_output_tokens,
            "safety_margin": llm_cfg.safety_margin,
            "input_budget": budget.input_budget,
            "repeat_instruction": budget.repeat_instruction,
            "tokenizer_id": budget.tokenizer_id,
            "temperature": llm_cfg.temperature,
            "ollama_truncation_drops": OLLAMA_TRUNCATION_DROPS,
            "ollama_truncation_notes": OLLAMA_TRUNCATION_NOTE,
            "stats": stats,
        }
        meta_path = task_dir / "run_meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)
            f.write("\n")
        print(f"[{task_id}] run metadata → {meta_path}", flush=True)

    print(f"Finished {len(all_stats)} task(s) → {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
