#!/usr/bin/env python3
"""OpenRouter zero-shot / 3-shot generation on the frozen eval test split."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_CONFIG,
    PROMPTINGS,
    TASKS,
    load_config,
    resolve_path,
    user_turn,
)
from holdings import model_slug, resolve_generator_dir  # noqa: E402
from io_local import (
    DurableShardWriter,
    append_done_id,
    load_done_ids,
    load_split_records,
)
from llm import LLMClient, llm_config_from_dict  # noqa: E402

logger = logging.getLogger("eval.baseline")


def load_exemplars(splits_dir: Path) -> dict[str, list[dict[str, Any]]]:
    path = splits_dir / "exemplars.json"
    if not path.is_file():
        raise FileNotFoundError(f"exemplars missing: {path} (run split_eval_data.py)")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"exemplars.json must be an object: {path}")
    return data


def build_messages(
    record: dict[str, Any],
    *,
    prompting: str,
    exemplars: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], int]:
    messages: list[dict[str, str]] = []
    k_eff = 0
    if prompting == "fewshot_k3":
        for ex in exemplars:
            messages.append(
                {
                    "role": "user",
                    "content": user_turn(
                        str(ex.get("instruction") or ""),
                        str(ex.get("input") or ""),
                    ),
                }
            )
            messages.append(
                {"role": "assistant", "content": str(ex.get("output") or "")}
            )
            k_eff += 1
    messages.append(
        {
            "role": "user",
            "content": user_turn(
                str(record.get("instruction") or ""),
                str(record.get("input") or ""),
            ),
        }
    )
    return messages, k_eff


def system_specs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    specs = cfg.get("systems") or []
    if not isinstance(specs, list) or not specs:
        raise ValueError("config systems: must be a non-empty list")
    return [s for s in specs if isinstance(s, dict)]


def resolve_jobs(
    specs: list[dict[str, Any]],
    *,
    system: str | None,
    all_systems: bool,
    prompting: str,
) -> list[tuple[dict[str, Any], str]]:
    selected: list[dict[str, Any]]
    if all_systems:
        selected = specs
    elif system:
        needle = system.strip().lower()
        selected = [
            s
            for s in specs
            if str(s.get("id") or "").lower() == needle
            or str(s.get("model") or "").lower() == needle
            or model_slug(str(s.get("model") or "")) == model_slug(system)
        ]
        if not selected:
            raise ValueError(f"unknown system: {system}")
    else:
        raise ValueError("pass --system MODEL or --all")
    promptings = list(PROMPTINGS) if prompting == "all" else [prompting]
    jobs: list[tuple[dict[str, Any], str]] = []
    for spec in selected:
        for p in promptings:
            jobs.append((spec, p))
    return jobs


def generate_one(
    client: LLMClient,
    record: dict[str, Any],
    *,
    prompting: str,
    exemplars: list[dict[str, Any]],
    spec: dict[str, Any],
    temperature: float,
    max_tokens: int,
    provider: str,
) -> dict[str, Any]:
    messages, k_eff = build_messages(
        record, prompting=prompting, exemplars=exemplars
    )
    pred = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
    model = str(spec.get("model") or "")
    return {
        "application_number": str(record.get("application_number") or ""),
        "task": str(record.get("task") or ""),
        "instruction": str(record.get("instruction") or ""),
        "input": str(record.get("input") or ""),
        "output": pred,
        "gold_output": str(record.get("output") or ""),
        "meta": {
            **(record.get("meta") if isinstance(record.get("meta"), dict) else {}),
            "baseline": {
                "model": model,
                "system_id": str(spec.get("id") or model_slug(model)),
                "role": str(spec.get("role") or ""),
                "provider": provider,
                "temperature": temperature,
                "prompting": prompting,
                "k": k_eff,
                "exemplar_ids": [
                    str(ex.get("application_number") or "") for ex in exemplars
                ]
                if prompting == "fewshot_k3"
                else [],
            },
        },
    }


def run_job(
    *,
    spec: dict[str, Any],
    prompting: str,
    task_id: str,
    records: list[dict[str, Any]],
    exemplars: list[dict[str, Any]],
    out_dir: Path,
    llm_cfg: dict[str, Any],
    temperature: float,
    max_tokens: int,
    workers: int,
    shard_size: int,
    limit: int | None,
) -> int:
    model = str(spec.get("model") or "")
    provider = str(llm_cfg.get("provider") or "openrouter")
    dest = out_dir
    dest.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(dest)
    pending = [
        rec
        for rec in records
        if str(rec.get("application_number") or "").strip() not in done
    ]
    if limit is not None:
        pending = pending[: max(0, limit)]
    if not pending:
        print(
            f"[{model_slug(model)} {prompting} {task_id}] nothing to do "
            f"(done={len(done)})",
            flush=True,
        )
        return 0

    client_cfg = dict(llm_cfg)
    client_cfg["provider"] = provider
    client_cfg["model"] = model
    client_cfg["temperature"] = temperature
    client_cfg["max_tokens"] = max_tokens
    client = LLMClient(llm_config_from_dict(client_cfg))
    writer = DurableShardWriter(dest, shard_size=shard_size)
    write_lock = threading.Lock()
    n_ok = 0
    n_fail = 0
    n_workers = max(1, workers)
    print(
        f"[{model_slug(model)} {prompting} {task_id}] "
        f"{len(pending)} pending (done={len(done)}) workers={n_workers}",
        flush=True,
    )

    def _one(rec: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
        app = str(rec.get("application_number") or "")
        try:
            out = generate_one(
                client,
                rec,
                prompting=prompting,
                exemplars=exemplars if prompting == "fewshot_k3" else [],
                spec=spec,
                temperature=temperature,
                max_tokens=max_tokens,
                provider=provider,
            )
            return app, out, None
        except Exception as exc:  # noqa: BLE001 — isolate worker failures
            return app, None, str(exc)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_one, rec) for rec in pending]
        for fut in as_completed(futures):
            app, rec, err = fut.result()
            if rec is None:
                n_fail += 1
                logger.warning("fail %s %s %s: %s", model, task_id, app, err)
                continue
            with write_lock:
                writer.add(rec)
                append_done_id(dest, app)
            n_ok += 1
            if n_ok % 25 == 0:
                print(
                    f"[{model_slug(model)} {prompting} {task_id}] "
                    f"{n_ok}/{len(pending)}",
                    flush=True,
                )
    writer.flush()
    print(
        f"[{model_slug(model)} {prompting} {task_id}] wrote {n_ok} failed={n_fail}",
        flush=True,
    )
    return 0 if n_fail == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run OpenRouter zero-shot / 3-shot baselines on frozen test IDs."
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--generator", default=None, help="Seed generator (Mode 1 passed/ slug)")
    p.add_argument("--system", default=None, help="System id or OpenRouter model")
    p.add_argument(
        "--all",
        action="store_true",
        dest="all_systems",
        help="Run all four systems in baselines.yaml",
    )
    p.add_argument(
        "--prompting",
        choices=("zeroshot", "fewshot", "fewshot_k3", "all"),
        default="all",
    )
    p.add_argument("--task", choices=TASKS, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    cfg_path = resolve_path(args.config)
    if not cfg_path.is_file():
        print(f"error: config not found: {cfg_path}", file=sys.stderr)
        return 1
    cfg = load_config(cfg_path)
    paths = cfg.get("paths") or {}
    llm_cfg = dict(cfg.get("llm") or {})
    decode = cfg.get("decode") or {}
    max_tokens_map = decode.get("max_tokens") or {}
    shard_size = int(decode.get("shard_size") or 100)
    temperature = float(llm_cfg.get("temperature") if llm_cfg.get("temperature") is not None else 0.0)
    workers = int(args.workers if args.workers is not None else llm_cfg.get("workers") or 12)
    prompting = "fewshot_k3" if args.prompting == "fewshot" else args.prompting

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
    splits_dir = output_root / "splits" / gen_slug
    if not splits_dir.is_dir():
        print(
            f"error: splits missing: {splits_dir} (run split_eval_data.py first)",
            file=sys.stderr,
        )
        return 1
    try:
        exemplars = load_exemplars(splits_dir)
        jobs = resolve_jobs(
            system_specs(cfg),
            system=args.system,
            all_systems=args.all_systems,
            prompting=prompting,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    tasks = [args.task] if args.task else list(TASKS)
    rc = 0
    for spec, prompt in jobs:
        sys_slug = model_slug(str(spec.get("model") or spec.get("id") or "unknown"))
        for task_id in tasks:
            test_rows = load_split_records(splits_dir / task_id, "test")
            if not test_rows:
                print(f"[{task_id}] no test rows; skip", flush=True)
                continue
            max_tokens = int(max_tokens_map.get(task_id) or 1024)
            dest = output_root / "predictions" / sys_slug / prompt / task_id
            job_rc = run_job(
                spec=spec,
                prompting=prompt,
                task_id=task_id,
                records=test_rows,
                exemplars=list(exemplars.get(task_id) or []),
                out_dir=dest,
                llm_cfg=llm_cfg,
                temperature=temperature,
                max_tokens=max_tokens,
                workers=workers,
                shard_size=shard_size,
                limit=args.limit,
            )
            rc = rc or job_rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
