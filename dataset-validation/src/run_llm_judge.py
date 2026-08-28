#!/usr/bin/env python3
"""CLI: Mode 2 LLM-as-a-judge over a sample of Mode 1 passed rows."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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

from holdings import resolve_generator_dir  # noqa: E402
from io_util import ShardWriter, iter_task_records, write_jsonl_gz  # noqa: E402
from judge_prompts import (  # noqa: E402
    TASKS,
    build_judge_messages,
    normalize_judge_result,
)
from judge_sample import (  # noqa: E402
    append_done_id,
    load_done_ids,
    load_ids_file,
    records_for_ids,
    remaining_records,
    sample_records,
)
from llm import LLMClient, chat_json, llm_config_from_dict  # noqa: E402
from ipc_lookup import IPCLookup  # noqa: E402

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
            "LLM-as-a-judge: grade Mode 1 passed rows (Mode 2 sample by default; "
            "cheap/full corpus with cheap_judge.yaml + --full)."
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
        help="Override sample_size per task (default from YAML). Incompatible with --full.",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Grade every Mode 1 passed row (no seed shuffle). Incompatible with --limit.",
    )
    p.add_argument(
        "--generator",
        default=None,
        help=(
            "Generator model id or slug under input_root "
            "(default: the only generator dir, error if several)"
        ),
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
        help="Override judge output dir (implies single --task)",
    )
    p.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help=(
            "Pin application_number list (one per line). Skips seed shuffle. "
            "Requires a single --task."
        ),
    )
    p.add_argument(
        "--provider",
        choices=("local", "openrouter"),
        default=None,
        help="Override LLM provider (default: openrouter from config)",
    )
    p.add_argument("--model", type=str, default=None, help="Override judge model")
    p.add_argument("--base-url", default=None, help="Override llm.base_url")
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Concurrent judge LLM calls (default: judge.workers in YAML, else 1). "
            "Use >1 for OpenRouter; keep 1 for a serial local server."
        ),
    )
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
    """ShardWriter that continues numbering after existing shards.

    Rewrites the in-progress shard on each add so a crash cannot leave
    done_ids pointing at records that were never flushed.
    """

    def __init__(self, out_dir: Path, *, shard_size: int = 100) -> None:
        super().__init__(out_dir, shard_size=shard_size)
        self._index = _next_shard_index(out_dir)

    def add(self, record: dict[str, Any]) -> None:
        self._buffer.append(record)
        self.n_written += 1
        path = self.out_dir / f"part-{self._index:05d}.jsonl.gz"
        write_jsonl_gz(path, self._buffer)
        if not self.paths or self.paths[-1] != path:
            self.paths.append(path)
        if len(self._buffer) >= self.shard_size:
            self._index += 1
            self._buffer.clear()

    def flush(self) -> None:
        if not self._buffer:
            return
        path = self.out_dir / f"part-{self._index:05d}.jsonl.gz"
        write_jsonl_gz(path, self._buffer)
        if not self.paths or self.paths[-1] != path:
            self.paths.append(path)
        self._index += 1
        self._buffer.clear()


def _try_judge(
    client: LLMClient,
    rec: dict[str, Any],
    *,
    task_id: str,
    truncate_chars: int,
    pass_score_min: int,
    ipc_lookup: IPCLookup | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, BaseException | None]:
    try:
        messages = build_judge_messages(
            rec, truncate_chars=truncate_chars, ipc_lookup=ipc_lookup
        )
        raw = chat_json(client, messages, expect=dict)
        if not isinstance(raw, dict):
            raise ValueError(f"expected dict, got {type(raw)}")
        result = normalize_judge_result(
            raw, task=task_id, pass_score_min=pass_score_min
        )
        return rec, result, None
    except Exception as exc:  # noqa: BLE001 — continue like generation
        return rec, None, exc


def judge_task(
    task_id: str,
    *,
    input_dir: Path,
    output_dir: Path,
    client: LLMClient,
    sample_size: int | None,
    seed: int,
    pass_score_min: int,
    truncate_chars: int,
    shard_size: int,
    workers: int,
    ipc_lookup: IPCLookup | None = None,
    pin_ids: list[str] | None = None,
    ids_file: Path | None = None,
) -> dict[str, Any]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Mode 1 passed dir missing: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output_dir)

    all_records = list(iter_task_records(input_dir))
    missing_ids: list[str] = []
    if pin_ids is not None:
        to_judge, missing_ids = records_for_ids(
            all_records, pin_ids, skip_ids=done_ids
        )
        sample_size_target = len(pin_ids)
        if missing_ids:
            log.warning(
                "%s: %d ids from --ids-file not in Mode 1 passed: %s",
                task_id,
                len(missing_ids),
                ", ".join(missing_ids[:10])
                + ("…" if len(missing_ids) > 10 else ""),
            )
        log.info(
            "%s: %d Mode1 passed, %d already judged, pinning %d new (target %d)",
            task_id,
            len(all_records),
            len(done_ids),
            len(to_judge),
            sample_size_target,
        )
    elif sample_size is None:
        to_judge = remaining_records(all_records, skip_ids=done_ids)
        sample_size_target = len(all_records)
        log.info(
            "%s: %d Mode1 passed, %d already judged, full remaining %d",
            task_id,
            len(all_records),
            len(done_ids),
            len(to_judge),
        )
    else:
        to_judge = sample_records(
            all_records,
            sample_size=sample_size,
            seed=seed,
            skip_ids=done_ids,
        )
        sample_size_target = sample_size
        log.info(
            "%s: %d Mode1 passed, %d already judged, sampling %d new (target %d)",
            task_id,
            len(all_records),
            len(done_ids),
            len(to_judge),
            sample_size_target,
        )
    workers = max(1, int(workers))
    if workers > 1:
        log.info("%s: workers=%d", task_id, workers)

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

    def _record_verdict(
        rec: dict[str, Any],
        result: dict[str, Any] | None,
        err: BaseException | None,
    ) -> None:
        nonlocal n_pass, n_fail, n_errors
        app = str(rec.get("application_number") or "").strip()
        if err is not None:
            n_errors += 1
            log.warning("%s %s: judge error: %s", task_id, app, err)
            return
        if result is None:
            n_errors += 1
            log.warning("%s %s: judge error: empty result", task_id, app)
            return

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
        n_ok = n_pass + n_fail
        if n_ok % 10 == 0:
            log.info("%s: judged %d…", task_id, n_ok)

    if workers == 1:
        for rec in to_judge:
            rec, result, err = _try_judge(
                client,
                rec,
                task_id=task_id,
                truncate_chars=truncate_chars,
                pass_score_min=pass_score_min,
                ipc_lookup=ipc_lookup,
            )
            _record_verdict(rec, result, err)
    else:
        _judge_parallel(
            to_judge,
            client=client,
            task_id=task_id,
            truncate_chars=truncate_chars,
            pass_score_min=pass_score_min,
            workers=workers,
            ipc_lookup=ipc_lookup,
            record_verdict=_record_verdict,
        )

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
        "sample_size_target": sample_size_target,
        "full_corpus": pin_ids is None and sample_size is None,
        "seed": None if pin_ids is not None or sample_size is None else seed,
        "ids_file": str(ids_file) if ids_file is not None else None,
        "n_ids_missing": len(missing_ids),
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
        "workers": workers,
        "failure_tag_counts": dict(merged_tags.most_common()),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "note": (
            "Full-corpus cheap judge; not a training filter. pass is score >= pass_score_min."
            if pin_ids is None and sample_size is None
            else "Sample-based Mode 2 judge; not a full-corpus pass. pass is score >= pass_score_min."
        ),
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


def _judge_parallel(
    to_judge: list[dict[str, Any]],
    *,
    client: LLMClient,
    task_id: str,
    truncate_chars: int,
    pass_score_min: int,
    workers: int,
    ipc_lookup: IPCLookup | None,
    record_verdict: Any,
) -> None:
    pending = iter(to_judge)
    exhausted = False
    inflight: dict[
        Future[tuple[dict[str, Any], dict[str, Any] | None, BaseException | None]],
        dict[str, Any],
    ] = {}
    target_inflight = workers * 3

    def _submit(executor: ThreadPoolExecutor) -> bool:
        nonlocal exhausted
        if exhausted:
            return False
        try:
            rec = next(pending)
        except StopIteration:
            exhausted = True
            return False
        fut = executor.submit(
            _try_judge,
            client,
            rec,
            task_id=task_id,
            truncate_chars=truncate_chars,
            pass_score_min=pass_score_min,
            ipc_lookup=ipc_lookup,
        )
        inflight[fut] = rec
        return True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while True:
            while not exhausted and len(inflight) < target_inflight:
                if not _submit(executor):
                    break
            if not inflight:
                break
            finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in finished:
                rec = inflight.pop(fut)
                try:
                    rec, result, err = fut.result()
                except Exception as exc:  # noqa: BLE001
                    record_verdict(rec, None, exc)
                    continue
                record_verdict(rec, result, err)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.full and args.limit is not None:
        log.error("--full is incompatible with --limit")
        return 2
    cfg = load_config(_resolve(args.config) if not args.config.is_absolute() else args.config)

    paths = cfg.get("paths") or {}
    judge_cfg = cfg.get("judge") or {}
    llm_raw = cfg.get("llm") or {}

    input_root = _resolve(Path(paths.get("input_root", "data/derived/instruction_generation_validation")))
    output_root = _resolve(Path(paths.get("output_root", "data/derived/instruction_generation_validation")))
    ipc_jsonl = _resolve(
        Path(paths.get("ipc_jsonl") or "data/ipc-codes/ipc_codes_20260101.jsonl")
    )
    ipc_lookup: IPCLookup | None = None
    if ipc_jsonl.is_file():
        ipc_lookup = IPCLookup.from_jsonl(ipc_jsonl)
        log.info("WIPO catalog: %s (%d codes)", ipc_jsonl, len(ipc_lookup))
    else:
        log.warning("WIPO catalog missing (%s); IPC judge will lack definition_statement", ipc_jsonl)

    sample_size: int | None
    if args.full:
        sample_size = None
    elif args.limit is not None:
        sample_size = int(args.limit)
    else:
        raw_size = judge_cfg.get("sample_size", 50)
        sample_size = None if raw_size is None else int(raw_size)
    seed = int(judge_cfg.get("seed", 42))
    pass_score_min = int(judge_cfg.get("pass_score_min", 4))
    truncate_chars = int(judge_cfg.get("truncate_chars", 12000))
    shard_size = int(judge_cfg.get("shard_size", 50))
    output_subdir = str(paths.get("output_subdir") or "llm_judge")
    workers = args.workers if args.workers is not None else judge_cfg.get("workers")
    workers = max(1, int(workers or 1))

    pin_ids: list[str] | None = None
    ids_file: Path | None = None
    if args.ids_file is not None:
        if args.all or not args.task:
            log.error("--ids-file requires a single --task")
            return 2
        ids_file = _resolve(args.ids_file)
        if not ids_file.is_file():
            log.error("--ids-file not found: %s", ids_file)
            return 1
        pin_ids = load_ids_file(ids_file)
        if not pin_ids:
            log.error("--ids-file is empty: %s", ids_file)
            return 1
        log.info("Pinned %d ids from %s", len(pin_ids), ids_file)

    overrides: dict[str, Any] = {}
    if args.provider:
        overrides["provider"] = args.provider
    if args.model:
        overrides["model"] = args.model
    if args.base_url:
        overrides["base_url"] = args.base_url
    llm_cfg = llm_config_from_dict(llm_raw, overrides=overrides)
    client = LLMClient(llm_cfg)
    log.info("Judge provider=%s model=%s workers=%s", llm_cfg.provider, llm_cfg.model, workers)

    if args.input_dir or args.output_dir:
        if args.all or not args.task:
            log.error("--input-dir/--output-dir require a single --task")
            return 2
        tasks = [args.task]
    elif args.all:
        tasks = list(TASKS)
    else:
        tasks = [args.task]

    gen_dir: Path | None = None
    if args.input_dir is None or args.output_dir is None:
        try:
            gen_dir = resolve_generator_dir(input_root, generator=args.generator)
        except (FileNotFoundError, ValueError) as exc:
            log.error("%s", exc)
            return 1
        log.info("Generator: %s", gen_dir.name)

    reports: list[dict[str, Any]] = []
    for task_id in tasks:
        if args.input_dir:
            in_dir = _resolve(args.input_dir)
        else:
            assert gen_dir is not None
            in_dir = gen_dir / task_id / "passed"
        if args.output_dir:
            out_dir = _resolve(args.output_dir)
        else:
            assert gen_dir is not None
            out_dir = gen_dir / task_id / output_subdir
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
                    workers=workers,
                    ipc_lookup=ipc_lookup,
                    pin_ids=pin_ids,
                    ids_file=ids_file,
                )
            )
        except FileNotFoundError as exc:
            log.error("%s", exc)
            return 1

    summary_dir = gen_dir if gen_dir is not None else output_root
    summary_path = summary_dir / f"{output_subdir}_summary.json"
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
        "output_subdir": output_subdir,
        "note": (
            "Cheap/full judge; see per-task cheap_judge/report.json"
            if output_subdir != "llm_judge"
            else "Sample-based Mode 2; see per-task llm_judge/report.json"
        ),
    }
    if len(reports) > 1 or args.all:
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        log.info("Wrote summary %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
