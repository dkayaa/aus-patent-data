#!/usr/bin/env python3
"""CLI: Mode 1 programmatic validation with lexical + semantic scores."""

from __future__ import annotations

import argparse
import gzip
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
from semantic import SemanticScorer, load_embedding_registry  # noqa: E402
from task_metrics import TASKS, score_record  # noqa: E402
from terms_coverage import TermsCoverageScorer  # noqa: E402

IG_SRC = REPO_ROOT / "instruction-generation" / "src"
if str(IG_SRC) not in sys.path:
    sys.path.insert(0, str(IG_SRC))

from holdings import resolve_generator_dir  # noqa: E402
from ipc_lookup import IPCLookup  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "dataset-validation" / "config" / "programmatic.yaml"
DEFAULT_REGISTRY = REPO_ROOT / "dataset-validation" / "config" / "embedding_models.yaml"


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
            "ROUGE-L / best-span-F1 + configurable embedding cosine + Terms Coverage."
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
        help="Override task input dir (implies single --task)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override task output dir (implies single --task)",
    )
    p.add_argument(
        "--embedding-model",
        default=None,
        help="Override semantic.embedding_model key (minilm|granite|granite_small|nomic)",
    )
    p.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Skip embedding cosine (lexical + structural only)",
    )
    p.add_argument(
        "--faithfulness",
        action="store_true",
        help=(
            "Run MiniCheck faithfulness on ipc_reasoning (additive, non-gating). "
            "Slow; prefer scripts/run_faithfulness_ipc.py for the report."
        ),
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
    terms: TermsCoverageScorer | None,
    faithfulness: Any | None,
    ipc_lookup: Any | None,
    shard_size: int,
    limit: int | None,
) -> dict[str, Any]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir missing: {input_dir}")

    passed_writer = ShardWriter(output_dir / "passed", shard_size=shard_size)
    rejected_writer = ShardWriter(output_dir / "rejected", shard_size=shard_size)
    faith_sent_path = output_dir / "faithfulness" / "sentences.jsonl.gz"
    faith_sent_rows: list[dict[str, Any]] = []

    fail_counts: Counter[str] = Counter()
    rouge_vals: list[float] = []
    cos_vals: list[float] = []
    best_span_vals: list[float] = []
    claims_cos_vals: list[float] = []
    claims_rouge_vals: list[float] = []
    terms_vals: list[float] = []
    faith_vals: list[float] = []
    n_chunked = 0
    n_total = 0
    n_pass = 0
    n_reject = 0

    for record in iter_task_records(input_dir):
        if limit is not None and n_total >= limit:
            break
        if not record.get("task"):
            record = {**record, "task": task_id}
        n_total += 1

        validation = score_record(
            record,
            semantic=semantic,
            floors=floors,
            ipc_lookup=ipc_lookup,
            terms=terms,
            faithfulness=faithfulness if task_id == "ipc_reasoning" else None,
        )
        sent_detail = validation.pop("faithfulness_sentences", None)
        if sent_detail is not None:
            faith_sent_rows.append(sent_detail)

        meta = dict(record.get("meta") or {})
        # Keep any prior validation block under a distinct key so re-runs with a
        # new embedding model do not silently erase the old cosine numbers.
        if "validation" in meta and isinstance(meta["validation"], dict):
            prev = meta["validation"]
            prev_scores = (prev.get("scores") or {}) if isinstance(prev, dict) else {}
            prev_model = prev_scores.get("embedding_model") or prev_scores.get(
                "embedding_model_id"
            )
            cur_model = (validation.get("scores") or {}).get("embedding_model")
            if prev_model and cur_model and prev_model != cur_model:
                archive_key = f"validation_prior_{prev_model}"
                meta.setdefault(archive_key, prev)
        meta["validation"] = validation
        out_rec = {**record, "meta": meta}

        scores = validation.get("scores") or {}
        if scores.get("rouge_l_f1") is not None:
            rouge_vals.append(float(scores["rouge_l_f1"]))
        if scores.get("semantic_cosine") is not None:
            cos_vals.append(float(scores["semantic_cosine"]))
        if scores.get("best_span_f1") is not None:
            best_span_vals.append(float(scores["best_span_f1"]))
        if scores.get("claims_semantic_cosine") is not None:
            claims_cos_vals.append(float(scores["claims_semantic_cosine"]))
        if scores.get("claims_rouge_l_f1") is not None:
            claims_rouge_vals.append(float(scores["claims_rouge_l_f1"]))
        if scores.get("terms_coverage") is not None:
            terms_vals.append(float(scores["terms_coverage"]))
        if scores.get("faithfulness_rate") is not None:
            faith_vals.append(float(scores["faithfulness_rate"]))
        if scores.get("n_chunks") is not None and int(scores["n_chunks"]) > 1:
            n_chunked += 1

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

    if faith_sent_rows:
        faith_sent_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(faith_sent_path, "wt", encoding="utf-8") as f:
            for row in faith_sent_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "task": task_id,
        "n_total": n_total,
        "n_passed": n_pass,
        "n_rejected": n_reject,
        "fail_rule_counts": dict(fail_counts),
        "mean_rouge_l_f1": _mean(rouge_vals),
        "mean_semantic_cosine": _mean(cos_vals),
        "mean_best_span_f1": _mean(best_span_vals),
        "mean_claims_semantic_cosine": _mean(claims_cos_vals),
        "mean_claims_rouge_l_f1": _mean(claims_rouge_vals),
        "mean_terms_coverage": _mean(terms_vals),
        "mean_faithfulness_rate": _mean(faith_vals),
        "n_chunked": n_chunked,
        "embedding_model": semantic.model_key if semantic else None,
        "embedding_model_id": semantic.model_name if semantic else None,
        "max_seq_length": semantic.max_seq_length if semantic else None,
        "passed_shards": [str(p) for p in passed_writer.paths],
        "rejected_shards": [str(p) for p in rejected_writer.paths],
    }
    if faith_sent_rows:
        report["faithfulness_sentences_path"] = str(faith_sent_path)
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

    floor_cfg = cfg.get("floors") or {}
    floors = {
        "semantic_cosine_min": float(floor_cfg.get("semantic_cosine_min", 0.15)),
        "rouge_l_f1_min": float(floor_cfg.get("rouge_l_f1_min", 0.02)),
        "mrc_best_span_f1_min": float(floor_cfg.get("mrc_best_span_f1_min", 0.5)),
        "ipc_wipo_cosine_min": float(floor_cfg.get("ipc_wipo_cosine_min", 0.55)),
        "ipc_wipo_rouge_l_f1_min": float(floor_cfg.get("ipc_wipo_rouge_l_f1_min", 0.08)),
        "ipc_wipo_rouge_l_f1_max": float(floor_cfg.get("ipc_wipo_rouge_l_f1_max", 0.60)),
        "ipc_claims_cosine_min": float(floor_cfg.get("ipc_claims_cosine_min", 0.50)),
        "abstract_cosine_min": float(floor_cfg.get("abstract_cosine_min", 0.40)),
    }

    if args.input_dir is not None or args.output_dir is not None:
        if args.all or not args.task:
            print(
                "error: --input-dir/--output-dir require a single --task",
                file=sys.stderr,
            )
            return 1

    task_ids = list(TASKS) if args.all else [args.task]
    semantic: SemanticScorer | None = None
    load_semantic = (not args.skip_semantic) and any(tid != "mrc" for tid in task_ids)
    if load_semantic:
        sem_cfg = cfg.get("semantic") or {}
        model_key = str(
            args.embedding_model or sem_cfg.get("embedding_model") or "nomic"
        )
        registry = load_embedding_registry(DEFAULT_REGISTRY)
        if model_key not in registry:
            print(
                f"error: unknown embedding_model {model_key!r}; "
                f"choose from {sorted(registry)}",
                file=sys.stderr,
            )
            return 1
        spec = registry[model_key]
        print(
            f"Loading semantic model key={model_key} name={spec.model_name}",
            flush=True,
        )
        semantic = SemanticScorer(
            spec,
            batch_size=int(sem_cfg.get("batch_size") or 8),
            chunk_overlap_frac=float(sem_cfg.get("chunk_overlap_frac") or 0.25),
        )

    terms: TermsCoverageScorer | None = None
    tc_cfg = cfg.get("terms_coverage") or {}
    if bool(tc_cfg.get("enabled", True)):
        bp = _resolve(
            Path(
                tc_cfg.get("boilerplate_config")
                or "dataset-validation/config/terms_boilerplate.yaml"
            )
        )
        terms = TermsCoverageScorer(bp)

    faithfulness = None
    faith_cfg = cfg.get("faithfulness") or {}
    if args.faithfulness or bool(faith_cfg.get("enabled")):
        from faithfulness import FaithfulnessScorer

        faithfulness = FaithfulnessScorer(
            cache_dir=_resolve(Path(faith_cfg.get("cache_dir") or "ckpts")),
            batch_size=int(faith_cfg.get("batch_size") or 8),
            support_high=float(faith_cfg.get("support_high") or 0.7),
            support_low=float(faith_cfg.get("support_low") or 0.3),
            score_halves=bool(faith_cfg.get("score_halves") or False),
            terms=terms,
        )

    ipc_lookup: Any | None = None
    if any(tid == "ipc_reasoning" for tid in task_ids):
        ipc_jsonl = _resolve(
            Path(paths.get("ipc_jsonl") or "data/ipc-codes/ipc_codes_20260101.jsonl")
        )
        if not ipc_jsonl.is_file():
            print(f"error: IPC catalog not found: {ipc_jsonl}", file=sys.stderr)
            return 1
        ipc_lookup = IPCLookup.from_jsonl(ipc_jsonl)
        print(f"IPC catalog: {len(ipc_lookup)} entries from {ipc_jsonl}", flush=True)

    gen_in: Path | None = None
    gen_out: Path | None = None
    if args.input_dir is None or args.output_dir is None:
        try:
            gen_in = resolve_generator_dir(input_root, generator=args.generator)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        gen_out = output_root / gen_in.name
        print(f"Generator: {gen_in.name}", flush=True)

    reports = []
    for task_id in task_ids:
        if args.input_dir is not None:
            in_dir = _resolve(args.input_dir)
        else:
            assert gen_in is not None
            in_dir = gen_in / task_id
        if args.output_dir is not None:
            out_dir = _resolve(args.output_dir)
        else:
            assert gen_out is not None
            out_dir = gen_out / task_id
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
                terms=terms,
                faithfulness=faithfulness,
                ipc_lookup=ipc_lookup,
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
