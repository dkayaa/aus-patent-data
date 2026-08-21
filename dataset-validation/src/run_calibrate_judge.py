"""Mode 2 calibration hook: distributions now; human agreement when labels exist."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
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
from io_util import iter_task_records  # noqa: E402
from ipc_lookup import IPCLookup  # noqa: E402
from judge_prompts import TASKS, wipo_fields_for_record  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "dataset-validation" / "config" / "llm_judge.yaml"
HUMAN_AUDIT_FILENAME = "human_audit.jsonl"
CALIBRATION_FILENAME = "llm_judge_calibration.json"
THRESHOLDS = (3, 4, 5)

log = logging.getLogger("calibrate_llm_judge")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def human_accept_to_bool(value: Any) -> bool | None:
    """Map Mode 3 accept labels. yes → True; no/fix → False."""
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"yes", "y", "true", "accept", "1"}:
        return True
    if text in {"no", "n", "false", "reject", "fix", "0"}:
        return False
    return None


def cohens_kappa(human: list[bool], pred: list[bool]) -> float | None:
    n = len(human)
    if n == 0 or n != len(pred):
        return None
    tp = sum(1 for h, p in zip(human, pred) if h and p)
    tn = sum(1 for h, p in zip(human, pred) if (not h) and (not p))
    fp = sum(1 for h, p in zip(human, pred) if (not h) and p)
    fn = sum(1 for h, p in zip(human, pred) if h and (not p))
    po = (tp + tn) / n
    p_human = (tp + fn) / n
    p_pred = (tp + fp) / n
    pe = p_human * p_pred + (1.0 - p_human) * (1.0 - p_pred)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def confusion_matrix(human: list[bool], pred: list[bool]) -> dict[str, int]:
    return {
        "human_accept_judge_pass": sum(1 for h, p in zip(human, pred) if h and p),
        "human_accept_judge_fail": sum(1 for h, p in zip(human, pred) if h and (not p)),
        "human_reject_judge_pass": sum(1 for h, p in zip(human, pred) if (not h) and p),
        "human_reject_judge_fail": sum(
            1 for h, p in zip(human, pred) if (not h) and (not p)
        ),
        "n": len(human),
    }


def load_judged_records(judge_dir: Path) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    for split in ("passed", "rejected"):
        split_dir = judge_dir / split
        if split_dir.is_dir():
            recs.extend(iter_task_records(split_dir))
    return recs


def load_human_audit(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec, dict):
                rows.append(rec)
    return rows


def _judge_meta(rec: dict[str, Any]) -> dict[str, Any]:
    meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
    judge = meta.get("llm_judge") if isinstance(meta.get("llm_judge"), dict) else {}
    return judge


def task_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores: list[int] = []
    tags: Counter[str] = Counter()
    n_pass = 0
    n_fail = 0
    histogram: Counter[int] = Counter()
    for rec in records:
        judge = _judge_meta(rec)
        if "score" not in judge:
            continue
        score = int(judge["score"])
        scores.append(score)
        histogram[score] += 1
        if judge.get("pass"):
            n_pass += 1
        else:
            n_fail += 1
        for tag in judge.get("failure_tags") or []:
            tags[str(tag)] += 1
    n = len(scores)
    return {
        "n_judged": n,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "pass_rate": (n_pass / n) if n else None,
        "mean_score": (sum(scores) / n) if n else None,
        "score_histogram": {str(k): histogram[k] for k in range(1, 6)},
        "failure_tag_counts": dict(tags.most_common()),
    }


def agreement_for_threshold(
    pairs: list[tuple[bool, int]],
    *,
    pass_score_min: int,
) -> dict[str, Any]:
    human = [h for h, _ in pairs]
    pred = [score >= pass_score_min for _, score in pairs]
    n = len(pairs)
    n_agree = sum(1 for h, p in zip(human, pred) if h == p)
    return {
        "pass_score_min": pass_score_min,
        "n_paired": n,
        "pct_agreement": (n_agree / n) if n else None,
        "cohens_kappa": cohens_kappa(human, pred),
        "confusion": confusion_matrix(human, pred),
    }


def pair_human_judge(
    judged: list[dict[str, Any]],
    human_rows: list[dict[str, Any]],
    *,
    task: str,
) -> list[tuple[bool, int]]:
    by_id: dict[str, dict[str, Any]] = {}
    for rec in judged:
        app = str(rec.get("application_number") or "").strip()
        if app:
            by_id[app] = rec
    pairs: list[tuple[bool, int]] = []
    for row in human_rows:
        if str(row.get("task") or "") != task:
            continue
        accept = human_accept_to_bool(row.get("accept"))
        if accept is None:
            continue
        app = str(row.get("application_number") or "").strip()
        rec = by_id.get(app)
        if rec is None:
            continue
        judge = _judge_meta(rec)
        if "score" not in judge:
            continue
        pairs.append((accept, int(judge["score"])))
    return pairs


def export_blind_csv(
    path: Path,
    *,
    records: list[dict[str, Any]],
    task: str,
    ipc_lookup: IPCLookup | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "application_number",
        "task",
        "instruction",
        "input",
        "output",
        "primary_ipc",
        "ipc_title",
        "definition_statement",
        "accept",
        "score",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
            wipo = wipo_fields_for_record(rec, ipc_lookup) or {}
            writer.writerow(
                {
                    "application_number": rec.get("application_number") or "",
                    "task": task,
                    "instruction": rec.get("instruction") or "",
                    "input": rec.get("input") or "",
                    "output": rec.get("output") or "",
                    "primary_ipc": meta.get("primary_ipc") or "",
                    "ipc_title": wipo.get("ipc_title") or meta.get("ipc_title") or "",
                    "definition_statement": wipo.get("definition_statement") or "",
                    "accept": "",
                    "score": "",
                    "note": "",
                }
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Summarize Mode 2 LLM-judge distributions; if human_audit.jsonl "
            "exists, compute agreement and a pass_score_min sweep."
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--generator",
        default=None,
        help="Generator model id or slug under input_root",
    )
    p.add_argument("--task", choices=list(TASKS), default=None)
    p.add_argument(
        "--export-csv",
        action="store_true",
        help="Write blind audit CSVs (no judge scores) next to human_audit.jsonl",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    cfg = load_config(_resolve(args.config) if not args.config.is_absolute() else args.config)
    paths = cfg.get("paths") or {}
    input_root = _resolve(
        Path(paths.get("input_root", "data/derived/instruction_generation_validation"))
    )
    try:
        gen_dir = resolve_generator_dir(input_root, generator=args.generator)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    log.info("Generator: %s", gen_dir.name)

    ipc_jsonl = _resolve(
        Path(paths.get("ipc_jsonl") or "data/ipc-codes/ipc_codes_20260101.jsonl")
    )
    ipc_lookup: IPCLookup | None = None
    if ipc_jsonl.is_file():
        ipc_lookup = IPCLookup.from_jsonl(ipc_jsonl)

    tasks = [args.task] if args.task else list(TASKS)
    human_path = gen_dir / HUMAN_AUDIT_FILENAME
    human_rows = load_human_audit(human_path)
    if human_rows:
        log.info("Loaded %d human audit rows from %s", len(human_rows), human_path)
    else:
        log.info("No %s yet — distributions only", HUMAN_AUDIT_FILENAME)

    report: dict[str, Any] = {
        "generator": gen_dir.name,
        "human_audit": str(human_path) if human_path.is_file() else None,
        "n_human_rows": len(human_rows),
        "tasks": [],
        "note": (
            "pass is score >= pass_score_min. Human accept: yes→accept; "
            "no/fix→reject. Threshold sweep is empty until human_audit.jsonl exists."
        ),
    }

    for task_id in tasks:
        judge_dir = gen_dir / task_id / "llm_judge"
        judged = load_judged_records(judge_dir)
        dist = task_distribution(judged)
        block: dict[str, Any] = {"task": task_id, **dist}
        pairs = pair_human_judge(judged, human_rows, task=task_id)
        if pairs:
            operating = int((cfg.get("judge") or {}).get("pass_score_min", 4))
            block["agreement"] = agreement_for_threshold(
                pairs, pass_score_min=operating
            )
            block["threshold_sweep"] = [
                agreement_for_threshold(pairs, pass_score_min=t) for t in THRESHOLDS
            ]
        report["tasks"].append(block)

        log.info(
            "%s: n=%s mean=%s pass_rate=%s tags=%s",
            task_id,
            dist["n_judged"],
            f"{dist['mean_score']:.3f}" if dist["mean_score"] is not None else "n/a",
            f"{dist['pass_rate']:.2%}" if dist["pass_rate"] is not None else "n/a",
            dist["failure_tag_counts"],
        )
        if pairs:
            agr = block["agreement"]
            log.info(
                "%s human↔judge: n=%s agree=%s kappa=%s",
                task_id,
                agr["n_paired"],
                f"{agr['pct_agreement']:.2%}" if agr["pct_agreement"] is not None else "n/a",
                f"{agr['cohens_kappa']:.3f}" if agr["cohens_kappa"] is not None else "n/a",
            )

        if args.export_csv:
            # Blind: Mode 1 passed rows for judged IDs, no llm_judge fields
            passed_dir = gen_dir / task_id / "passed"
            source = list(iter_task_records(passed_dir)) if passed_dir.is_dir() else []
            wanted = {
                str(r.get("application_number") or "").strip()
                for r in judged
                if str(r.get("application_number") or "").strip()
            }
            blind = [
                r
                for r in source
                if str(r.get("application_number") or "").strip() in wanted
            ]
            csv_path = gen_dir / f"human_audit_{task_id}.csv"
            export_blind_csv(
                csv_path, records=blind, task=task_id, ipc_lookup=ipc_lookup
            )
            log.info("Wrote blind CSV %s (%d rows)", csv_path, len(blind))

    out_path = gen_dir / CALIBRATION_FILENAME
    if human_rows:
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        log.info("Wrote %s", out_path)
    else:
        # Still persist distributions so the hook has an artifact before Mode 3
        dist_path = gen_dir / "llm_judge_distributions.json"
        dist_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        log.info("Wrote %s (no human labels yet)", dist_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
