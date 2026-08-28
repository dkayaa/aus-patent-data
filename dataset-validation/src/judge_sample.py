"""Deterministic sampling and resume helpers for LLM judge."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any


DONE_IDS_FILENAME = "done_ids.txt"


def load_done_ids(judge_dir: Path) -> set[str]:
    path = judge_dir / DONE_IDS_FILENAME
    if not path.is_file():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                ids.add(value)
    return ids


def append_done_id(judge_dir: Path, application_number: str) -> None:
    judge_dir.mkdir(parents=True, exist_ok=True)
    path = judge_dir / DONE_IDS_FILENAME
    with path.open("a", encoding="utf-8") as f:
        f.write(application_number)
        f.write("\n")
        f.flush()


def load_ids_file(path: Path) -> list[str]:
    """Load application numbers, preserving order and dropping blanks/duplicates."""
    ids: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            ids.append(value)
    return ids


def records_for_ids(
    records: list[dict[str, Any]],
    ids: list[str],
    *,
    skip_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select records in ``ids`` order. Returns (matched, missing_ids)."""
    skip = skip_ids or set()
    by_id: dict[str, dict[str, Any]] = {}
    for rec in records:
        app = str(rec.get("application_number") or "").strip()
        if not app or app in by_id:
            continue
        by_id[app] = rec
    matched: list[dict[str, Any]] = []
    missing: list[str] = []
    for app in ids:
        if app in skip:
            continue
        rec = by_id.get(app)
        if rec is None:
            missing.append(app)
            continue
        matched.append(rec)
    return matched, missing


def remaining_records(
    records: list[dict[str, Any]],
    *,
    skip_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Mode 1 order, unique ids, skip already judged. No shuffle."""
    skip = skip_ids or set()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in records:
        app = str(rec.get("application_number") or "").strip()
        if not app or app in seen or app in skip:
            continue
        seen.add(app)
        out.append(rec)
    return out


def sample_records(
    records: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
    skip_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Deterministic sample, then drop resume ids (same sample on re-run)."""
    skip = skip_ids or set()
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in records:
        app = str(rec.get("application_number") or "").strip()
        if not app or app in seen:
            continue
        seen.add(app)
        pool.append(rec)
    rng = random.Random(seed)
    rng.shuffle(pool)
    if sample_size < 1:
        return []
    sampled = pool[:sample_size]
    return [
        rec
        for rec in sampled
        if str(rec.get("application_number") or "").strip() not in skip
    ]
