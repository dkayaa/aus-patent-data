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
