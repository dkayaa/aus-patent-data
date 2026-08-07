"""Write Alpaca-style instruction JSONL shards with resume support."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

DONE_IDS_FILENAME = "done_ids.txt"


def task_output_dir(output_root: Path, task_id: str) -> Path:
    return output_root / task_id


def load_done_ids(task_dir: Path) -> set[str]:
    path = task_dir / DONE_IDS_FILENAME
    if not path.is_file():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                ids.add(value)
    return ids


def append_done_id(task_dir: Path, application_number: str) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / DONE_IDS_FILENAME
    with path.open("a", encoding="utf-8") as f:
        f.write(application_number)
        f.write("\n")
        f.flush()


def next_shard_index(task_dir: Path) -> int:
    task_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(task_dir.glob("part-*.jsonl.gz"))
    if not existing:
        return 0
    last = existing[-1].name  # part-00012.jsonl.gz
    try:
        return int(last.split("-")[1].split(".")[0]) + 1
    except (IndexError, ValueError):
        return len(existing)


def write_shard(task_dir: Path, index: int, records: list[dict[str, Any]]) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / f"part-{index:05d}.jsonl.gz"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    tmp.replace(path)
    return path


class ShardWriter:
    """Buffer records and flush gzip shards of ``shard_size``."""

    def __init__(self, task_dir: Path, *, shard_size: int = 100) -> None:
        if shard_size < 1:
            raise ValueError("shard_size must be >= 1")
        self.task_dir = task_dir
        self.shard_size = shard_size
        self._index = next_shard_index(task_dir)
        self._buffer: list[dict[str, Any]] = []
        self.n_written = 0
        self.written_paths: list[Path] = []

    def add(self, record: dict[str, Any]) -> Path | None:
        self._buffer.append(record)
        self.n_written += 1
        if len(self._buffer) >= self.shard_size:
            return self.flush()
        return None

    def flush(self) -> Path | None:
        if not self._buffer:
            return None
        path = write_shard(self.task_dir, self._index, self._buffer)
        self.written_paths.append(path)
        self._index += 1
        self._buffer.clear()
        return path


def make_record(
    *,
    task: str,
    application_number: str,
    instruction: str,
    input_text: str,
    output_text: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": task,
        "application_number": application_number,
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
        "meta": meta,
    }
