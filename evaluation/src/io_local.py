"""JSONL shard IO and resume helpers for evaluation."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator

from common import DV_SRC  # noqa: F401  — puts dataset-validation/src on sys.path
from io_util import iter_task_records, write_jsonl_gz  # dataset-validation

DONE_IDS_FILENAME = "done_ids.txt"


def iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        for line in f:
            text = line.strip()
            if not text:
                continue
            rec = json.loads(text)
            if isinstance(rec, dict):
                yield rec


def load_split_records(split_dir: Path, split: str) -> list[dict[str, Any]]:
    path = split_dir / f"{split}.jsonl.gz"
    if not path.is_file():
        return []
    return list(iter_jsonl_gz(path))


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


def next_shard_index(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("part-*.jsonl.gz"))
    if not existing:
        return 0
    try:
        return int(existing[-1].name.split("-")[1].split(".")[0]) + 1
    except (IndexError, ValueError):
        return len(existing)


class DurableShardWriter:
    """Rewrite the in-progress shard on each add so done_ids stay consistent."""

    def __init__(self, out_dir: Path, *, shard_size: int = 100) -> None:
        if shard_size < 1:
            raise ValueError("shard_size must be >= 1")
        self.out_dir = out_dir
        self.shard_size = shard_size
        self._index = next_shard_index(out_dir)
        self._buffer: list[dict[str, Any]] = []
        self.n_written = 0
        self.paths: list[Path] = []

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


__all__ = [
    "DONE_IDS_FILENAME",
    "DurableShardWriter",
    "append_done_id",
    "iter_jsonl_gz",
    "iter_task_records",
    "load_done_ids",
    "load_split_records",
    "write_jsonl_gz",
]
