"""Read instruction JSONL shards (gz or plain) and write pass/reject shards."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scrape" / "src"))

from jsonl_gz import iter_records  # noqa: E402


def iter_task_records(task_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield records from part-*.jsonl.gz then part-*.jsonl (non-gz)."""
    gz_paths = sorted(task_dir.glob("part-*.jsonl.gz"))
    for path in gz_paths:
        yield from iter_records(path)

    plain = sorted(task_dir.glob("part-*.jsonl"))
    # Skip if a .gz sibling exists (already read).
    for path in plain:
        if path.with_suffix(path.suffix + ".gz").exists():
            continue
        # also skip part-00000.jsonl if part-00000.jsonl.gz naming differs
        if Path(str(path) + ".gz").exists():
            continue
        yield from iter_records(path)


def write_jsonl_gz(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    tmp.replace(path)
    return path


class ShardWriter:
    def __init__(self, out_dir: Path, *, shard_size: int = 100) -> None:
        self.out_dir = out_dir
        self.shard_size = shard_size
        self._index = 0
        self._buffer: list[dict[str, Any]] = []
        self.n_written = 0
        self.paths: list[Path] = []

    def add(self, record: dict[str, Any]) -> None:
        self._buffer.append(record)
        self.n_written += 1
        if len(self._buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        path = self.out_dir / f"part-{self._index:05d}.jsonl.gz"
        write_jsonl_gz(path, self._buffer)
        self.paths.append(path)
        self._index += 1
        self._buffer.clear()
