"""JSONL / JSONL.GZ shard helpers for Patent Search interim storage."""

from __future__ import annotations

import gzip
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

FETCHED_IDS_FILENAME = "fetched_ids.txt"
_PART_RE = re.compile(r"^part-(\d+)\.(jsonl(?:\.gz)?)$")


def shard_stem(index: int) -> str:
    return f"part-{index:05d}"


def shard_jsonl_path(output_dir: Path, index: int) -> Path:
    return output_dir / f"{shard_stem(index)}.jsonl"


def shard_jsonl_gz_path(output_dir: Path, index: int) -> Path:
    return output_dir / f"{shard_stem(index)}.jsonl.gz"


def parse_shard_index(path: Path) -> int | None:
    match = _PART_RE.match(path.name)
    if not match:
        return None
    return int(match.group(1))


def iter_jsonl_gz_shards(directory: Path) -> list[Path]:
    """Sorted finalized ``part-*.jsonl.gz`` shards."""
    if not directory.is_dir():
        return []
    shards = [
        p
        for p in directory.glob("part-*.jsonl.gz")
        if not p.name.startswith("._") and parse_shard_index(p) is not None
    ]
    return sorted(shards, key=lambda p: parse_shard_index(p) or 0)


def iter_open_jsonl_shards(directory: Path) -> list[Path]:
    """Sorted open (uncompressed) ``part-*.jsonl`` shards."""
    if not directory.is_dir():
        return []
    shards = [
        p
        for p in directory.glob("part-*.jsonl")
        if not p.name.startswith("._") and parse_shard_index(p) is not None
    ]
    return sorted(shards, key=lambda p: parse_shard_index(p) or 0)


def next_shard_index(directory: Path) -> int:
    """Next unused shard index from existing ``part-NNNNN.*`` names."""
    indices: list[int] = []
    if not directory.is_dir():
        return 0
    for path in directory.iterdir():
        idx = parse_shard_index(path)
        if idx is not None:
            indices.append(idx)
    return (max(indices) + 1) if indices else 0


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSON objects from a ``.jsonl`` or ``.jsonl.gz`` file."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                logger.warning("skip %s:%s: %s", path.name, line_no, exc)
                continue
            if not isinstance(data, dict):
                logger.warning(
                    "skip %s:%s: root is not an object", path.name, line_no
                )
                continue
            yield data


def iter_shard_records(
    directory: Path, *, include_open_jsonl: bool = False
) -> Iterator[dict[str, Any]]:
    """Yield records from finalized shards, optionally then open ``.jsonl``."""
    for path in iter_jsonl_gz_shards(directory):
        yield from iter_records(path)
    if include_open_jsonl:
        for path in iter_open_jsonl_shards(directory):
            yield from iter_records(path)


def load_fetched_ids(output_dir: Path) -> set[str]:
    path = output_dir / FETCHED_IDS_FILENAME
    if not path.is_file():
        return set()
    ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                ids.add(value)
    return ids


def append_fetched_id(output_dir: Path, application_number: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / FETCHED_IDS_FILENAME
    with path.open("a", encoding="utf-8") as f:
        f.write(application_number)
        f.write("\n")
        f.flush()


def write_fetched_ids(output_dir: Path, ids: set[str] | list[str]) -> Path:
    """Rewrite ``fetched_ids.txt`` from a full set (used by packer)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / FETCHED_IDS_FILENAME
    tmp = path.with_suffix(".txt.tmp")
    ordered = sorted(ids) if isinstance(ids, set) else list(ids)
    with tmp.open("w", encoding="utf-8") as f:
        for application_number in ordered:
            f.write(application_number)
            f.write("\n")
    tmp.replace(path)
    return path


def recover_ids_from_open_shards(output_dir: Path, known: set[str]) -> set[str]:
    """Add application_numbers found in open ``.jsonl`` shards to ``known``."""
    recovered = set(known)
    for path in iter_open_jsonl_shards(output_dir):
        for record in iter_records(path):
            app = record.get("application_number")
            if isinstance(app, str) and app.strip():
                recovered.add(app.strip())
    return recovered


def count_jsonl_lines(path: Path) -> int:
    count = 0
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[operator]
        for line in f:
            if line.strip():
                count += 1
    return count


def gzip_jsonl_file(jsonl_path: Path) -> Path:
    """Atomically compress ``part-N.jsonl`` → ``part-N.jsonl.gz`` and remove source."""
    if not jsonl_path.is_file():
        raise FileNotFoundError(jsonl_path)
    gz_path = Path(str(jsonl_path) + ".gz")
    tmp = gz_path.with_suffix(gz_path.suffix + ".tmp")
    with jsonl_path.open("rb") as src, gzip.open(tmp, "wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    tmp.replace(gz_path)
    jsonl_path.unlink()
    return gz_path


def write_jsonl_gz_records(path: Path, records: list[dict[str, Any]]) -> Path:
    """Write an entire shard as ``.jsonl.gz`` atomically (for mirrored clean shards)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    tmp.replace(path)
    return path


class ShardWriter:
    """Append compact JSON lines to open ``.jsonl``; gzip when full or on close."""

    def __init__(
        self,
        output_dir: Path,
        shard_size: int,
        *,
        start_index: int | None = None,
        resume_open: bool = True,
    ) -> None:
        if shard_size < 1:
            raise ValueError("shard_size must be >= 1")
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._fh: Any | None = None
        self._index = 0
        self._count_in_shard = 0
        self._closed = False

        open_shards = iter_open_jsonl_shards(output_dir)
        if resume_open and open_shards:
            # Resume the highest-index open shard.
            open_path = open_shards[-1]
            idx = parse_shard_index(open_path)
            assert idx is not None
            self._index = idx
            self._count_in_shard = count_jsonl_lines(open_path)
            if self._count_in_shard >= self.shard_size:
                gzip_jsonl_file(open_path)
                self._index = idx + 1
                self._count_in_shard = 0
                self._open_new()
            else:
                self._fh = open_path.open("a", encoding="utf-8")
        else:
            self._index = (
                next_shard_index(output_dir) if start_index is None else start_index
            )
            self._open_new()

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def count_in_shard(self) -> int:
        return self._count_in_shard

    def _path(self) -> Path:
        return shard_jsonl_path(self.output_dir, self._index)

    def _open_new(self) -> None:
        path = self._path()
        if path.exists() or shard_jsonl_gz_path(self.output_dir, self._index).exists():
            raise FileExistsError(
                f"shard already exists for index {self._index}: {path}"
            )
        self._fh = path.open("w", encoding="utf-8")
        self._count_in_shard = 0

    def write(self, record: dict[str, Any]) -> Path:
        if self._closed:
            raise RuntimeError("ShardWriter is closed")
        if self._fh is None:
            self._open_new()
        assert self._fh is not None
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        self._fh.write(line)
        self._fh.write("\n")
        self._fh.flush()
        self._count_in_shard += 1
        current = self._path()
        if self._count_in_shard >= self.shard_size:
            self._finalize_current()
            self._index += 1
            self._open_new()
        return current

    def _finalize_current(self) -> Path | None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        path = self._path()
        if not path.is_file() or self._count_in_shard == 0:
            if path.is_file() and self._count_in_shard == 0:
                path.unlink()
            return None
        return gzip_jsonl_file(path)

    def close(self) -> Path | None:
        if self._closed:
            return None
        self._closed = True
        return self._finalize_current()

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
