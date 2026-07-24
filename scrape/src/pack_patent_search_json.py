"""Pack per-application Patent Search ``*.json`` files into ``part-*.jsonl.gz``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from jsonl_gz import (
    FETCHED_IDS_FILENAME,
    ShardWriter,
    iter_jsonl_gz_shards,
    iter_open_jsonl_shards,
    write_fetched_ids,
)

logger = logging.getLogger(__name__)

SUMMARY_FILENAME = "summary.json"
DEFAULT_SHARD_SIZE = 1000


def iter_legacy_json_paths(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.glob("*.json")
        if not p.name.startswith("._") and p.name != SUMMARY_FILENAME
    )


def load_legacy_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("skip %s: %s", path.name, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("skip %s: root is not an object", path.name)
        return None
    return data


def _clear_shards(directory: Path) -> None:
    for path in iter_jsonl_gz_shards(directory):
        path.unlink()
    for path in iter_open_jsonl_shards(directory):
        path.unlink()
    ids_path = directory / FETCHED_IDS_FILENAME
    if ids_path.is_file():
        ids_path.unlink()


def pack_directory(
    directory: Path,
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
    force: bool = False,
) -> int:
    if not directory.is_dir():
        logger.error("not a directory: %s", directory)
        return 1

    existing_gz = iter_jsonl_gz_shards(directory)
    existing_open = iter_open_jsonl_shards(directory)
    if (existing_gz or existing_open) and not force:
        logger.error(
            "refusing to overwrite existing shards in %s "
            "(%s .jsonl.gz, %s open .jsonl). Pass --force to replace.",
            directory,
            len(existing_gz),
            len(existing_open),
        )
        return 1

    if force and (existing_gz or existing_open):
        logger.warning("removing existing shards in %s", directory)
        _clear_shards(directory)

    paths = iter_legacy_json_paths(directory)
    if not paths:
        logger.error("no legacy *.json files found in %s", directory)
        return 1

    written = 0
    skipped = 0
    ids: list[str] = []
    with ShardWriter(
        directory, shard_size, start_index=0, resume_open=False
    ) as writer:
        for path in paths:
            data = load_legacy_json(path)
            if data is None:
                skipped += 1
                continue
            app = data.get("application_number")
            if not isinstance(app, str) or not app.strip():
                app = path.stem
                data = {**data, "application_number": app}
            app = app.strip()
            writer.write(data)
            ids.append(app)
            written += 1

    write_fetched_ids(directory, ids)
    logger.info(
        "packed dir=%s legacy_json=%s written=%s skipped=%s shard_size=%s "
        "fetched_ids=%s",
        directory,
        len(paths),
        written,
        skipped,
        shard_size,
        len(ids),
    )
    logger.info(
        "Source *.json files were left in place. After verifying shard "
        "record counts, it is safe to delete the per-application *.json files."
    )
    return 0 if written else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Pack per-application patent_search *.json files into "
            "part-*.jsonl.gz shards + fetched_ids.txt."
        )
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing legacy {application_number}.json files",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help=f"Records per shard (default: {DEFAULT_SHARD_SIZE})",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Replace existing part-*.jsonl.gz / open .jsonl / fetched_ids.txt",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    if args.shard_size < 1:
        logger.error("--shard-size must be >= 1")
        return 1
    return pack_directory(
        args.input_dir.expanduser().resolve(),
        shard_size=args.shard_size,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
