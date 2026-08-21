"""Generator-scoped paths under data/derived/instruction_generation/.

Holdings layout::

    {root}/
      _pools/                 # shared instruction phrasings (not per-model)
      {model_slug}/
        manifest.json
        <task>/part-*.jsonl.gz
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

POOLS_DIRNAME = "_pools"
MANIFEST_FILENAME = "manifest.json"

# Task folder names that used to live directly under the holdings root.
_LEGACY_TASK_DIRS = ("abstract_drafting", "ipc_reasoning", "mrc")
_SLUG_UNSAFE = re.compile(r"[^a-z0-9._-]+")


def model_slug(model: str) -> str:
    """Filesystem-safe generator id.

    ``anthropic/claude-sonnet-4.6`` → ``anthropic-claude-sonnet-4.6``
    ``llama3.1:8b`` → ``llama3.1-8b``
    """
    text = (model or "").strip().lower()
    if not text:
        raise ValueError("model id is empty")
    for ch in ("/", "\\", ":", " "):
        text = text.replace(ch, "-")
    slug = _SLUG_UNSAFE.sub("-", text)
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")
    if not slug or slug.startswith("_"):
        raise ValueError(f"invalid model slug from {model!r}")
    return slug


def pools_dir(holdings_root: Path) -> Path:
    return holdings_root / POOLS_DIRNAME


def generator_dir(holdings_root: Path, model: str) -> Path:
    return holdings_root / model_slug(model)


def iter_generator_dirs(holdings_root: Path) -> list[Path]:
    if not holdings_root.is_dir():
        return []
    found: list[Path] = []
    for path in sorted(holdings_root.iterdir()):
        if path.is_dir() and not path.name.startswith("_"):
            found.append(path)
    return found


def legacy_task_dirs(holdings_root: Path) -> list[str]:
    if not holdings_root.is_dir():
        return []
    return [name for name in _LEGACY_TASK_DIRS if (holdings_root / name).is_dir()]


def resolve_generator_dir(
    holdings_root: Path,
    *,
    generator: str | None = None,
) -> Path:
    """Return the generator directory, or raise if it cannot be chosen uniquely."""
    leftover = legacy_task_dirs(holdings_root)
    if leftover:
        raise ValueError(
            f"legacy task dirs still at {holdings_root}: {', '.join(leftover)}. "
            f"Move them under {holdings_root}/<model-slug>/."
        )
    if generator:
        return generator_dir(holdings_root, generator)
    found = iter_generator_dirs(holdings_root)
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(
            f"no generator dirs under {holdings_root}; pass --generator MODEL"
        )
    names = ", ".join(path.name for path in found)
    raise ValueError(
        f"multiple generators under {holdings_root}: {names}. "
        "Pass --generator MODEL (id or slug)."
    )


def write_manifest(dest: Path, payload: dict[str, Any]) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / MANIFEST_FILENAME
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            loaded = {}
        if isinstance(loaded, dict):
            existing = loaded
    merged = {**existing, **payload}
    merged["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path
