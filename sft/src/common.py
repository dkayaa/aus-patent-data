"""Shared paths and dataset ids for the SFT stage."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

IG_SRC = REPO_ROOT / "instruction-generation" / "src"
DV_SRC = REPO_ROOT / "dataset-validation" / "src"
EVAL_SRC = REPO_ROOT / "evaluation" / "src"
SCRAPE_SRC = REPO_ROOT / "scrape" / "src"
for extra in (IG_SRC, DV_SRC, EVAL_SRC, SCRAPE_SRC):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

DEFAULT_CONFIG = REPO_ROOT / "sft" / "config" / "sft.yaml"
# Fallback if config omits `generator:` (keep in sync with sft/config/sft.yaml).
DEFAULT_GENERATOR = "qwen/qwen3-235b-a22b-2507"

# Flat SFT dataset ids (same directory level under data/derived/sft/{slug}/).
SFT_DATASETS = (
    "ipc_reasoning_full",
    "ipc_reasoning_classification_only",
    "abstract_drafting",
    "mrc",
)

# SFT dataset id → eval split task folder name.
DATASET_TO_SPLIT_TASK: dict[str, str] = {
    "ipc_reasoning_full": "ipc_reasoning",
    "ipc_reasoning_classification_only": "ipc_reasoning",
    "abstract_drafting": "abstract_drafting",
    "mrc": "mrc",
}

SPLITS = ("train", "val", "test")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def user_turn(instruction: str, input_text: str) -> str:
    inst = (instruction or "").strip()
    inp = (input_text or "").strip()
    if inst and inp:
        return f"{inst}\n\n{inp}"
    return inst or inp
