#!/usr/bin/env python3
"""Thin runner: QLoRA SFT on a prepared flat dataset (CUDA)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sft" / "src"))

from run_train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
