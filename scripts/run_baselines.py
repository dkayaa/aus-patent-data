#!/usr/bin/env python3
"""Thin runner: OpenRouter zero-shot / 3-shot baselines (evaluation/)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation" / "src"))

from run_baseline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
