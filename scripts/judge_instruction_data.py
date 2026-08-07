#!/usr/bin/env python3
"""Thin runner: Mode 2 LLM-as-a-judge (dataset-validation/)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dataset-validation" / "src"))

from run_llm_judge import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
