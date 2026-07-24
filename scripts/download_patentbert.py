#!/usr/bin/env python3
"""Thin runner: download PatentBERT checkpoint into classification/models/patentbert."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "classification" / "src"))

from download_patentbert import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
