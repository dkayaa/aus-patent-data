#!/usr/bin/env python3
"""Thin runner: sample ipc_reasoning apps with a per-primary_ipc cap."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "instruction-generation" / "src"))

from sample_ipc_apps import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
