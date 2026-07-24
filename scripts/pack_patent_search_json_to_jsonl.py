#!/usr/bin/env python3
"""Thin runner: pack legacy patent_search *.json → part-*.jsonl.gz shards."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scrape" / "src"))

from pack_patent_search_json import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
