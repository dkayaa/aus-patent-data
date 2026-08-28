#!/usr/bin/env python3
"""Thin runner: cheap/full LLM-as-a-judge (dataset-validation/)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dataset-validation" / "src"))

from run_llm_judge import main as judge_main  # noqa: E402

CHEAP_CONFIG = REPO_ROOT / "dataset-validation" / "config" / "cheap_judge.yaml"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--config" not in args:
        args = ["--config", str(CHEAP_CONFIG), *args]
    return judge_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
