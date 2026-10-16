#!/usr/bin/env python3
# Thin wrapper so we can run: python scripts/run_ipc_lcpn.py ...

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "classification" / "src"))
sys.path.insert(0, str(REPO_ROOT))

from ipc_lcpn.run import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
