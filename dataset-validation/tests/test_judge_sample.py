"""Unit tests for ID pinning helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from judge_sample import load_ids_file, records_for_ids, remaining_records  # noqa: E402


class IdsFileTests(unittest.TestCase):
    def test_load_ids_file_preserves_order_drops_dupes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.txt"
            path.write_text("a\n\nb\na\nc\n", encoding="utf-8")
            self.assertEqual(load_ids_file(path), ["a", "b", "c"])

    def test_records_for_ids_order_and_missing(self) -> None:
        records = [
            {"application_number": "b", "task": "mrc"},
            {"application_number": "a", "task": "mrc"},
            {"application_number": "a", "task": "mrc"},
        ]
        matched, missing = records_for_ids(records, ["a", "z", "b"], skip_ids={"b"})
        self.assertEqual([r["application_number"] for r in matched], ["a"])
        self.assertEqual(missing, ["z"])

    def test_remaining_records_preserves_order_skips_done(self) -> None:
        records = [
            {"application_number": "a"},
            {"application_number": "b"},
            {"application_number": "a"},
            {"application_number": "c"},
        ]
        out = remaining_records(records, skip_ids={"b"})
        self.assertEqual([r["application_number"] for r in out], ["a", "c"])


if __name__ == "__main__":
    unittest.main()
