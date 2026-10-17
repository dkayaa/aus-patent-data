"""Application-level splits from meta.filed_date."""

from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "evaluation" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from split import (  # noqa: E402
    application_dates,
    assign_splits,
    main,
    record_application_number,
    record_filed_date,
)


def _row(app: str, filed: str | None, hop: int) -> dict:
    return {
        "task": "ipc_lcpn",
        "application_number": app,
        "instruction": "pick",
        "input": f"hop {hop}",
        "output": "A",
        "meta": {
            "filed_date": filed,
            "hop_index": hop,
        },
    }


class SplitAssignmentTests(unittest.TestCase):
    def test_same_application_stays_together(self) -> None:
        rows = [
            _row("100", "2024-06-01", 0),
            _row("100", "2024-06-01", 1),
            _row("200", "2020-01-01", 0),
            _row("200", "2020-01-01", 1),
        ]
        dates, missing_app, conflicts = application_dates(rows)
        split_of = assign_splits(
            dates,
            cutoff=date(2024, 1, 1),
            test_fraction=1.0,
            train_val_ratio=1.0,
            seed=42,
        )
        self.assertEqual(missing_app, 0)
        self.assertEqual(conflicts, 0)
        self.assertEqual(split_of["100"], "test")
        self.assertEqual(split_of["200"], "train")
        sides = {record_application_number(row): split_of[record_application_number(row)] for row in rows}
        self.assertEqual(sides["100"], "test")
        self.assertEqual(sides["200"], "train")

    def test_undated_filing_is_not_in_test(self) -> None:
        rows = [_row("100", None, 0), _row("100", None, 1)]
        dates, _, _ = application_dates(rows)
        split_of = assign_splits(
            dates,
            cutoff=date(2024, 1, 1),
            test_fraction=1.0,
            train_val_ratio=1.0,
            seed=1,
        )
        self.assertEqual(split_of["100"], "train")
        self.assertIsNone(record_filed_date(rows[0]))

    def test_meta_application_number_groups_rows(self) -> None:
        row = {
            "application_number": "",
            "meta": {"application_number": "300", "filed_date": "2024-02-02"},
        }
        self.assertEqual(record_application_number(row), "300")

    def test_cli_writes_one_split_per_application(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = root / "ipc_lcpn" / "run-a"
            shards.mkdir(parents=True)
            rows = [
                _row("100", "2024-06-01", 0),
                _row("100", "2024-06-01", 1),
                _row("200", "2019-05-01", 0),
            ]
            (shards / "part-00000.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            config = root / "baselines.yaml"
            config.write_text(
                "\n".join(
                    [
                        "paths:",
                        f"  derived_root: {root}",
                        f"  output_root: {root / 'evaluation'}",
                        "split:",
                        "  seed: 1",
                        '  test_min_filed_date: "2024-01-01"',
                        "  test_fraction: 1.0",
                        "  train_val_ratio: 1.0",
                        "fewshot:",
                        "  k: 1",
                        "  exemplar_input_chars: 20",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            code = main(
                ["--config", str(config), "--task", "ipc_lcpn", "--run", "run-a"]
            )
            self.assertEqual(code, 0)
            out = root / "evaluation" / "splits" / "run-a" / "ipc_lcpn"
            test_rows = [
                json.loads(line)
                for line in gzip.open(out / "test.jsonl.gz", "rt", encoding="utf-8")
            ]
            train_rows = [
                json.loads(line)
                for line in gzip.open(out / "train.jsonl.gz", "rt", encoding="utf-8")
            ]
            self.assertEqual(
                [row["meta"]["hop_index"] for row in test_rows],
                [0, 1],
            )
            self.assertEqual(train_rows[0]["application_number"], "200")


if __name__ == "__main__":
    unittest.main()
