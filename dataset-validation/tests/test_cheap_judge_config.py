"""Cheap-judge config isolation from Mode 2."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_llm_judge import main as judge_main  # noqa: E402

CFG = Path(__file__).resolve().parents[1] / "config"


class CheapJudgeConfigTests(unittest.TestCase):
    def test_cheap_yaml_writes_cheap_judge_not_llm_judge(self) -> None:
        data = yaml.safe_load((CFG / "cheap_judge.yaml").read_text(encoding="utf-8"))
        self.assertEqual(data["paths"]["output_subdir"], "cheap_judge")
        self.assertIsNone(data["judge"]["sample_size"])
        self.assertEqual(data["llm"]["provider"], "openrouter")
        self.assertEqual(data["llm"]["model"], "anthropic/claude-haiku-4.5")
        self.assertEqual(float(data["llm"]["temperature"]), 0.0)
        self.assertGreaterEqual(data["judge"]["truncate_chars"], 64000)
        self.assertTrue(data["llm"]["json_object"])

    def test_mode2_yaml_still_writes_llm_judge(self) -> None:
        data = yaml.safe_load((CFG / "llm_judge.yaml").read_text(encoding="utf-8"))
        self.assertEqual(data["paths"]["output_subdir"], "llm_judge")
        self.assertEqual(data["judge"]["sample_size"], 50)

    def test_full_incompatible_with_limit(self) -> None:
        self.assertEqual(
            judge_main(["--task", "mrc", "--full", "--limit", "10"]),
            2,
        )


if __name__ == "__main__":
    unittest.main()
