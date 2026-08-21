"""Unit tests for human↔judge agreement helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_calibrate_judge import (  # noqa: E402
    agreement_for_threshold,
    cohens_kappa,
    human_accept_to_bool,
    pair_human_judge,
)


class KappaTests(unittest.TestCase):
    def test_perfect_agreement(self) -> None:
        human = [True, True, False, False]
        pred = [True, True, False, False]
        self.assertEqual(cohens_kappa(human, pred), 1.0)

    def test_human_accept_mapping(self) -> None:
        self.assertIs(human_accept_to_bool("yes"), True)
        self.assertIs(human_accept_to_bool("fix"), False)
        self.assertIs(human_accept_to_bool("no"), False)
        self.assertIsNone(human_accept_to_bool(""))

    def test_threshold_sweep_changes_pass(self) -> None:
        pairs = [(True, 3), (True, 4), (False, 2)]
        at3 = agreement_for_threshold(pairs, pass_score_min=3)
        at4 = agreement_for_threshold(pairs, pass_score_min=4)
        self.assertEqual(at3["confusion"]["human_accept_judge_pass"], 2)
        self.assertEqual(at4["confusion"]["human_accept_judge_pass"], 1)

    def test_pair_skips_unmatched_and_other_tasks(self) -> None:
        judged = [
            {
                "application_number": "1",
                "meta": {"llm_judge": {"score": 5, "pass": True}},
            }
        ]
        human = [
            {"application_number": "1", "task": "mrc", "accept": "yes"},
            {"application_number": "1", "task": "ipc_reasoning", "accept": "no"},
            {"application_number": "9", "task": "mrc", "accept": "yes"},
        ]
        pairs = pair_human_judge(judged, human, task="mrc")
        self.assertEqual(pairs, [(True, 5)])


if __name__ == "__main__":
    unittest.main()
