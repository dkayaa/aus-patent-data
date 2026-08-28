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
    cascade_vs_frontier,
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


class CascadeTests(unittest.TestCase):
    def test_enrichment_positive_when_cheap_keeps_frontier_passes(self) -> None:
        def rec(app: str, score: int, passed: bool) -> dict:
            return {
                "application_number": app,
                "meta": {"llm_judge": {"score": score, "pass": passed}},
            }

        cheap = [
            rec("a", 4, True),
            rec("b", 4, True),
            rec("c", 2, False),
            rec("d", 2, False),
        ]
        frontier = [
            rec("a", 5, True),
            rec("b", 4, True),
            rec("c", 3, False),
            rec("d", 2, False),
        ]
        out = cascade_vs_frontier(cheap, frontier, pass_score_min=4)
        self.assertEqual(out["n_paired"], 4)
        self.assertEqual(out["frontier_pass_rate"], 0.5)
        self.assertEqual(out["frontier_pass_rate_given_cheap_pass"], 1.0)
        self.assertEqual(out["enrichment"], 0.5)
        self.assertEqual(out["n_false_kills"], 0)
        self.assertEqual(out["n_missed_junk"], 0)

    def test_false_kills_and_missed_junk(self) -> None:
        def rec(app: str, score: int, passed: bool) -> dict:
            return {
                "application_number": app,
                "meta": {"llm_judge": {"score": score, "pass": passed}},
            }

        cheap = [rec("keep", 4, True), rec("kill", 2, False)]
        frontier = [rec("keep", 2, False), rec("kill", 5, True)]
        out = cascade_vs_frontier(cheap, frontier, pass_score_min=4)
        self.assertEqual(out["n_false_kills"], 1)
        self.assertEqual(out["n_missed_junk"], 1)
        self.assertEqual(out["frontier_pass_rate_given_cheap_pass"], 0.0)


if __name__ == "__main__":
    unittest.main()
