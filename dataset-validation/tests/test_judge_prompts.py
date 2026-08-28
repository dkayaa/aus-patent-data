"""Unit tests for Mode 2 judge payload, tags, and pass derivation."""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from judge_prompts import (  # noqa: E402
    build_judge_messages,
    build_judge_payload,
    normalize_judge_result,
    wipo_fields_for_record,
)


@dataclass
class _FakeEntry:
    title: str
    definition_statement: str | None = None
    scheme_note: str | None = None


class _FakeLookup:
    def __init__(self, by_code: dict[str, _FakeEntry]) -> None:
        self._by_code = by_code

    def get(self, code: str) -> _FakeEntry | None:
        key = code.strip().upper().replace(" ", "")
        return self._by_code.get(key)


def _ipc_record() -> dict:
    return {
        "task": "ipc_reasoning",
        "application_number": "2024396373",
        "instruction": "Justify the IPC.",
        "input": "Abstract:\nA widget.\n\nClaims:\n1. A widget",
        "output": "Classification: G06F17/00\nJustification: Because widgets.",
        "meta": {
            "primary_ipc": "G06F17/00",
            "ipc_title": "stale title from generator",
            "model": "meta-llama/llama-3.3-70b-instruct",
            "provider": "openrouter",
            "has_definition_entry": True,
        },
    }


class NormalizeJudgeResultTests(unittest.TestCase):
    def test_score_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_judge_result({"score": 0, "rationale": "x"}, task="mrc")
        with self.assertRaises(ValueError):
            normalize_judge_result({"score": 6, "rationale": "x"}, task="mrc")

    def test_pass_derived_from_score_ignores_model_pass(self) -> None:
        low = normalize_judge_result(
            {"score": 3, "pass": True, "rationale": "ok", "failure_tags": []},
            task="mrc",
            pass_score_min=4,
        )
        self.assertFalse(low["pass"])
        self.assertEqual(low["score"], 3)

        high = normalize_judge_result(
            {"score": 4, "pass": False, "rationale": "ok", "failure_tags": []},
            task="mrc",
            pass_score_min=4,
        )
        self.assertTrue(high["pass"])

    def test_unknown_tags_become_other(self) -> None:
        result = normalize_judge_result(
            {
                "score": 2,
                "rationale": "bad",
                "failure_tags": ["insufficient_detail", "topic_mismatch"],
            },
            task="abstract_drafting",
        )
        self.assertEqual(result["failure_tags"], ["other", "topic_mismatch"])

    def test_forbidden_ipc_tags_stripped(self) -> None:
        result = normalize_judge_result(
            {
                "score": 2,
                "rationale": "reclassified",
                "failure_tags": [
                    "wrong_ipc",
                    "obsolete_ipc_code",
                    "unfaithful_to_claims",
                    "classification_mismatch",
                ],
            },
            task="ipc_reasoning",
        )
        self.assertEqual(result["failure_tags"], ["unfaithful_to_claims"])

    def test_duplicate_and_spaced_tags_normalized(self) -> None:
        result = normalize_judge_result(
            {
                "score": 2,
                "rationale": "x",
                "failure_tags": ["Unfaithful To Claims", "unfaithful_to_claims"],
            },
            task="ipc_reasoning",
        )
        self.assertEqual(result["failure_tags"], ["unfaithful_to_claims"])


class PayloadTests(unittest.TestCase):
    def test_strips_generator_identity(self) -> None:
        payload = build_judge_payload(_ipc_record(), truncate_chars=12000)
        blob = json.dumps(payload)
        self.assertNotIn("meta-llama/llama-3.3-70b-instruct", blob)
        self.assertNotIn("openrouter", blob)
        self.assertNotIn("model", payload["meta"])
        self.assertNotIn("provider", payload["meta"])
        self.assertEqual(payload["meta"]["primary_ipc"], "G06F17/00")

    def test_wipo_definition_overrides_stale_title(self) -> None:
        wipo = {
            "ipc_title": "Catalog title",
            "definition_statement": "Official definition from WIPO.",
        }
        payload = build_judge_payload(
            _ipc_record(), truncate_chars=12000, wipo=wipo
        )
        self.assertEqual(payload["meta"]["ipc_title"], "Catalog title")
        self.assertEqual(
            payload["meta"]["definition_statement"],
            "Official definition from WIPO.",
        )
        self.assertEqual(payload["meta"]["definition_source"], "wipo_catalog")

    def test_wipo_fields_for_record_uses_lookup(self) -> None:
        lookup = _FakeLookup(
            {
                "G06F17/00": _FakeEntry(
                    title="Digital computing",
                    definition_statement="Methods specially adapted…",
                )
            }
        )
        fields = wipo_fields_for_record(_ipc_record(), lookup)
        assert fields is not None
        self.assertEqual(fields["ipc_title"], "Digital computing")
        self.assertIn("specially adapted", fields["definition_statement"])

    def test_messages_are_rationale_first_and_grounded(self) -> None:
        lookup = _FakeLookup(
            {
                "G06F17/00": _FakeEntry(
                    title="Digital computing",
                    definition_statement="Catalog definition text.",
                )
            }
        )
        messages = build_judge_messages(
            _ipc_record(), truncate_chars=12000, ipc_lookup=lookup
        )
        self.assertEqual(messages[0]["role"], "system")
        user = messages[1]["content"]
        rationale_at = user.find('"rationale"')
        score_at = user.find('"score"')
        self.assertGreater(rationale_at, 0)
        self.assertGreater(score_at, rationale_at)
        self.assertIn("chain of thought", messages[0]["content"].lower())
        self.assertIn("before the score field", user)
        self.assertNotIn('"pass"', user.split("Example to evaluate:")[0])
        self.assertIn("Catalog definition text.", user)
        self.assertNotIn("meta-llama/llama-3.3-70b-instruct", user)

    def test_reasoning_alias_fills_rationale(self) -> None:
        result = normalize_judge_result(
            {"score": 4, "reasoning": "stepwise check", "failure_tags": []},
            task="mrc",
        )
        self.assertEqual(result["rationale"], "stepwise check")


if __name__ == "__main__":
    unittest.main()
