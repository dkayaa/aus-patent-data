# Tests for prompts, parse, and walk (walk uses a fake model, real IPC tree).

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "classification" / "src"
for path in (REPO, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ipc.accessor import IPCAccessor, IPCNode  # noqa: E402

from ipc_lcpn.config import load_config  # noqa: E402
from ipc_lcpn.hops import hop_messages, menu_codes, score_hops  # noqa: E402
from ipc_lcpn.parse import HopParseError, parse_hop_response  # noqa: E402
from ipc_lcpn.prompts import build_hop_messages, choice_lines  # noqa: E402
from ipc_lcpn.walk import walk_ipc  # noqa: E402


def _node(node_id: str, title: str, kind: str = "c") -> IPCNode:
    return IPCNode(
        node_id=node_id,
        title=title,
        definition=None,
        kind=kind,
        entry_type="K",
    )


class ParseTests(unittest.TestCase):
    def test_parse_child_with_justification(self) -> None:
        text = "Spades are hand tools.\nClassification: A01B1/02\n"
        parsed = parse_hop_response(
            text,
            allowed_codes={"A01B1/02", "A01B1/04"},
            stop_token="STOP",
            justify=True,
        )
        self.assertEqual(parsed.choice, "A01B1/02")
        self.assertFalse(parsed.is_stop)
        self.assertIn("Spades", parsed.justification)

    def test_parse_stop(self) -> None:
        text = "None fit.\nClassification: STOP\n"
        parsed = parse_hop_response(
            text,
            allowed_codes={"A01B1/02"},
            stop_token="STOP",
            justify=True,
        )
        self.assertTrue(parsed.is_stop)
        self.assertEqual(parsed.choice, "STOP")

    def test_reject_off_menu(self) -> None:
        with self.assertRaises(HopParseError):
            parse_hop_response(
                "Classification: Z99Z99/00\n",
                allowed_codes={"A01B1/02"},
                stop_token="STOP",
                justify=True,
            )


class PromptTests(unittest.TestCase):
    def test_menu_includes_stop(self) -> None:
        kids = [_node("A01", "AGRICULTURE"), _node("A21", "BAKING")]
        lines = choice_lines(kids, stop_token="STOP")
        self.assertIn("1. A01 - AGRICULTURE", lines)
        self.assertIn("3. STOP -", lines)

    def test_messages_shape(self) -> None:
        kids = [_node("A", "HUMAN NECESSITIES", kind="s")]
        messages = build_hop_messages(
            document="Abstract:\nHello",
            current_id="<ROOT>",
            titles="<ROOT>",
            children=kids,
            stop_token="STOP",
            justify=True,
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        user = messages[1]["content"]
        self.assertIn("Current node: <ROOT>", user)
        self.assertIn("Classification:", user)
        self.assertIn("STOP", user)


class WalkMockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.accessor = IPCAccessor()
        cls.accessor.init()
        cls.cfg = load_config()

    def test_walk_stops_on_stop_token(self) -> None:
        replies = [
            "Section A fits.\nClassification: A\n",
            "Stop here.\nClassification: <STOP>\n",
        ]

        class Scripted:
            def __init__(self) -> None:
                self.i = 0

            def chat(self, messages, *, temperature=None, max_tokens=None):
                text = replies[self.i]
                self.i += 1
                return text

        result = walk_ipc(
            accessor=self.accessor,
            client=Scripted(),
            cfg=self.cfg,
            abstract="A farming tool.",
            claim_1="A spade.",
        )
        self.assertEqual(result.path, ["A"])
        self.assertEqual(result.stop_reason, "stop")
        self.assertEqual(len(result.hops), 2)


def _hop(gold: str, *, justify: bool = False) -> dict:
    return {
        "application_number": "2020000001",
        "instruction": "Pick the next symbol.",
        "input": (
            "Claim 1:\nA spade\n\n"
            "Current node: A01B\n"
            "Titles: SOIL\n\n"
            "Children:\n"
            "1. A01B1/02 - SPADES\n"
            "2. A01B1/20 - PADDED\n"
            "3. <STOP> - none of these children; remain at the current place"
        ),
        "meta": {
            "gold_node_id": gold,
            "hop_index": 0,
            "justify": justify,
            "stop_token": "<STOP>",
        },
    }


class IndependentHopTests(unittest.TestCase):
    def test_messages_match_training_rows(self) -> None:
        messages = hop_messages(_hop("A01B1/02"))
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertTrue(messages[0]["content"].startswith("Pick the next symbol.\n\nClaim 1:"))
        self.assertEqual(
            menu_codes(messages[0]["content"], stop_token="<STOP>"),
            {"A01B1/02", "A01B1/20"},
        )

    def test_each_hop_is_scored_on_its_own(self) -> None:
        replies = ["A01B1/02", "A01B1/20", "<STOP>", "not-a-code"]

        class Scripted:
            def __init__(self) -> None:
                self.i = 0
                self.seen: list[str] = []

            def chat(self, messages, *, temperature=None, max_tokens=None):
                self.seen.append(messages[0]["content"])
                text = replies[self.i]
                self.i += 1
                return text

        client = Scripted()
        result = score_hops(
            [
                _hop("A01B1/02"),
                _hop("A01B1/02"),
                _hop("<STOP>"),
                _hop("A01B1/02"),
            ],
            client=client,
            cfg=load_config(),
        )
        self.assertEqual([score.correct for score in result.scores], [True, False, True, False])
        self.assertEqual(result.n_correct, 2)
        self.assertEqual(result.n_parse_errors, 1)
        self.assertEqual(len(client.seen), 4)
        self.assertNotIn("system", client.seen[0])


if __name__ == "__main__":
    unittest.main()
