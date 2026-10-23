# Tests for prompts, parse, and walk (walk uses a fake model, real IPC tree).

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
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
from ipc_lcpn.walk_eval import filings_from_hops, patent_text, score_walks  # noqa: E402


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

    def test_parse_back(self) -> None:
        text = "Wrong branch.\nClassification: <BACK>\n"
        parsed = parse_hop_response(
            text,
            allowed_codes={"A01"},
            stop_token="<STOP>",
            justify=True,
            back_token="<BACK>",
        )
        self.assertTrue(parsed.is_back)
        self.assertFalse(parsed.is_stop)
        self.assertEqual(parsed.choice, "<BACK>")

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

    def test_back_from_section_returns_to_root(self) -> None:
        replies = [
            "Classification: A\n",
            "Classification: <BACK>\n",
            "Classification: <STOP>\n",
        ]
        seen: list[str] = []

        class Scripted:
            def __init__(self) -> None:
                self.i = 0

            def chat(self, messages, *, temperature=None, max_tokens=None):
                seen.append(messages[-1]["content"])
                text = replies[self.i]
                self.i += 1
                return text

        result = walk_ipc(
            accessor=self.accessor,
            client=Scripted(),
            cfg=replace(self.cfg, back_token="<BACK>"),
            abstract="A farming tool.",
            claim_1="A spade.",
        )
        self.assertEqual(result.path, [])
        self.assertEqual(result.stop_reason, "stop")
        self.assertTrue(result.hops[1].parsed.is_back)
        self.assertIn("<BACK> - return to the parent", seen[1])
        third_children = seen[2].split("Children:\n", 1)[1]
        codes = [
            line.split(". ", 1)[1].split(" - ", 1)[0]
            for line in third_children.splitlines()
            if ". " in line and " - " in line
        ]
        self.assertNotIn("A", codes)
        self.assertIn("<STOP>", codes)
        self.assertNotIn("<BACK>", codes)
        section = self.accessor.get("A")
        self.assertIn("Titles: <ROOT>\n", seen[0])
        self.assertIn(f"Titles: {section.title.strip()}\n", seen[1])
        self.assertNotIn("Titles: <ROOT> >", seen[1])
        self.assertIn("Titles: <ROOT>\n", seen[2])

    def test_deeper_titles_omit_the_root_prefix(self) -> None:
        replies = [
            "Classification: A\n",
            "Classification: A01\n",
            "Classification: <STOP>\n",
        ]
        seen: list[str] = []

        class Scripted:
            def __init__(self) -> None:
                self.i = 0

            def chat(self, messages, *, temperature=None, max_tokens=None):
                seen.append(messages[-1]["content"])
                text = replies[self.i]
                self.i += 1
                return text

        result = walk_ipc(
            accessor=self.accessor,
            client=Scripted(),
            cfg=replace(self.cfg, back_token="<BACK>"),
            abstract="A farming tool.",
            claim_1="A spade.",
        )
        section = self.accessor.get("A")
        klass = self.accessor.get("A01")
        self.assertEqual(result.path, ["A", "A01"])
        self.assertIn(
            f"Titles: {section.title.strip()} > {klass.title.strip()}\n",
            seen[2],
        )
        self.assertNotIn("Titles: <ROOT> >", seen[2])

    def test_back_hides_every_child_left(self) -> None:
        replies = [
            "Classification: A\n",
            "Classification: <BACK>\n",
            "Classification: B\n",
            "Classification: <BACK>\n",
            "Classification: <STOP>\n",
        ]
        seen: list[str] = []

        class Scripted:
            def __init__(self) -> None:
                self.i = 0

            def chat(self, messages, *, temperature=None, max_tokens=None):
                seen.append(messages[-1]["content"])
                text = replies[self.i]
                self.i += 1
                return text

        result = walk_ipc(
            accessor=self.accessor,
            client=Scripted(),
            cfg=replace(self.cfg, back_token="<BACK>"),
            abstract="A farming tool.",
            claim_1="A spade.",
        )
        self.assertEqual(result.path, [])
        self.assertEqual(result.stop_reason, "stop")
        final_children = seen[-1].split("Children:\n", 1)[1]
        codes = [
            line.split(". ", 1)[1].split(" - ", 1)[0]
            for line in final_children.splitlines()
            if ". " in line and " - " in line
        ]
        self.assertNotIn("A", codes)
        self.assertNotIn("B", codes)
        self.assertIn("<STOP>", codes)


class WalkEvalTests(unittest.TestCase):
    def test_patent_text_drops_the_menu(self) -> None:
        abstract, claim = patent_text(
            {
                "input": (
                    "Abstract:\nA tool.\n\n"
                    "Claim 1:\nA spade.\n\n"
                    "Current node: <ROOT>\n"
                    "Children:\n1. A - HUMAN NECESSITIES"
                )
            }
        )
        self.assertEqual(abstract, "A tool.")
        self.assertEqual(claim, "A spade.")

    def test_one_filing_per_application(self) -> None:
        rows = [
            {"application_number": "1", "input": "Claim 1:\nfirst"},
            {"application_number": "1", "input": "Claim 1:\nsecond"},
            {"application_number": "2", "input": "Claim 1:\nother"},
        ]
        self.assertEqual(
            [row["input"] for row in filings_from_hops(rows)],
            ["Claim 1:\nfirst", "Claim 1:\nother"],
        )

    def test_exact_is_the_place_where_the_walk_stops(self) -> None:
        accessor = IPCAccessor()
        accessor.init()
        record = {
            "application_number": "2020000001",
            "input": "Claim 1:\nA spade.\n\nCurrent node: <ROOT>\n",
            "meta": {
                "primary_ipc": "A",
                "path_node_ids": ["<ROOT>", "A"],
            },
        }

        class Scripted:
            def __init__(self) -> None:
                self.i = 0

            def chat(self, messages, *, temperature=None, max_tokens=None):
                text = ["Classification: A\n", "Classification: <STOP>\n"][self.i]
                self.i += 1
                return text

        result = score_walks(
            [record, {**record, "meta": {**record["meta"], "hop_index": 1}}],
            accessor=accessor,
            client=Scripted(),
            cfg=replace(load_config(), justify=True, back_token="<BACK>"),
        )
        self.assertEqual(result.n_filings, 1)
        self.assertEqual(result.n_exact, 1)
        self.assertEqual(result.filings[0].predicted, "A")
        self.assertEqual(result.filings[0].path, ["A"])
        self.assertEqual(result.filings[0].stop_reason, "stop")


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

    def test_back_gold_matches_and_is_not_a_child_code(self) -> None:
        record = _hop("<BACK>")
        record["input"] = (
            record["input"] + "\n4. <BACK> - return to the parent"
        )
        record["meta"]["back_token"] = "<BACK>"
        self.assertEqual(
            menu_codes(record["input"], stop_token="<STOP>", back_token="<BACK>"),
            {"A01B1/02", "A01B1/20"},
        )

        class Scripted:
            def chat(self, messages, *, temperature=None, max_tokens=None):
                return "<BACK>"

        result = score_hops([record], client=Scripted(), cfg=load_config())
        self.assertEqual(result.n_correct, 1)
        self.assertEqual(result.scores[0].choice, "<BACK>")
        self.assertFalse(result.scores[0].parse_error)


if __name__ == "__main__":
    unittest.main()
