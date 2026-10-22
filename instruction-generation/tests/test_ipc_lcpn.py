"""Hop records along a primary-IPC path."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "instruction-generation" / "src"
for path in (str(REPO), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ipc.accessor import IPCAccessor, IPCNode  # noqa: E402
from patents import PatentText  # noqa: E402
from tasks.ipc_lcpn import IPCLCPNTask, _lookup, _wrong_child  # noqa: E402

_NS = "http://www.wipo.int/classifications/ipc/masterfiles"

_SCHEME = f"""<?xml version="1.0" encoding="UTF-8"?>
<IPCScheme xmlns="{_NS}">
  <ipcEntry kind="s" symbol="A" entryType="K">
    <textBody><title><titlePart><text>HUMAN</text></titlePart></title></textBody>
    <ipcEntry kind="c" symbol="A01" entryType="K">
      <textBody><title><titlePart><text>AGRI</text></titlePart></title></textBody>
      <ipcEntry kind="u" symbol="A01B" entryType="K">
        <textBody><title><titlePart><text>SOIL</text></titlePart></title></textBody>
        <ipcEntry kind="m" symbol="A01B0001000000" entryType="K">
          <textBody><title><titlePart><text>HAND</text></titlePart></title></textBody>
          <ipcEntry kind="1" symbol="A01B0001020000" entryType="K">
            <textBody><title><titlePart><text>SPADES</text></titlePart></title></textBody>
          </ipcEntry>
          <ipcEntry kind="1" symbol="A01B0001200000" entryType="K">
            <textBody><title><titlePart><text>PADDED</text></titlePart></title></textBody>
          </ipcEntry>
        </ipcEntry>
        <ipcEntry kind="m" symbol="A01B0003000000" entryType="K">
          <textBody><title><titlePart><text>PLOUGHS</text></titlePart></title></textBody>
        </ipcEntry>
      </ipcEntry>
      <ipcEntry kind="u" symbol="A01C" entryType="K">
        <textBody><title><titlePart><text>PLANT</text></titlePart></title></textBody>
      </ipcEntry>
    </ipcEntry>
  </ipcEntry>
</IPCScheme>
"""

_DEFINITIONS = f"""<?xml version="1.0" encoding="UTF-8"?>
<IPCDefinitions xmlns="{_NS}"></IPCDefinitions>
"""


def _patent(**overrides: str) -> PatentText:
    fields = {
        "application_number": "2020000001",
        "invention_title": "A spade",
        "primary_ipc": "A01B1/02",
        "abstract": "ABSTRACT_TEXT",
        "claims": ["CLAIM_ONE", "CLAIM_TWO"],
        "claims_text": "CLAIM_ONE\n\nCLAIM_TWO",
        "claim_1": "CLAIM_ONE",
        "document_type": "A1",
        "filed_date": "2020-03-15",
    }
    fields.update(overrides)
    return PatentText(**fields)  # type: ignore[arg-type]


class IPCLCPNTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        scheme = root / "scheme.xml"
        definitions = root / "definitions.xml"
        scheme.write_text(_SCHEME, encoding="utf-8")
        definitions.write_text(_DEFINITIONS, encoding="utf-8")
        self.accessor = IPCAccessor(scheme, definitions)
        self.accessor.init()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _task(self, **options: object) -> IPCLCPNTask:
        options.setdefault("justify", False)
        client = SimpleNamespace(config=SimpleNamespace(model="none", provider="none"))
        task = IPCLCPNTask(client, options=options)  # type: ignore[arg-type]
        task._accessor = self.accessor
        return task

    def test_depth_yields_one_fewer_hops(self) -> None:
        records = self._task().generate(_patent())
        assert records is not None
        self.assertEqual(len(records), 5)
        self.assertEqual(
            [record["output"] for record in records],
            ["A", "A01", "A01B", "A01B1/00", "A01B1/02"],
        )
        self.assertEqual(records[0]["meta"]["now_node_id"], "<ROOT>")
        self.assertEqual(records[0]["meta"]["gold_node_id"], "A")
        self.assertEqual(records[0]["meta"]["path_node_ids"][0], "<ROOT>")
        self.assertEqual(records[0]["meta"]["depth"], 5)
        self.assertEqual(records[0]["meta"]["n_hops"], 5)
        self.assertFalse(records[0]["meta"]["is_back"])
        self.assertEqual(
            records[3]["input"].split("Titles: ", 1)[1].split("\n", 1)[0],
            "HUMAN > AGRI > SOIL",
        )
        menu = records[3]["input"]
        self.assertIn("1. A01B1/00 - HAND", menu)
        self.assertIn("2. A01B3/00 - PLOUGHS", menu)
        self.assertIn(
            "3. <STOP> - none of these children; remain at the current place",
            menu,
        )
        self.assertNotIn("A01C", menu)
        self.assertNotIn("<BACK>", records[0]["input"])
        self.assertEqual(records[0]["meta"]["stop_token"], "<STOP>")
        self.assertEqual(records[0]["meta"]["back_token"], "")
        self.assertFalse(records[-1]["meta"]["is_stop"])
        self.assertEqual(records[0]["meta"]["filed_date"], "2020-03-15")
        self.assertEqual(records[-1]["meta"]["filed_date"], "2020-03-15")
        self.assertIn("or <STOP> if none of the children fit", records[0]["instruction"])

    def test_primary_with_children_ends_on_stop(self) -> None:
        records = self._task().generate(_patent(primary_ipc="A01B1/00"))
        assert records is not None
        self.assertEqual(
            [record["output"] for record in records],
            ["A", "A01", "A01B", "A01B1/00", "<STOP>"],
        )
        stop = records[-1]
        self.assertTrue(stop["meta"]["is_stop"])
        self.assertEqual(stop["meta"]["now_node_id"], "A01B1/00")
        self.assertEqual(stop["meta"]["gold_node_id"], "<STOP>")
        self.assertEqual(stop["meta"]["hop_index"], 4)
        self.assertEqual(stop["meta"]["n_hops"], 5)
        self.assertEqual(stop["meta"]["depth"], 4)
        self.assertIn("Current node: A01B1/00", stop["input"])
        self.assertIn("1. A01B1/02 - SPADES", stop["input"])
        self.assertIn("2. A01B1/20 - PADDED", stop["input"])
        self.assertIn(
            "3. <STOP> - none of these children; remain at the current place",
            stop["input"],
        )
        self.assertEqual(
            stop["input"].split("Titles: ", 1)[1].split("\n", 1)[0],
            "HUMAN > AGRI > SOIL > HAND",
        )

    def test_stop_token_is_configurable(self) -> None:
        records = self._task(stop_token="HALT").generate(_patent())
        assert records is not None
        self.assertIn(
            "3. HALT - none of these children; remain at the current place",
            records[3]["input"],
        )
        self.assertNotIn("STOP", records[3]["input"])
        self.assertIn("or HALT only", records[0]["instruction"])
        self.assertEqual(records[0]["output"], "A")

    def test_abstract_and_claim_switches(self) -> None:
        both = self._task(include_abstract=True, claims="claim_1").generate(_patent())
        assert both is not None
        self.assertIn("ABSTRACT_TEXT", both[0]["input"])
        self.assertIn("CLAIM_ONE", both[0]["input"])
        self.assertNotIn("CLAIM_TWO", both[0]["input"])

        neither = self._task(include_abstract=False, claims="omit").generate(_patent())
        assert neither is not None
        self.assertNotIn("ABSTRACT_TEXT", neither[0]["input"])
        self.assertNotIn("CLAIM_ONE", neither[0]["input"])
        self.assertIn("Current node: <ROOT>", neither[0]["input"])

    def test_section_and_unknown_are_skipped(self) -> None:
        task = self._task()
        self.assertFalse(task.eligible(_patent(primary_ipc="A")))
        self.assertIsNone(task.generate(_patent(primary_ipc="A")))
        self.assertFalse(task.eligible(_patent(primary_ipc="Z99")))

    def test_short_subgroup_matches_printed_symbol(self) -> None:
        node = _lookup(self.accessor, "A01B1/2")
        assert node is not None
        self.assertEqual(node.node_id, "A01B1/20")

    def test_rejects_unknown_claims_mode(self) -> None:
        with self.assertRaises(ValueError):
            self._task(claims="all").generate(_patent())

    def test_justify_defaults_on_and_can_be_disabled(self) -> None:
        self.assertTrue(IPCLCPNTask.requires_llm({}))
        self.assertTrue(IPCLCPNTask.requires_llm({"justify": True}))
        self.assertFalse(IPCLCPNTask.requires_llm({"justify": False}))
        bare = IPCLCPNTask(
            SimpleNamespace(config=SimpleNamespace(model="none", provider="none")),
            options={},
        )
        self.assertTrue(bare._justify())

    def test_teacher_justification_precedes_classification(self) -> None:
        seen: list[str] = []

        class _Client:
            config = SimpleNamespace(model="teacher", provider="fake")

            def chat(self, messages: list[dict[str, str]], **_kwargs: object) -> str:
                seen.append(messages[-1]["content"])
                return "The composition integrates an exogenous sequence into a cell."

        task = IPCLCPNTask(
            _Client(),  # type: ignore[arg-type]
            options={"justify": True, "claims": "omit", "include_abstract": True},
        )
        task._accessor = self.accessor
        records = task.generate(_patent())
        assert records is not None
        self.assertEqual(len(seen), 5)
        self.assertIn("The correct next IPC symbol is A (HUMAN).", seen[0])
        self.assertNotIn("The correct next IPC symbol", records[0]["input"])
        self.assertEqual(
            records[0]["output"],
            "The composition integrates an exogenous sequence into a cell.\n"
            "Classification: A",
        )
        self.assertTrue(records[0]["meta"]["justify"])
        self.assertIn("Classification:", records[0]["instruction"])

    def test_stop_hop_justification(self) -> None:
        seen: list[str] = []

        class _Client:
            config = SimpleNamespace(model="teacher", provider="fake")

            def chat(self, messages: list[dict[str, str]], **_kwargs: object) -> str:
                seen.append(messages[-1]["content"])
                return "The disclosure is already covered by the current place."

        task = IPCLCPNTask(
            _Client(),  # type: ignore[arg-type]
            options={"justify": True, "claims": "omit", "include_abstract": False},
        )
        task._accessor = self.accessor
        records = task.generate(_patent(primary_ipc="A01B1/00"))
        assert records is not None
        self.assertEqual(len(seen), 5)
        self.assertIn("The correct choice is <STOP>.", seen[-1])
        self.assertIn("Remain at the current place A01B1/00 (HAND).", seen[-1])
        self.assertNotIn("The correct choice is <STOP>.", records[-1]["input"])
        self.assertEqual(
            records[-1]["output"],
            "The disclosure is already covered by the current place.\n"
            "Classification: <STOP>",
        )

    def test_back_token_adds_root_menu_and_one_wrong_child(self) -> None:
        records = self._task(back_token="<BACK>").generate(_patent())
        assert records is not None
        on_path = [record for record in records if not record["meta"]["is_back"]]
        backs = [record for record in records if record["meta"]["is_back"]]
        self.assertEqual(
            [record["output"] for record in on_path],
            ["A", "A01", "A01B", "A01B1/00", "A01B1/02"],
        )
        self.assertNotIn("<BACK>", records[0]["input"])
        self.assertNotIn("<BACK>", records[0]["instruction"])
        section = on_path[1]
        self.assertIn("3. <BACK> - return to the parent", section["input"])
        self.assertEqual(section["output"], "A01")
        self.assertEqual(section["meta"]["parent_node_id"], "<ROOT>")
        self.assertEqual(
            [record["meta"]["now_node_id"] for record in backs],
            ["A01C", "A01B3/00", "A01B1/20"],
        )
        self.assertTrue(all(record["output"] == "<BACK>" for record in backs))
        self.assertEqual(backs[0]["meta"]["parent_node_id"], "A01")
        self.assertEqual(backs[0]["meta"]["gold_node_id"], "<BACK>")
        self.assertEqual(backs[0]["meta"]["hop_index"], len(on_path))
        self.assertIn("Current node: A01C", backs[0]["input"])
        leaf_menu = backs[0]["input"].split("Children:\n", 1)[1]
        self.assertTrue(leaf_menu.startswith("1. <STOP>"))
        self.assertIn("2. <BACK> - return to the parent", leaf_menu)
        self.assertNotIn("A01B", leaf_menu)

    def test_back_hop_justification(self) -> None:
        seen: list[str] = []

        class _Client:
            config = SimpleNamespace(model="teacher", provider="fake")

            def chat(self, messages: list[dict[str, str]], **_kwargs: object) -> str:
                seen.append(messages[-1]["content"])
                return "The current place does not cover the claim."

        task = IPCLCPNTask(
            _Client(),  # type: ignore[arg-type]
            options={
                "justify": True,
                "claims": "omit",
                "include_abstract": False,
                "back_token": "<BACK>",
            },
        )
        task._accessor = self.accessor
        records = task.generate(_patent())
        assert records is not None
        backs = [record for record in records if record["meta"]["is_back"]]
        self.assertIn("The correct choice is <BACK>.", seen[-len(backs)])
        self.assertIn("Return to the parent place A01 (AGRI).", seen[-len(backs)])
        self.assertEqual(
            backs[0]["output"],
            "The current place does not cover the claim.\nClassification: <BACK>",
        )

    def test_wrong_child_is_shuffled_and_stable(self) -> None:
        nodes = [
            IPCNode(node_id=code, title=code, definition=None, kind="u", entry_type="K")
            for code in ("B", "C", "D")
        ]
        picks = {
            _wrong_child(nodes, "A", is_stop=False, seed=str(index)).node_id
            for index in range(40)
        }
        self.assertEqual(picks, {"B", "C", "D"})
        seed = "2020000001|A01|A01B"
        self.assertEqual(
            _wrong_child(nodes, "A", is_stop=False, seed=seed).node_id,
            _wrong_child(nodes, "A", is_stop=False, seed=seed).node_id,
        )
        first = self._task(back_token="<BACK>").generate(_patent())
        second = self._task(back_token="<BACK>").generate(_patent())
        assert first is not None and second is not None
        self.assertEqual(
            [record["meta"]["now_node_id"] for record in first if record["meta"]["is_back"]],
            [record["meta"]["now_node_id"] for record in second if record["meta"]["is_back"]],
        )


if __name__ == "__main__":
    unittest.main()
