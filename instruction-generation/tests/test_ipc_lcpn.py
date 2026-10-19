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

from ipc.accessor import IPCAccessor  # noqa: E402
from patents import PatentText  # noqa: E402
from tasks.ipc_lcpn import IPCLCPNTask, _lookup  # noqa: E402

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
        self.assertEqual(len(records), 4)
        self.assertEqual(
            [record["output"] for record in records],
            ["A01", "A01B", "A01B1/00", "A01B1/02"],
        )
        self.assertEqual(records[0]["meta"]["depth"], 5)
        self.assertEqual(records[0]["meta"]["n_hops"], 4)
        self.assertEqual(
            records[2]["input"].split("Titles: ", 1)[1].split("\n", 1)[0],
            "HUMAN > AGRI > SOIL",
        )
        menu = records[2]["input"]
        self.assertIn("1. A01B1/00 - HAND", menu)
        self.assertIn("2. A01B3/00 - PLOUGHS", menu)
        self.assertIn(
            "3. <STOP> - none of these children; remain at the current place",
            menu,
        )
        self.assertNotIn("A01C", menu)
        self.assertEqual(records[0]["meta"]["stop_token"], "<STOP>")
        self.assertEqual(records[0]["meta"]["filed_date"], "2020-03-15")
        self.assertEqual(records[-1]["meta"]["filed_date"], "2020-03-15")
        self.assertIn("or <STOP> if none of the children fit", records[0]["instruction"])

    def test_stop_token_is_configurable(self) -> None:
        records = self._task(stop_token="HALT").generate(_patent())
        assert records is not None
        self.assertIn(
            "3. HALT - none of these children; remain at the current place",
            records[2]["input"],
        )
        self.assertNotIn("STOP", records[2]["input"])
        self.assertIn("or HALT only", records[0]["instruction"])
        self.assertEqual(records[0]["output"], "A01")

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
        self.assertIn("Current node: A", neither[0]["input"])

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
        self.assertEqual(len(seen), 4)
        self.assertIn("The correct next IPC symbol is A01 (AGRI).", seen[0])
        self.assertNotIn("The correct next IPC symbol", records[0]["input"])
        self.assertEqual(
            records[0]["output"],
            "The composition integrates an exogenous sequence into a cell.\n"
            "Classification: A01",
        )
        self.assertTrue(records[0]["meta"]["justify"])
        self.assertIn("Classification:", records[0]["instruction"])


if __name__ == "__main__":
    unittest.main()
