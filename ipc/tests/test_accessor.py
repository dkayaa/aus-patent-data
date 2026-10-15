"""IPCAccessor parent/child edges from scheme nesting."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ipc.accessor import IPCAccessor  # noqa: E402

_NS = "http://www.wipo.int/classifications/ipc/masterfiles"

_SCHEME = f"""<?xml version="1.0" encoding="UTF-8"?>
<IPCScheme xmlns="{_NS}">
  <ipcEntry kind="s" symbol="A" entryType="K">
    <textBody><title><titlePart><text>HUMAN NECESSITIES</text></titlePart></title></textBody>
    <ipcEntry kind="t" symbol="A01" entryType="K">
      <textBody><title><titlePart><text>AGRICULTURE</text></titlePart></title></textBody>
    </ipcEntry>
    <ipcEntry kind="c" symbol="A01" entryType="K">
      <textBody><title>
        <titlePart><text>AGRICULTURE</text></titlePart>
        <titlePart><text>FORESTRY</text></titlePart>
      </title></textBody>
      <ipcEntry kind="u" symbol="A01B" entryType="K">
        <textBody><title><titlePart><text>SOIL WORKING</text></titlePart></title></textBody>
        <ipcEntry kind="i" symbol="A01B" entryType="K">
          <textBody><title><titlePart><text>INDEX</text></titlePart></title></textBody>
        </ipcEntry>
        <ipcEntry kind="n" symbol="A01B" entryType="K">
          <textBody><note><p>A note</p></note></textBody>
        </ipcEntry>
        <ipcEntry kind="m" symbol="A01B0001000000" entryType="K">
          <textBody><title><titlePart><text>Hand tools</text></titlePart></title></textBody>
          <ipcEntry kind="1" symbol="A01B0001020000" entryType="K">
            <textBody><title><titlePart><text>Spades</text></titlePart></title></textBody>
            <ipcEntry kind="2" symbol="A01B0001100000" entryType="K">
              <textBody><title><titlePart><text>with two blades</text></titlePart></title></textBody>
            </ipcEntry>
          </ipcEntry>
        </ipcEntry>
        <ipcEntry kind="g" symbol="A01B0003000000" entryType="K">
          <textBody><title><titlePart><text>Ploughs</text></titlePart></title></textBody>
          <ipcEntry kind="m" symbol="A01B0003000000" entryType="K">
            <textBody><title><titlePart><text>Ploughs with fixed plough-shares</text></titlePart></title></textBody>
          </ipcEntry>
        </ipcEntry>
        <ipcEntry kind="m" symbol="A01B0101000000" entryType="I">
          <textBody><title><titlePart><text>Indexing lawn</text></titlePart></title></textBody>
        </ipcEntry>
      </ipcEntry>
      <ipcEntry kind="u" symbol="A01C" entryType="K">
        <textBody><title><titlePart><text>PLANTING</text></titlePart></title></textBody>
      </ipcEntry>
    </ipcEntry>
  </ipcEntry>
  <ipcEntry kind="s" symbol="B" entryType="K">
    <textBody><title><titlePart><text>PERFORMING OPERATIONS</text></titlePart></title></textBody>
  </ipcEntry>
</IPCScheme>
"""

_DEFINITIONS = f"""<?xml version="1.0" encoding="UTF-8"?>
<IPCDefinitions xmlns="{_NS}" xmlns:xhtml="http://www.w3.org/1999/xhtml">
  <IPC-DEFINITION IPC="A01">
    <GLOSSARYOFTERMS><xhtml:p>not a definition statement</xhtml:p></GLOSSARYOFTERMS>
  </IPC-DEFINITION>
  <IPC-DEFINITION IPC="A01B0001020000">
    <DEFINITION-STATEMENT><xhtml:p>Spade definition.</xhtml:p></DEFINITION-STATEMENT>
  </IPC-DEFINITION>
</IPCDefinitions>
"""


class IPCAccessorTests(unittest.TestCase):
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

    def test_requires_init(self) -> None:
        fresh = IPCAccessor(self.accessor.scheme_path, self.accessor.definitions_path)
        with self.assertRaises(RuntimeError):
            fresh.get_children("A")
        with self.assertRaises(RuntimeError):
            fresh.get_parent("A")

    def test_sections_and_skipped_guidance_heading(self) -> None:
        sections = self.accessor.get_children("")
        self.assertEqual([node.node_id for node in sections], ["A", "B"])
        self.assertEqual(sections[0].title, "HUMAN NECESSITIES")
        self.assertEqual(sections[0].kind, "s")
        self.assertEqual(self.accessor.get_parent("A"), [])

        classes = self.accessor.get_children("A")
        self.assertEqual([node.node_id for node in classes], ["A01"])
        self.assertEqual(classes[0].title, "AGRICULTURE; FORESTRY")
        self.assertEqual(classes[0].kind, "c")
        self.assertEqual(classes[0].definition, None)
        parents = self.accessor.get_parent("A01")
        self.assertEqual([node.node_id for node in parents], ["A"])

    def test_scheme_order_skips_index_note_and_guidance(self) -> None:
        subclasses = self.accessor.get_children("A01")
        self.assertEqual([node.node_id for node in subclasses], ["A01B", "A01C"])

        groups = self.accessor.get_children("A01B")
        self.assertEqual(
            [node.node_id for node in groups],
            ["A01B1/00", "A01B3/00", "A01B101/00"],
        )
        self.assertEqual(groups[0].title, "Hand tools")
        self.assertEqual(groups[1].title, "Ploughs with fixed plough-shares")
        self.assertEqual(groups[2].entry_type, "I")
        self.assertEqual(
            [node.node_id for node in self.accessor.get_parent("A01B3/00")],
            ["A01B"],
        )

    def test_dot_parent_follows_nesting_not_symbol_spelling(self) -> None:
        parents = self.accessor.get_parent("A01B1/10")
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0].node_id, "A01B1/02")
        self.assertEqual(self.accessor.get("A01B1/10").title, "with two blades")
        self.assertEqual(parents[0].title, "Spades")
        self.assertEqual(parents[0].definition, "Spade definition.")
        self.assertEqual(parents[0].kind, "1")

        blade = self.accessor.get_children("A01B1/02")
        self.assertEqual([node.node_id for node in blade], ["A01B1/10"])
        self.assertIsNone(blade[0].definition)

    def test_accepts_spaced_and_raw_symbols(self) -> None:
        spaced = self.accessor.get_parent("A01B 1/10")
        raw = self.accessor.get_parent("A01B0001100000")
        self.assertEqual(spaced[0].node_id, "A01B1/02")
        self.assertEqual(raw[0].node_id, "A01B1/02")

    def test_unknown_id_raises(self) -> None:
        with self.assertRaises(KeyError):
            self.accessor.get_children("Z99")
        with self.assertRaises(KeyError):
            self.accessor.get_parent("A01B1/1")


class IPCAccessorRealSchemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.accessor = IPCAccessor()
        cls.accessor.init()

    def test_a01b_children_and_parent_of_one_dot_ten(self) -> None:
        children = self.accessor.get_children("A01B")
        ids = [node.node_id for node in children]
        self.assertIn("A01B1/00", ids)
        hand_tools = next(node for node in children if node.node_id == "A01B1/00")
        self.assertEqual(hand_tools.kind, "m")
        self.assertIn("Hand tools", hand_tools.title)

        parents = self.accessor.get_parent("A01B1/10")
        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0].node_id, "A01B1/06")
        self.assertEqual(parents[0].kind, "1")

        sections = [node.node_id for node in self.accessor.get_children("")]
        self.assertEqual(sections, list("ABCDEFGH"))

        a01d = next(
            node for node in self.accessor.get_children("A01") if node.node_id == "A01D"
        )
        self.assertIsNotNone(a01d.definition)
        assert a01d.definition is not None
        self.assertIn("Methods, apparatus or hand tools", a01d.definition)


if __name__ == "__main__":
    unittest.main()
