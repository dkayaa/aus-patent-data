"""Tests for lib.ipc.hierarchy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.ipc.hierarchy import (  # noqa: E402
    DEFAULT_IPC_JSONL,
    IPCHierarchy,
    IPCPlace,
    canonicalise_ipc,
    format_ipc_symbol,
    norm_code,
)


def _place(code: str, raw: str, kind: str, title: str) -> IPCPlace:
    return IPCPlace(code=code, code_raw=raw, kind=kind, title=title)


def _toy_tree() -> IPCHierarchy:
    places = {
        "A": _place("A", "A", "s", "HUMAN NECESSITIES"),
        "A01": _place("A01", "A01", "c", "AGRICULTURE"),
        "A01B": _place("A01B", "A01B", "u", "SOIL WORKING"),
        "A01B1/00": _place("A01B1/00", "A01B0001000000", "m", "Hand tools"),
        "A01B1/02": _place("A01B1/02", "A01B0001020000", "1", "Spades; Shovels"),
        "A01B1/04": _place("A01B1/04", "A01B0001040000", "2", "with teeth"),
    }
    return IPCHierarchy(places)


class NormTests(unittest.TestCase):
    def test_norm_code(self) -> None:
        self.assertEqual(norm_code(" g06n 3/08 "), "G06N3/08")

    def test_format_ipc_symbol(self) -> None:
        self.assertEqual(format_ipc_symbol("A01B0001020000"), "A01B1/02")

    def test_canonicalise_trailing_subgroup_zeros(self) -> None:
        self.assertEqual(canonicalise_ipc("C07K16/30"), "C07K16/3")
        self.assertEqual(canonicalise_ipc("A01B1/00"), "A01B1/00")


class ToyHierarchyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = _toy_tree()

    def test_children_section_to_class(self) -> None:
        self.assertEqual([p.code for p in self.tree.children("A")], ["A01"])

    def test_children_main_group(self) -> None:
        self.assertEqual(
            [p.code for p in self.tree.children("A01B1/00")],
            ["A01B1/02"],
        )

    def test_parent_and_leaf(self) -> None:
        parent = self.tree.parent("A01B1/04")
        assert parent is not None
        self.assertEqual(parent.code, "A01B1/02")
        self.assertTrue(self.tree.is_leaf("A01B1/04"))
        self.assertFalse(self.tree.is_leaf("A01B1/00"))
        self.assertIsNone(self.tree.parent("A"))

    def test_unknown_code(self) -> None:
        self.assertIsNone(self.tree.get("ZZZZ"))
        self.assertEqual(self.tree.children("ZZZZ"), [])
        self.assertIsNone(self.tree.is_leaf("ZZZZ"))


@unittest.skipUnless(DEFAULT_IPC_JSONL.is_file(), "IPC JSONL catalog missing")
class CatalogGoldenTests(unittest.TestCase):
    tree: IPCHierarchy

    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = IPCHierarchy.from_jsonl()

    def test_load_nonempty(self) -> None:
        self.assertGreater(len(self.tree), 0)

    def test_g06n_children(self) -> None:
        codes = [p.code for p in self.tree.children("G06N")]
        self.assertEqual(
            codes,
            [
                "G06N3/00",
                "G06N5/00",
                "G06N7/00",
                "G06N10/00",
                "G06N20/00",
                "G06N99/00",
            ],
        )

    def test_leaf_b62m9_1246(self) -> None:
        place = self.tree.get("B62M9/1246")
        assert place is not None
        self.assertEqual(place.kind, "8")
        self.assertTrue(self.tree.is_leaf("B62M9/1246"))
        self.assertEqual(self.tree.children("B62M9/1246"), [])
        parent = self.tree.parent("B62M9/1246")
        assert parent is not None
        self.assertEqual(parent.code, "B62M9/1244")

    def test_office_form_c07k16_30(self) -> None:
        place = self.tree.get("C07K16/30")
        assert place is not None
        self.assertEqual(place.code, "C07K16/3")


if __name__ == "__main__":
    unittest.main()
