"""Load the WIPO IPC scheme tree and look up immediate relatives.

Parent and child come from ``ipcEntry`` nesting in the scheme XML. Guidance
headings, notes, and subclass indexes are not places: their descendant places
attach to the nearest kept ancestor. Definition text is the definitions-file
``DEFINITION-STATEMENT`` for that symbol, when one exists.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEME = REPO_ROOT / "data" / "ipc-codes" / "EN_ipc_scheme_20260101.xml"
DEFAULT_DEFINITIONS = (
    REPO_ROOT / "data" / "ipc-codes" / "EN_ipc_definitions_20260101.xml"
)

# Classification and indexing places. Guidance headings (t, g), notes (n),
# and subclass indexes (i) are skipped.
_PLACE_KINDS = frozenset(
    {"s", "c", "u", "m", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
)

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class IPCNode:
    node_id: str
    title: str
    definition: str | None
    kind: str
    entry_type: str | None


@dataclass
class _Open:
    node_id: str
    is_place: bool


def format_symbol(symbol: str) -> str:
    """WIPO 14-character symbol to printed form, spaces removed.

    Subgroup padding zeros are stripped, and at least two subgroup digits are
    kept, so ``A01B0001100000`` is ``A01B1/10`` rather than ``A01B1/1``.
    Section, class, and subclass symbols are returned unchanged.
    """
    if len(symbol) != 14 or not symbol[4:].isdigit():
        return symbol
    group = str(int(symbol[4:8]))
    subgroup = symbol[8:].rstrip("0")
    if len(subgroup) < 2:
        subgroup = subgroup.ljust(2, "0")
    return f"{symbol[:4]}{group}/{subgroup}"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _norm_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _plain_text(elem: ET.Element) -> str | None:
    text = _norm_ws("".join(elem.itertext()))
    return text or None


def _extract_title(elem: ET.Element) -> str:
    for child in elem:
        if _local(child.tag) != "textBody":
            continue
        for body_child in child:
            if _local(body_child.tag) != "title":
                continue
            parts: list[str] = []
            for part in body_child:
                if _local(part.tag) != "titlePart":
                    continue
                bits: list[str] = []
                for el in part:
                    if _local(el.tag) != "text":
                        continue
                    bit = _norm_ws("".join(el.itertext()))
                    if bit:
                        bits.append(bit)
                if bits:
                    parts.append(" ".join(bits))
            return "; ".join(parts) if parts else ""
    return ""


def _parse_definitions(path: Path) -> dict[str, str]:
    """Map printed node id → definition statement text."""
    out: dict[str, str] = {}
    for _event, elem in ET.iterparse(path, events=("end",)):
        if _local(elem.tag) != "IPC-DEFINITION":
            continue
        symbol = (elem.get("IPC") or "").strip()
        statement: str | None = None
        if symbol:
            for child in elem:
                if _local(child.tag) != "DEFINITION-STATEMENT":
                    continue
                statement = _plain_text(child)
                break
        elem.clear()
        if symbol and statement:
            out[format_symbol(symbol)] = statement
    return out


def _parse_scheme(
    path: Path,
    definitions: dict[str, str],
) -> tuple[dict[str, IPCNode], dict[str, list[str]], dict[str, str]]:
    nodes: dict[str, IPCNode] = {}
    children: dict[str, list[str]] = {}
    parent_of: dict[str, str] = {}
    open_entries: list[_Open] = []

    for event, elem in ET.iterparse(path, events=("start", "end")):
        if _local(elem.tag) != "ipcEntry":
            continue
        if event == "start":
            kind = elem.get("kind") or ""
            symbol = (elem.get("symbol") or "").strip()
            is_place = kind in _PLACE_KINDS and bool(symbol)
            node_id = format_symbol(symbol) if is_place else ""
            open_entries.append(_Open(node_id=node_id, is_place=is_place))
            continue

        current = open_entries.pop()
        if current.is_place and current.node_id not in nodes:
            parent_id = ""
            for ancestor in reversed(open_entries):
                if ancestor.is_place:
                    parent_id = ancestor.node_id
                    break
            kind = elem.get("kind") or ""
            nodes[current.node_id] = IPCNode(
                node_id=current.node_id,
                title=_extract_title(elem),
                definition=definitions.get(current.node_id),
                kind=kind,
                entry_type=elem.get("entryType"),
            )
            children.setdefault(parent_id, []).append(current.node_id)
            parent_of[current.node_id] = parent_id
        elem.clear()

    return nodes, children, parent_of


class IPCAccessor:
    """Immediate children and parent of an IPC place.

    ``init`` reads the scheme and definitions XMLs once. ``get_children("")``
    returns the sections. ``get_parent`` of a section is an empty list.
    """

    def __init__(
        self,
        scheme_path: Path | None = None,
        definitions_path: Path | None = None,
    ) -> None:
        self.scheme_path = Path(scheme_path) if scheme_path else DEFAULT_SCHEME
        self.definitions_path = (
            Path(definitions_path) if definitions_path else DEFAULT_DEFINITIONS
        )
        self._ready = False
        self._nodes: dict[str, IPCNode] = {}
        self._children: dict[str, list[str]] = {}
        self._parent: dict[str, str] = {}

    def init(self) -> None:
        for path in (self.scheme_path, self.definitions_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        definitions = _parse_definitions(self.definitions_path)
        self._nodes, self._children, self._parent = _parse_scheme(
            self.scheme_path, definitions
        )
        self._ready = True

    def get(self, node_id: str) -> IPCNode:
        """The place itself."""
        self._require_init()
        key = self._resolve(node_id)
        if key not in self._nodes:
            raise KeyError(node_id)
        return self._nodes[key]

    def get_children(self, node_id: str) -> list[IPCNode]:
        """Immediate child places in scheme order."""
        self._require_init()
        key = self._resolve(node_id)
        if key != "" and key not in self._nodes:
            raise KeyError(node_id)
        return [self._nodes[child_id] for child_id in self._children.get(key, ())]

    def get_parent(self, node_id: str) -> list[IPCNode]:
        """The immediate parent place, or an empty list for a section."""
        self._require_init()
        key = self._resolve(node_id)
        if key not in self._nodes:
            raise KeyError(node_id)
        parent_id = self._parent[key]
        if not parent_id:
            return []
        return [self._nodes[parent_id]]

    def _require_init(self) -> None:
        if not self._ready:
            raise RuntimeError("IPCAccessor.init() has not been called")

    @staticmethod
    def _resolve(node_id: str) -> str:
        text = node_id.strip().upper().replace(" ", "")
        if not text:
            return ""
        if len(text) == 14 and text[4:].isdigit():
            return format_symbol(text)
        return text
