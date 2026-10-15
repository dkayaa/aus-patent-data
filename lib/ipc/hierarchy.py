"""A module that given an IPC code, returns its children"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IPC_JSONL = REPO_ROOT / "data" / "ipc-codes" / "ipc_codes_20260101.jsonl"

# KIND = DEPTH in WIPO (where smaller = coarser, larger = finer or deeper)
KIND_RANK: dict[str, int] = {"s": 0, "c": 1, "u": 2, "m": 3}
for _i in range(1, 10):
    KIND_RANK[str(_i)] = 3 + _i

# Converts IPC codes to human readable format eg A01B1/02 or G06N
_HUMAN_IPC_RE = re.compile(
    r"^([A-H])(\d{2})([A-Z])(?:(\d+)/(\d+))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IPCPlace:
    code: str
    code_raw: str
    kind: str
    title: str

    @property
    def rank(self) -> int:
        return KIND_RANK.get(self.kind, 99)

# Eg: Given " g06n 3/08 ", returns "G06N3/08"
def norm_code(code: str) -> str:
    return (code or "").strip().upper().replace(" ", "")

# WIPO 14-char symbol → human form (A01B0001020000 → A01B1/02).
def format_ipc_symbol(symbol: str) -> str:
    if len(symbol) != 14:
        return symbol
    subclass = symbol[:4]
    group = str(int(symbol[4:8]))
    subgroup = symbol[8:].rstrip("0") or "00"
    return f"{subclass}{group}/{subgroup}"


def canonicalise_ipc(code: str) -> str:
    n = norm_code(code)
    if not n:
        return ""
    if len(n) == 14 and n[4:].isdigit():
        n = format_ipc_symbol(n)
    if "/" in n:
        head, tail = n.split("/", 1)
        if tail != "00":
            n = f"{head}/{tail.rstrip('0') or '00'}"
    return n


def _build_parents(
    places: dict[str, IPCPlace],
) -> tuple[dict[str, str | None], dict[str, list[str]]]:
    ordered = sorted(places.values(), key=lambda p: p.code_raw)
    parent_of: dict[str, str | None] = {}
    stack: list[tuple[int, str]] = []
    for place in ordered:
        while stack and stack[-1][0] >= place.rank:
            stack.pop()
        parent_of[place.code] = stack[-1][1] if stack else None
        stack.append((place.rank, place.code))
    children: dict[str, list[str]] = defaultdict(list)
    for code, parent in parent_of.items():
        if parent:
            children[parent].append(code)
    return parent_of, children


def _human_to_raw14(code: str) -> str | None:
    c = norm_code(code)
    if not c:
        return None
    if len(c) == 14 and c[4:].isdigit():
        return c
    m = _HUMAN_IPC_RE.fullmatch(c)
    if not m:
        return None
    section, cl, subcl, group, subgroup = m.groups()
    short = f"{section.upper()}{cl}{subcl.upper()}"
    if group is None:
        return short
    g = f"{int(group):04d}"
    if len(subgroup) > 6:
        return None
    sg = subgroup + ("0" * (6 - len(subgroup)))
    return short + g + sg


class IPCHierarchy:
    def __init__(self, places: dict[str, IPCPlace]) -> None:
        self._places = places
        self._parent_of, children = _build_parents(places)
        self._children = dict(children)
        self._by_raw = {p.code_raw: p for p in places.values()}

    @classmethod
    def from_jsonl(cls, path: Path | None = None) -> IPCHierarchy:
        path = path or DEFAULT_IPC_JSONL
        places: dict[str, IPCPlace] = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                code = norm_code(str(row.get("ipc_code") or ""))
                if not code:
                    continue
                title = str(row.get("title") or "").strip()
                if not title:
                    continue
                kind = str(row.get("kind") or "")
                if kind not in KIND_RANK:
                    continue
                places[code] = IPCPlace(
                    code=code,
                    code_raw=str(row.get("ipc_code_raw") or code),
                    kind=kind,
                    title=title,
                )
        return cls(places)

    def resolve(self, code: str) -> IPCPlace | None:
        key = canonicalise_ipc(code)
        if not key:
            return None
        hit = self._places.get(key)
        if hit is not None:
            return hit
        raw = _human_to_raw14(key)
        if raw is None:
            return None
        return self._by_raw.get(raw)

    def get(self, code: str) -> IPCPlace | None:
        return self.resolve(code)

    def parent(self, code: str) -> IPCPlace | None:
        place = self.resolve(code)
        if place is None:
            return None
        parent_code = self._parent_of.get(place.code)
        if parent_code is None:
            return None
        return self._places[parent_code]

    def children(self, code: str) -> list[IPCPlace]:
        place = self.resolve(code)
        if place is None:
            return []
        return [self._places[c] for c in self._children.get(place.code, [])]

    def is_leaf(self, code: str) -> bool | None:
        place = self.resolve(code)
        if place is None:
            return None
        return not bool(self._children.get(place.code))

    def __contains__(self, code: str) -> bool:
        return self.resolve(code) is not None

    def __len__(self) -> int:
        return len(self._places)

    def __iter__(self) -> Iterator[IPCPlace]:
        return iter(self._places.values())
