"""IPC regex and Classification/Justification parsing for ipc_reasoning."""

from __future__ import annotations

import re
from typing import Any

IPC_RE = re.compile(
    r"^[A-H][0-9]{2}[A-Z](?:[0-9]{1,4}/[0-9]{2,6})?$"
)
IPC_FIND_RE = re.compile(
    r"\b([A-H][0-9]{2}[A-Z](?:[0-9]{1,4}/[0-9]{2,6})?)\b"
)
# Same shape, no word boundaries — used on whitespace-stripped text so
# "H04N 19/593" still counts as H04N19/593.
_IPC_FIND_COMPACT_RE = re.compile(
    r"([A-H][0-9]{2}[A-Z](?:[0-9]{1,4}/[0-9]{2,6})?)"
)
OUTPUT_RE = re.compile(
    r"Classification:\s*(?P<code>\S+)\s*\nJustification:\s*(?P<body>.+)",
    re.DOTALL | re.IGNORECASE,
)
_IPC_PARTS_RE = re.compile(
    r"^([A-H])([0-9]{2})([A-Z])(?:([0-9]{1,4})(?:/([0-9]{2,6}))?)?$"
)


def normalize_ipc(code: str) -> str:
    return code.strip().upper().replace(" ", "")


def parse_ipc_output(output: str) -> tuple[str | None, str | None]:
    m = OUTPUT_RE.search(output.strip())
    if not m:
        return None, None
    return normalize_ipc(m.group("code")), m.group("body").strip()


def find_ipc_mentions(text: str) -> list[str]:
    """Normalized IPC-shaped tokens anywhere in free text.

    Strips whitespace first so office-style ``H04N 19/593`` matches
    ``H04N19/593``. Order is left-to-right; duplicates are dropped.
    """
    compact = re.sub(r"\s+", "", text.upper())
    found: list[str] = []
    seen: set[str] = set()
    for match in _IPC_FIND_COMPACT_RE.finditer(compact):
        norm = normalize_ipc(match.group(1))
        if norm in seen or not IPC_RE.match(norm):
            continue
        seen.add(norm)
        found.append(norm)
    return found


def parse_ipc_symbol(
    code: str,
) -> tuple[str, str, str, int | None, str | None] | None:
    """Return (section, class, subclass, group, subgroup) or None."""
    m = _IPC_PARTS_RE.match(normalize_ipc(code))
    if not m:
        return None
    section, cls, subclass, group, subgroup = m.groups()
    group_i = int(group) if group is not None else None
    return section, cls, subclass, group_i, subgroup


def is_same_place_or_ancestor(found: str, primary: str) -> bool:
    """True if ``found`` is ``primary`` or a coarser place that contains it.

    Allows subclass/group mentions of the gold symbol (e.g. G05D vs G05D1/00,
    C12Q1/68 vs C12Q1/6876). Sibling or unrelated symbols return False.
    """
    fp = parse_ipc_symbol(found)
    pp = parse_ipc_symbol(primary)
    if fp is None or pp is None:
        return False
    if fp[0] != pp[0] or fp[1] != pp[1] or fp[2] != pp[2]:
        return False
    if fp[3] is None:
        return True
    if pp[3] is None or fp[3] != pp[3]:
        return False
    if fp[4] is None:
        return True
    prim_sg = pp[4] or ""
    found_sg = fp[4]
    return prim_sg == found_sg or prim_sg.startswith(found_sg)


def wipo_grounding_text(entry: Any | None) -> str:
    """Title plus definition (or scheme note) for Mode 1 pairing."""
    if entry is None:
        return ""
    title = str(getattr(entry, "title", "") or "").strip()
    definition = str(getattr(entry, "definition_statement", "") or "").strip()
    note = str(getattr(entry, "scheme_note", "") or "").strip()
    parts: list[str] = []
    if title:
        parts.append(title)
    if definition:
        parts.append(definition)
    elif note:
        parts.append(note)
    return "\n".join(parts)


def check_ipc_reasoning(record: dict[str, Any]) -> tuple[list[str], str | None]:
    """Return (failures, justification_body)."""
    failures: list[str] = []
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    primary_raw = str(meta.get("primary_ipc") or "").strip()
    primary = normalize_ipc(primary_raw) if primary_raw else ""

    if not primary or not IPC_RE.match(primary):
        failures.append("primary_ipc_invalid")
        primary = primary or None  # type: ignore[assignment]

    code, body = parse_ipc_output(str(record.get("output") or ""))
    if code is None or not body:
        failures.append("output_parse_failed")
        return failures, body

    if not IPC_RE.match(code):
        failures.append("classification_code_invalid")

    if primary and code != primary:
        failures.append("ipc_code_mismatch")

    for found in IPC_FIND_RE.findall(body):
        norm = normalize_ipc(found)
        if not primary or is_same_place_or_ancestor(norm, primary):
            continue
        failures.append("conflicting_ipc_in_justification")
        break

    return failures, body
