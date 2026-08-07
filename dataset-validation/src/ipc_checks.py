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
OUTPUT_RE = re.compile(
    r"Classification:\s*(?P<code>\S+)\s*\nJustification:\s*(?P<body>.+)",
    re.DOTALL | re.IGNORECASE,
)


def normalize_ipc(code: str) -> str:
    return code.strip().upper().replace(" ", "")


def parse_ipc_output(output: str) -> tuple[str | None, str | None]:
    m = OUTPUT_RE.search(output.strip())
    if not m:
        return None, None
    return normalize_ipc(m.group("code")), m.group("body").strip()


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
        if primary and norm != primary:
            failures.append("conflicting_ipc_in_justification")
            break

    return failures, body
