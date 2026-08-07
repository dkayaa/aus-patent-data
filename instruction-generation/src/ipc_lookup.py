"""Load WIPO IPC catalog JSONL for grounding legal-reasoning prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IPCEntry:
    ipc_code: str
    ipc_code_raw: str
    title: str
    definition_statement: str | None
    scheme_note: str | None
    has_definition_entry: bool

    def grounding_text(self) -> str:
        parts = [f"IPC code: {self.ipc_code}", f"Title: {self.title}"]
        if self.definition_statement:
            parts.append(f"Definition: {self.definition_statement}")
        elif self.scheme_note:
            parts.append(f"Scheme note: {self.scheme_note}")
        return "\n".join(parts)


def _normalize_code(code: str) -> str:
    return code.strip().upper().replace(" ", "")


class IPCLookup:
    def __init__(self, by_code: dict[str, IPCEntry]) -> None:
        self._by_code = by_code

    @classmethod
    def from_jsonl(cls, path: Path) -> IPCLookup:
        by_code: dict[str, IPCEntry] = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row: dict[str, Any] = json.loads(line)
                code = _normalize_code(str(row.get("ipc_code") or ""))
                if not code:
                    continue
                title = str(row.get("title") or "").strip()
                if not title:
                    continue
                entry = IPCEntry(
                    ipc_code=code,
                    ipc_code_raw=str(row.get("ipc_code_raw") or ""),
                    title=title,
                    definition_statement=row.get("definition_statement"),
                    scheme_note=row.get("scheme_note"),
                    has_definition_entry=bool(row.get("has_definition_entry")),
                )
                by_code[code] = entry
        return cls(by_code)

    def get(self, code: str) -> IPCEntry | None:
        return self._by_code.get(_normalize_code(code))

    def __len__(self) -> int:
        return len(self._by_code)
