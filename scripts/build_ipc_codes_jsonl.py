#!/usr/bin/env python3
"""Join WIPO IPC scheme + definitions XMLs into one JSONL catalog.

Reads:
  data/ipc-codes/EN_ipc_scheme_<edition>.xml
  data/ipc-codes/EN_ipc_definitions_<edition>.xml

Writes one JSON object per classification place (section/class/subclass/group),
including scheme metadata/title/notes and every definitions block when present
(definition statement, references, glossary, large subjects, synonyms, special
rules). Codes without a definitions entry still appear with null definition
fields.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IPC_DIR = REPO_ROOT / "data" / "ipc-codes"
DEFAULT_EDITION = "20260101"

# Classification places only (skip index trees and note-only entries).
_INCLUDE_KINDS = frozenset({"s", "c", "u", "m", "1", "2", "3", "4", "5", "6", "7", "8", "9"})
# Prefer more specific classification kinds when a symbol appears twice.
_KIND_RANK = {
    "s": 10,
    "c": 20,
    "u": 30,
    "m": 40,
    "1": 50,
    "2": 51,
    "3": 52,
    "4": 53,
    "5": 54,
    "6": 55,
    "7": 56,
    "8": 57,
    "9": 58,
}

_REF_KINDS = (
    "LIMITINGREFERENCES",
    "APPLICATIONORIENTEDREFERENCES",
    "INFORMATIVEREFERENCES",
    "REFERENCESOUTOFARESIDUALPLACE",
)

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def format_ipc_symbol(symbol: str) -> str:
    """Convert WIPO 14-char padded symbol to human form (e.g. A01B0001020000 → A01B1/02)."""
    if len(symbol) != 14:
        return symbol
    subclass = symbol[:4]
    group = str(int(symbol[4:8]))
    subgroup = symbol[8:].rstrip("0") or "00"
    return f"{subclass}{group}/{subgroup}"


def _plain_text(elem: ET.Element | None) -> str | None:
    if elem is None:
        return None
    text = _norm_ws("".join(elem.itertext()))
    return text or None


def _collect_hrefs(elem: ET.Element) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for node in elem.iter():
        if _local(node.tag) != "a":
            continue
        href = (node.get("href") or "").strip()
        if href and href not in seen:
            seen.add(href)
            out.append(href)
    return out


def _parse_reference_block(block: ET.Element) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tr in block.iter():
        if _local(tr.tag) != "tr":
            continue
        tds = [c for c in tr if _local(c.tag) == "td"]
        if not tds:
            continue
        text = _plain_text(tds[0]) or ""
        symbols = _collect_hrefs(tds[1]) if len(tds) > 1 else []
        # Heading rows often have a bare "." in the symbol column and no links.
        if not text and not symbols:
            continue
        rows.append({"text": text, "symbols": symbols})
    return rows


def _parse_glossary(block: ET.Element) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for dl in block.iter():
        if _local(dl.tag) != "dl":
            continue
        terms: list[str] = []
        for item in dl:
            name = _local(item.tag)
            if name == "dt":
                term = _plain_text(item)
                if term:
                    terms.append(term)
            elif name == "dd":
                definition = _plain_text(item)
                if terms or definition:
                    entries.append({"terms": terms, "definition": definition})
                terms = []
    return entries


def _parse_synonym_lists(block: ET.Element) -> list[list[str]]:
    groups: list[list[str]] = []
    for ul in block.iter():
        if _local(ul.tag) != "ul":
            continue
        items: list[str] = []
        for li in ul:
            if _local(li.tag) != "li":
                continue
            text = _plain_text(li)
            if text:
                items.append(text)
        if items:
            groups.append(items)
    return groups


def _parse_term_definition_dl(block: ET.Element) -> list[dict[str, str | None]]:
    out: list[dict[str, str | None]] = []
    for dl in block.iter():
        if _local(dl.tag) != "dl":
            continue
        term: str | None = None
        for item in dl:
            name = _local(item.tag)
            if name == "dt":
                term = _plain_text(item)
            elif name == "dd":
                out.append({"term": term, "definition": _plain_text(item)})
                term = None
    return out


def _parse_synonyms_and_keywords(block: ET.Element) -> dict[str, Any] | None:
    result: dict[str, Any] = {}
    for child in block:
        name = _local(child.tag)
        if name == "SYNONYMS":
            groups = _parse_synonym_lists(child)
            if groups:
                result["synonyms"] = groups
        elif name == "ABBREVIATIONS":
            rows = _parse_term_definition_dl(child)
            if rows:
                result["abbreviations"] = rows
        elif name == "INSTEADOFWORDS":
            rows = _parse_term_definition_dl(child)
            if rows:
                result["instead_of_words"] = rows
        elif name == "SPECIALMEANINGS":
            rows = _parse_term_definition_dl(child)
            if rows:
                result["special_meanings"] = rows
        else:
            # Preserve unexpected IPC children as plain text.
            text = _plain_text(child)
            if text:
                result[name.lower()] = text
    return result or None


def _empty_definition_fields() -> dict[str, Any]:
    return {
        "definition_statement": None,
        "references": None,
        "glossary": None,
        "large_subjects": None,
        "synonyms_and_keywords": None,
        "special_rules": None,
        "has_definition_entry": False,
    }


def parse_definitions(path: Path) -> dict[str, dict[str, Any]]:
    """Map IPC symbol → flattened definition attributes."""
    out: dict[str, dict[str, Any]] = {}
    for _event, elem in ET.iterparse(path, events=("end",)):
        if _local(elem.tag) != "IPC-DEFINITION":
            continue
        symbol = (elem.get("IPC") or "").strip()
        if not symbol:
            elem.clear()
            continue

        fields = _empty_definition_fields()
        fields["has_definition_entry"] = True
        references: dict[str, list[dict[str, Any]]] = {}

        for child in elem:
            name = _local(child.tag)
            if name == "DEFINITION-STATEMENT":
                fields["definition_statement"] = _plain_text(child)
            elif name == "REFERENCES":
                for block in child:
                    ref_kind = _local(block.tag)
                    if ref_kind not in _REF_KINDS:
                        continue
                    rows = _parse_reference_block(block)
                    if rows:
                        references[ref_kind.lower()] = rows
            elif name == "GLOSSARYOFTERMS":
                glossary = _parse_glossary(child)
                fields["glossary"] = glossary or None
            elif name == "LARGESUBJECTS":
                fields["large_subjects"] = _plain_text(child)
            elif name == "SYNONYMSANDKEYWORDS":
                fields["synonyms_and_keywords"] = _parse_synonyms_and_keywords(child)
            elif name == "SPECIALRULES":
                fields["special_rules"] = _plain_text(child)
            else:
                # Keep unknown top-level definition blocks rather than drop them.
                text = _plain_text(child)
                if text:
                    fields[name.lower()] = text

        fields["references"] = references or None
        out[symbol] = fields
        elem.clear()
    return out


def _extract_title(text_body: ET.Element) -> str | None:
    for child in text_body:
        if _local(child.tag) != "title":
            continue
        parts: list[str] = []
        for part in child:
            if _local(part.tag) != "titlePart":
                continue
            bits: list[str] = []
            for el in part:
                if _local(el.tag) == "text":
                    bit = _norm_ws("".join(el.itertext()))
                    if bit:
                        bits.append(bit)
            if bits:
                parts.append(" ".join(bits))
        return "; ".join(parts) if parts else None
    return None


def _extract_note(text_body: ET.Element) -> str | None:
    notes: list[str] = []
    for child in text_body:
        if _local(child.tag) != "note":
            continue
        text = _plain_text(child)
        if text:
            notes.append(text)
    return " ".join(notes) if notes else None


def iter_scheme_entries(path: Path) -> Iterator[dict[str, Any]]:
    """Yield scheme ipcEntry records (classification + notes)."""
    for _event, elem in ET.iterparse(path, events=("end",)):
        if _local(elem.tag) != "ipcEntry":
            continue
        symbol = (elem.get("symbol") or "").strip()
        kind = elem.get("kind") or ""
        if not symbol:
            elem.clear()
            continue

        text_body = None
        for child in elem:
            if _local(child.tag) == "textBody":
                text_body = child
                break

        title = _extract_title(text_body) if text_body is not None else None
        note = _extract_note(text_body) if text_body is not None else None
        record = {
            "symbol": symbol,
            "kind": kind,
            "entry_type": elem.get("entryType"),
            "edition": elem.get("edition"),
            "end_symbol": elem.get("endSymbol"),
            "title": title,
            "note": note,
        }
        yield record
        for child in list(elem):
            child.clear()
        elem.clear()


def build_catalog(
    scheme_path: Path,
    definitions_path: Path,
) -> list[dict[str, Any]]:
    definitions = parse_definitions(definitions_path)

    by_symbol: dict[str, dict[str, Any]] = {}
    notes_by_symbol: dict[str, list[str]] = {}

    for entry in iter_scheme_entries(scheme_path):
        symbol = entry["symbol"]
        kind = entry["kind"]

        if kind == "n" and entry["note"]:
            notes_by_symbol.setdefault(symbol, []).append(entry["note"])
            continue

        if kind not in _INCLUDE_KINDS:
            continue
        if not entry["title"]:
            continue

        rank = _KIND_RANK.get(kind, 0)
        existing = by_symbol.get(symbol)
        if existing is not None and existing["_rank"] >= rank:
            continue

        def_fields = definitions.get(symbol, _empty_definition_fields())
        by_symbol[symbol] = {
            "_rank": rank,
            "ipc_code": format_ipc_symbol(symbol),
            "ipc_code_raw": symbol,
            "kind": kind,
            "entry_type": entry["entry_type"],
            "edition": entry["edition"],
            "title": entry["title"],
            "scheme_note": None,  # filled after pass
            **def_fields,
        }

    rows: list[dict[str, Any]] = []
    for symbol, row in by_symbol.items():
        notes = notes_by_symbol.get(symbol)
        row["scheme_note"] = " ".join(notes) if notes else None
        row.pop("_rank", None)
        rows.append(row)

    # Stable order: raw symbol sorts roughly by hierarchy for short codes,
    # and padded 14-char symbols sort lexicographically within subclasses.
    rows.sort(key=lambda r: r["ipc_code_raw"])
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build IPC code catalog JSONL from WIPO scheme + definitions XMLs."
    )
    p.add_argument(
        "--ipc-dir",
        type=Path,
        default=DEFAULT_IPC_DIR,
        help=f"Directory containing EN_ipc_*.xml (default: {DEFAULT_IPC_DIR})",
    )
    p.add_argument(
        "--edition",
        default=DEFAULT_EDITION,
        help=f"IPC edition stamp in filenames (default: {DEFAULT_EDITION})",
    )
    p.add_argument(
        "--scheme",
        type=Path,
        default=None,
        help="Override path to scheme XML",
    )
    p.add_argument(
        "--definitions",
        type=Path,
        default=None,
        help="Override path to definitions XML",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <ipc-dir>/ipc_codes_<edition>.jsonl)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ipc_dir: Path = args.ipc_dir
    edition: str = args.edition
    scheme_path = args.scheme or (ipc_dir / f"EN_ipc_scheme_{edition}.xml")
    definitions_path = args.definitions or (
        ipc_dir / f"EN_ipc_definitions_{edition}.xml"
    )
    output_path = args.output or (ipc_dir / f"ipc_codes_{edition}.jsonl")

    for path in (scheme_path, definitions_path):
        if not path.is_file():
            print(f"error: missing input file: {path}", file=sys.stderr)
            return 1

    rows = build_catalog(scheme_path, definitions_path)
    write_jsonl(output_path, rows)

    n_def = sum(1 for r in rows if r["has_definition_entry"])
    n_stmt = sum(1 for r in rows if r["definition_statement"])
    print(
        f"Wrote {len(rows)} codes → {output_path} "
        f"({n_def} with definition entry, {n_stmt} with definition_statement)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
