"""Read cleaned patent shards and extract fields for instruction tasks."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scrape" / "src"))

from jsonl_gz import iter_shard_records  # noqa: E402

_DOC_TYPE_RANK = {
    "B9": 60,
    "B2": 50,
    "B1": 40,
    "C1": 30,
    "A2": 20,
    "A1": 10,
}


@dataclass(frozen=True)
class PatentText:
    application_number: str
    invention_title: str
    primary_ipc: str
    abstract: str
    claims: list[str]
    claims_text: str
    claim_1: str
    document_type: str


def select_primary_document(docs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not docs:
        return None

    def sort_key(doc: dict[str, Any]) -> tuple[int, int, int]:
        dtype = str(doc.get("documentTypeCode") or "")
        claims = doc.get("claims") if isinstance(doc.get("claims"), list) else []
        abstract = doc.get("abstract") if isinstance(doc.get("abstract"), str) else ""
        return (
            _DOC_TYPE_RANK.get(dtype, 0),
            len(claims),
            len(abstract.strip()),
        )

    return max(docs, key=sort_key)


def join_claims(claims: list[str]) -> str:
    return "\n\n".join(c.strip() for c in claims if isinstance(c, str) and c.strip())


def extract_patent_text(record: dict[str, Any]) -> PatentText | None:
    app = str(record.get("application_number") or "").strip()
    if not app:
        return None

    docs_raw = record.get("publishedDocuments") or []
    docs = [d for d in docs_raw if isinstance(d, dict)]
    primary = select_primary_document(docs)
    if primary is None:
        return None

    if primary.get("claims_parse_ok") is False:
        return None

    abstract = primary.get("abstract") if isinstance(primary.get("abstract"), str) else ""
    abstract = abstract.strip()
    claims_raw = primary.get("claims") if isinstance(primary.get("claims"), list) else []
    claims = [c.strip() for c in claims_raw if isinstance(c, str) and c.strip()]
    if not abstract or not claims:
        return None

    primary_ipc = str(record.get("primary_ipc") or "").strip()
    title = str(record.get("inventionTitle") or "").strip()
    return PatentText(
        application_number=app,
        invention_title=title,
        primary_ipc=primary_ipc,
        abstract=abstract,
        claims=claims,
        claims_text=join_claims(claims),
        claim_1=claims[0],
        document_type=str(primary.get("documentTypeCode") or ""),
    )


def iter_patent_texts(
    patents_dir: Path,
    *,
    limit: int | None = None,
    skip_ids: set[str] | None = None,
) -> Iterator[PatentText]:
    skip = skip_ids or set()
    n = 0
    for record in iter_shard_records(patents_dir, include_open_jsonl=False):
        text = extract_patent_text(record)
        if text is None:
            continue
        if text.application_number in skip:
            continue
        yield text
        n += 1
        if limit is not None and n >= limit:
            break
