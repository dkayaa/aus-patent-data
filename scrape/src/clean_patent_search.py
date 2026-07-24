"""Reshape Patent Search interim JSONL shards into a flatter cleaned set.

Reads ``part-*.jsonl.gz`` from ``data/interim/patent_search/``, parses
``claimsText`` into claim lists, and writes mirrored
``part-*.jsonl.gz`` under ``data/interim/patent_search_clean/`` plus a
``summary.json`` manifest (counts and parse failures).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from jsonl_gz import (
    iter_jsonl_gz_shards,
    iter_records,
    parse_shard_index,
    shard_jsonl_gz_path,
    write_jsonl_gz_records,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "scrape" / "config" / "clean_patent_search.yaml"
SUMMARY_FILENAME = "summary.json"

_MIN_CLAIM_BODY_CHARS = 20
_WO_PCT_HEADER = re.compile(
    r"WO\s+\d{4}/\d+\s+PCT/[A-Z]{2}\d{4}/\d+",
    flags=re.IGNORECASE,
)
# Match the section header "CLAIMS" / "CLAIMS:" only — not the word "claim" in
# bodies like "The assembly of claim 1".
_CLAIMS_PREAMBLE = re.compile(
    r"\bCLAIMS\b:?|What is claimed is:?",
    flags=re.IGNORECASE,
)
_CLAIM_BOUNDARY = re.compile(r"(?:(?<=\s)|^)(\d{1,3})\.\s+")


@dataclass(frozen=True)
class CleanConfig:
    input_dir: Path
    output_dir: Path
    limit: int | None = None


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_config(path: Path) -> CleanConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    paths = raw.get("paths") or {}
    if "input_dir" not in paths or "output_dir" not in paths:
        raise ValueError(f"config {path} must set paths.input_dir and paths.output_dir")
    return CleanConfig(
        input_dir=_repo_path(paths["input_dir"]),
        output_dir=_repo_path(paths["output_dir"]),
    )


def write_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / SUMMARY_FILENAME
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)
    return path


def parse_claims(claims_text: str) -> tuple[list[str], bool]:
    """Split messy PDF/OCR claimsText into numbered claim strings.

    Returns ``(claims, claims_parse_ok)``. ``claims_parse_ok`` is true only when
    at least one claim is recovered and the first starts with ``1.``.
    """
    if not claims_text or not claims_text.strip():
        return [], False

    text = claims_text.replace("\x0c", " ").replace("\x0e", " ")
    text = _WO_PCT_HEADER.sub(" ", text)
    text = _CLAIMS_PREAMBLE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    matches = list(_CLAIM_BOUNDARY.finditer(text))
    if not matches:
        return [], False

    claims: list[str] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = re.sub(r"\s+", " ", text[start:end]).strip()
        # Boundary match is "N. "; require a real body after the number.
        body = chunk.split(".", 1)[-1].strip() if "." in chunk else ""
        if len(body) < _MIN_CLAIM_BODY_CHARS:
            continue
        claims.append(chunk)

    ok = bool(claims) and claims[0].startswith("1.")
    return claims, ok


def _entity_names(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    names: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = (item.get("legalEntityName") or "").strip()
        if name:
            names.append(name)
    return names


def _invention_title(bibliographic: dict[str, Any]) -> str:
    titles = bibliographic.get("inventionTitle") or []
    if not isinstance(titles, list):
        return ""
    for item in titles:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        if title:
            return title
    return ""


def _ipcr_codes(bibliographic: dict[str, Any]) -> list[str]:
    rows = bibliographic.get("ipcrClassification") or []
    if not isinstance(rows, list):
        return []
    codes: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = (row.get("classificationText") or "").strip()
        if text:
            codes.append(text)
    return codes


def _clean_published_document(doc: dict[str, Any]) -> dict[str, Any]:
    claims_text = doc.get("claimsText") or ""
    if not isinstance(claims_text, str):
        claims_text = ""
    claims, parse_ok = parse_claims(claims_text)
    abstract = doc.get("abstractText") or ""
    if not isinstance(abstract, str):
        abstract = ""
    file_name = doc.get("fileName") or ""
    if not isinstance(file_name, str):
        file_name = ""
    doc_type = doc.get("documentTypeCode") or ""
    if not isinstance(doc_type, str):
        doc_type = ""
    return {
        "documentTypeCode": doc_type,
        "fileName": file_name,
        "abstract": abstract.strip(),
        "claims": claims,
        "claims_parse_ok": parse_ok,
    }


def clean_record(raw: dict[str, Any]) -> dict[str, Any]:
    response = raw.get("response")
    if not isinstance(response, dict):
        response = {}
    bibliographic = response.get("bibliographicData")
    if not isinstance(bibliographic, dict):
        bibliographic = {}

    published_raw = response.get("publishedDocuments") or []
    published: list[dict[str, Any]] = []
    if isinstance(published_raw, list):
        for doc in published_raw:
            if isinstance(doc, dict):
                published.append(_clean_published_document(doc))

    application_number = raw.get("application_number") or bibliographic.get(
        "applicationNumber"
    )
    if not isinstance(application_number, str):
        application_number = "" if application_number is None else str(application_number)

    fetched_at = raw.get("fetched_at") or ""
    if not isinstance(fetched_at, str):
        fetched_at = ""

    def _str_field(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    return {
        "application_number": application_number.strip(),
        "fetched_at": fetched_at,
        "ipRightStatusCode": _str_field(response.get("ipRightStatusCode")),
        "inventionTitle": _invention_title(bibliographic),
        "ipcrClassification": _ipcr_codes(bibliographic),
        "patentApplicationType": _str_field(bibliographic.get("patentApplicationType")),
        "filedDate": _str_field(response.get("filedDate")),
        "priorityDate": _str_field(response.get("priorityDate")),
        "expiryDate": _str_field(response.get("expiryDate")),
        "applicant": _entity_names(response.get("applicant")),
        "inventors": _entity_names(response.get("inventors")),
        "publishedDocuments": published,
    }


def _tally_claims_stats(
    raw: dict[str, Any],
    record: dict[str, Any],
    application_number: str,
    *,
    docs_with_claims_text: int,
    docs_parse_ok: int,
    docs_parse_fail: int,
    claims_parse_failures: list[dict[str, str]],
) -> tuple[int, int, int]:
    response = raw.get("response") if isinstance(raw.get("response"), dict) else {}
    raw_docs = response.get("publishedDocuments") or []
    if not isinstance(raw_docs, list):
        return docs_with_claims_text, docs_parse_ok, docs_parse_fail

    for raw_doc, cleaned_doc in zip(raw_docs, record["publishedDocuments"]):
        if not isinstance(raw_doc, dict):
            continue
        claims_text = raw_doc.get("claimsText") or ""
        if not isinstance(claims_text, str) or not claims_text.strip():
            continue
        docs_with_claims_text += 1
        if cleaned_doc.get("claims_parse_ok"):
            docs_parse_ok += 1
        else:
            docs_parse_fail += 1
            failure = {
                "application_number": application_number,
                "documentTypeCode": str(raw_doc.get("documentTypeCode") or ""),
                "fileName": str(raw_doc.get("fileName") or ""),
            }
            claims_parse_failures.append(failure)
            logger.info(
                "claims_parse_ok=false %s:%s fileName=%s",
                failure["application_number"],
                failure["documentTypeCode"] or "?",
                failure["fileName"],
            )
    return docs_with_claims_text, docs_parse_ok, docs_parse_fail


def run(cfg: CleanConfig) -> int:
    if not cfg.input_dir.is_dir():
        logger.error("input_dir does not exist: %s", cfg.input_dir)
        return 1

    shard_paths = iter_jsonl_gz_shards(cfg.input_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    records_in = 0
    docs_with_claims_text = 0
    docs_parse_ok = 0
    docs_parse_fail = 0
    applications_without_published_docs = 0
    claims_parse_failures: list[dict[str, str]] = []
    skipped_records: list[dict[str, str]] = []
    shards_written = 0
    stop = False

    for shard_path in shard_paths:
        if stop:
            break
        shard_idx = parse_shard_index(shard_path)
        if shard_idx is None:
            continue

        cleaned_records: list[dict[str, Any]] = []
        for raw in iter_records(shard_path):
            if cfg.limit is not None and records_in >= cfg.limit:
                stop = True
                break
            records_in += 1

            record = clean_record(raw)
            application_number = record["application_number"]
            if not application_number:
                skipped += 1
                skipped_records.append(
                    {
                        "input_shard": shard_path.name,
                        "reason": "empty application_number",
                    }
                )
                continue

            if not record["publishedDocuments"]:
                applications_without_published_docs += 1

            (
                docs_with_claims_text,
                docs_parse_ok,
                docs_parse_fail,
            ) = _tally_claims_stats(
                raw,
                record,
                application_number,
                docs_with_claims_text=docs_with_claims_text,
                docs_parse_ok=docs_parse_ok,
                docs_parse_fail=docs_parse_fail,
                claims_parse_failures=claims_parse_failures,
            )
            cleaned_records.append(record)
            written += 1

        if cleaned_records:
            out_path = shard_jsonl_gz_path(cfg.output_dir, shard_idx)
            write_jsonl_gz_records(out_path, cleaned_records)
            shards_written += 1
            logger.info(
                "wrote %s records=%s",
                _display_path(out_path),
                len(cleaned_records),
            )

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_dir": _display_path(cfg.input_dir),
        "output_dir": _display_path(cfg.output_dir),
        "limit": cfg.limit,
        "shards_in": len(shard_paths),
        "shards_written": shards_written,
        "records_in": records_in,
        "written": written,
        "skipped": skipped,
        "applications_without_published_documents": (
            applications_without_published_docs
        ),
        "docs_with_claims_text": docs_with_claims_text,
        "claims_parse_ok": docs_parse_ok,
        "claims_parse_fail": docs_parse_fail,
        "claims_parse_failures": claims_parse_failures,
        "skipped_records": skipped_records,
    }
    summary_path = write_summary(cfg.output_dir, summary)

    logger.info(
        "done input=%s output=%s shards_in=%s records_in=%s written=%s skipped=%s "
        "docs_with_claims_text=%s claims_parse_ok=%s claims_parse_fail=%s "
        "summary=%s",
        _display_path(cfg.input_dir),
        _display_path(cfg.output_dir),
        len(shard_paths),
        records_in,
        written,
        skipped,
        docs_with_claims_text,
        docs_parse_ok,
        docs_parse_fail,
        _display_path(summary_path),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Clean Patent Search interim JSONL.GZ shards into analysis-ready "
            "JSONL.GZ shards."
        )
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config path (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Override paths.input_dir",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override paths.output_dir",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of input records to process",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Debug logging",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    cfg = load_config(args.config.resolve())
    if args.input_dir is not None:
        cfg = replace(cfg, input_dir=_repo_path(args.input_dir))
    if args.output_dir is not None:
        cfg = replace(cfg, output_dir=_repo_path(args.output_dir))
    if args.limit is not None:
        cfg = replace(cfg, limit=args.limit)
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
