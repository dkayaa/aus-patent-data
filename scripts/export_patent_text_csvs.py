#!/usr/bin/env python3
"""Export patent_search_clean shards to flat text CSVs for classification.

Writes three files from the primary published document (prefer B* over A*):

- abstracts.csv[.gz]: one row per application
  ``application_number,invention_title,abstract``
- first_claims.csv[.gz]: one row per application (first claim only)
  ``application_number,invention_title,claim``
- claims.csv[.gz]: one row per claim
  ``application_number,invention_title,claim``

Pass ``--gzip`` to write ``*.csv.gz`` instead of plain ``*.csv``.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterator, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scrape" / "src"))

from jsonl_gz import iter_shard_records  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "data" / "interim" / "patent_search_clean"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "interim" / "patent_search_text"

# Higher rank = preferred published document (same rule as analyze).
_DOC_TYPE_RANK = {
    "B9": 60,
    "B2": 50,
    "B1": 40,
    "C1": 30,
    "A2": 20,
    "A1": 10,
}


def _select_primary_document(docs: list[dict[str, Any]]) -> dict[str, Any] | None:
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


def _first_claim_text(claims: Any) -> str:
    if not isinstance(claims, list):
        return ""
    for claim in claims:
        if isinstance(claim, str) and claim.strip():
            return claim.strip()
    return ""


def _iter_limited_records(
    input_dir: Path, *, max_records: int | None
) -> Iterator[dict[str, Any]]:
    n = 0
    for record in iter_shard_records(input_dir, include_open_jsonl=False):
        yield record
        n += 1
        if max_records is not None and n >= max_records:
            break


def _csv_paths(output_dir: Path, *, use_gzip: bool) -> tuple[Path, Path, Path]:
    suffix = ".csv.gz" if use_gzip else ".csv"
    return (
        output_dir / f"abstracts{suffix}",
        output_dir / f"first_claims{suffix}",
        output_dir / f"claims{suffix}",
    )


def _open_text_write(path: Path, *, use_gzip: bool) -> TextIO:
    if use_gzip:
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def export_text_csvs(
    input_dir: Path,
    output_dir: Path,
    *,
    max_records: int | None = None,
    use_gzip: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    abstracts_path, first_claims_path, claims_path = _csv_paths(
        output_dir, use_gzip=use_gzip
    )

    n_apps = 0
    n_abstract_rows = 0
    n_first_claim_rows = 0
    n_claim_rows = 0
    n_without_primary = 0
    n_without_first_claim = 0

    with ExitStack() as stack:
        abstracts_f = stack.enter_context(
            _open_text_write(abstracts_path, use_gzip=use_gzip)
        )
        first_claims_f = stack.enter_context(
            _open_text_write(first_claims_path, use_gzip=use_gzip)
        )
        claims_f = stack.enter_context(_open_text_write(claims_path, use_gzip=use_gzip))

        abstracts_writer = csv.DictWriter(
            abstracts_f,
            fieldnames=["application_number", "invention_title", "abstract"],
        )
        first_claims_writer = csv.DictWriter(
            first_claims_f,
            fieldnames=["application_number", "invention_title", "claim"],
        )
        claims_writer = csv.DictWriter(
            claims_f,
            fieldnames=["application_number", "invention_title", "claim"],
        )
        abstracts_writer.writeheader()
        first_claims_writer.writeheader()
        claims_writer.writeheader()

        for record in _iter_limited_records(input_dir, max_records=max_records):
            n_apps += 1
            app_no = str(record.get("application_number") or "").strip()
            title = str(record.get("inventionTitle") or "").strip()
            docs = record.get("publishedDocuments") or []
            if not isinstance(docs, list):
                docs = []
            primary = _select_primary_document(
                [d for d in docs if isinstance(d, dict)]
            )
            if primary is None:
                n_without_primary += 1
                n_without_first_claim += 1
                abstracts_writer.writerow(
                    {
                        "application_number": app_no,
                        "invention_title": title,
                        "abstract": "",
                    }
                )
                first_claims_writer.writerow(
                    {
                        "application_number": app_no,
                        "invention_title": title,
                        "claim": "",
                    }
                )
                n_abstract_rows += 1
                n_first_claim_rows += 1
                continue

            abstract = primary.get("abstract") if isinstance(primary.get("abstract"), str) else ""
            abstracts_writer.writerow(
                {
                    "application_number": app_no,
                    "invention_title": title,
                    "abstract": abstract.strip(),
                }
            )
            n_abstract_rows += 1

            claims = primary.get("claims") if isinstance(primary.get("claims"), list) else []
            first_claim = _first_claim_text(claims)
            if not first_claim:
                n_without_first_claim += 1
            first_claims_writer.writerow(
                {
                    "application_number": app_no,
                    "invention_title": title,
                    "claim": first_claim,
                }
            )
            n_first_claim_rows += 1

            for claim in claims:
                if not isinstance(claim, str):
                    continue
                text = claim.strip()
                if not text:
                    continue
                claims_writer.writerow(
                    {
                        "application_number": app_no,
                        "invention_title": title,
                        "claim": text,
                    }
                )
                n_claim_rows += 1

    return {
        "applications": n_apps,
        "abstract_rows": n_abstract_rows,
        "first_claim_rows": n_first_claim_rows,
        "claim_rows": n_claim_rows,
        "without_primary_doc": n_without_primary,
        "without_first_claim": n_without_first_claim,
        "abstracts_path": str(abstracts_path),
        "first_claims_path": str(first_claims_path),
        "claims_path": str(claims_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"patent_search_clean shard dir (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "directory for abstracts / first_claims / claims CSV "
            f"(default: {DEFAULT_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="max applications to export (default: all)",
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        help="write abstracts.csv.gz, first_claims.csv.gz, claims.csv.gz",
    )
    args = parser.parse_args(argv)

    if args.max_records is not None and args.max_records < 1:
        parser.error("--max-records must be >= 1")

    input_dir = args.input_dir
    if not input_dir.is_absolute():
        input_dir = REPO_ROOT / input_dir
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    if not input_dir.is_dir():
        print(f"error: input dir not found: {input_dir}", file=sys.stderr)
        return 1

    stats = export_text_csvs(
        input_dir,
        output_dir,
        max_records=args.max_records,
        use_gzip=args.gzip,
    )
    print(
        f"wrote {stats['abstracts_path']} ({stats['abstract_rows']} rows), "
        f"{stats['first_claims_path']} ({stats['first_claim_rows']} rows), "
        f"{stats['claims_path']} ({stats['claim_rows']} rows) "
        f"from {stats['applications']} applications"
        + (
            f" ({stats['without_primary_doc']} without primary doc, "
            f"{stats['without_first_claim']} without first claim)"
            if stats["without_primary_doc"] or stats["without_first_claim"]
            else ""
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
