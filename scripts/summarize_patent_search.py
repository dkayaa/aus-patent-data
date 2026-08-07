#!/usr/bin/env python3
"""Summarize patent_search derived JSONL.GZ payloads in a folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scrape" / "src"))

from jsonl_gz import iter_shard_records  # noqa: E402

DEFAULT_DIR = REPO_ROOT / "data" / "derived" / "patent_search"


def summarize(input_dir: Path) -> dict[str, float | int]:
    n_patents = 0
    n_with_published = 0
    n_with_ipcr = 0
    published_counts: list[int] = []
    claims_lengths: list[int] = []
    abstract_lengths: list[int] = []

    for data in iter_shard_records(input_dir, include_open_jsonl=True):
        n_patents += 1
        response = data.get("response")
        if not isinstance(response, dict):
            response = {}

        published = response.get("publishedDocuments")
        if not isinstance(published, list):
            published = []
        published_counts.append(len(published))
        if published:
            n_with_published += 1

        for doc in published:
            if not isinstance(doc, dict):
                continue
            claims = doc.get("claimsText")
            if isinstance(claims, str):
                claims_lengths.append(len(claims))
            abstract = doc.get("abstractText")
            if isinstance(abstract, str):
                abstract_lengths.append(len(abstract))

        biblio = response.get("bibliographicData")
        if isinstance(biblio, dict) and "ipcrClassification" in biblio:
            n_with_ipcr += 1

    def _mean(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "n_patent_objects": n_patents,
        "avg_published_documents": _mean(published_counts),
        "avg_claims_text_chars": _mean(claims_lengths),
        "avg_abstract_text_chars": _mean(abstract_lengths),
        "n_with_published_document": n_with_published,
        "n_with_ipcr_classification": n_with_ipcr,
        "n_claims_text_samples": len(claims_lengths),
        "n_abstract_text_samples": len(abstract_lengths),
    }


def _print_table(stats: dict[str, float | int], input_dir: Path) -> None:
    rows = [
        ("# patent objects", f"{stats['n_patent_objects']:,}"),
        (
            "avg # publishedDocuments per patent",
            f"{stats['avg_published_documents']:.3f}",
        ),
        (
            "avg claimsText length (chars)",
            f"{stats['avg_claims_text_chars']:.1f}"
            f"  (n={stats['n_claims_text_samples']:,})",
        ),
        (
            "avg abstractText length (chars)",
            f"{stats['avg_abstract_text_chars']:.1f}"
            f"  (n={stats['n_abstract_text_samples']:,})",
        ),
        (
            "# patents with ≥1 publishedDocument",
            f"{stats['n_with_published_document']:,}",
        ),
        (
            "# patents with ipcrClassification",
            f"{stats['n_with_ipcr_classification']:,}",
        ),
    ]
    label_w = max(len(label) for label, _ in rows)
    print(f"folder: {input_dir}")
    print()
    print(f"{'metric'.ljust(label_w)}  value")
    print(f"{'-' * label_w}  -----")
    for label, value in rows:
        print(f"{label.ljust(label_w)}  {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize patent_search derived JSONL.GZ payloads."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Folder of part-*.jsonl.gz shards (default: {DEFAULT_DIR})",
    )
    args = parser.parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        print(f"error: not a directory: {input_dir}", file=sys.stderr)
        return 1

    stats = summarize(input_dir)
    _print_table(stats, input_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
