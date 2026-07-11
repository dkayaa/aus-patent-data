#!/usr/bin/env python3
"""Summarize patent_search interim JSON payloads in a folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "data" / "interim" / "patent_search"


def _iter_patent_json_paths(input_dir: Path) -> list[Path]:
    # Skip macOS AppleDouble sidecars (._*.json) common on external volumes.
    return sorted(
        p for p in input_dir.glob("*.json") if not p.name.startswith("._")
    )


def _load_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"warn: skip {path.name}: {exc}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"warn: skip {path.name}: root is not an object", file=sys.stderr)
        return None
    return data


def summarize(input_dir: Path) -> dict[str, float | int]:
    paths = _iter_patent_json_paths(input_dir)
    n_patents = 0
    n_with_published = 0
    n_with_ipcr = 0
    published_counts: list[int] = []
    claims_lengths: list[int] = []
    abstract_lengths: list[int] = []

    for path in paths:
        data = _load_json(path)
        if data is None:
            continue
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
        description="Summarize patent_search interim JSON payloads."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_DIR,
        help=f"Folder of *.json patent payloads (default: {DEFAULT_DIR})",
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
