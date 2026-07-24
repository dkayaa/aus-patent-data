#!/usr/bin/env python3
"""Analyze patent_search_clean interim JSONL.GZ → CSV tables and plots.

Per-patent metrics use one primary published document (prefer B* over A*) so
A1/B2 versions of the same application are not double-counted.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Headless-friendly plotting
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scrape" / "src"))

from jsonl_gz import iter_shard_records  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "data" / "interim" / "patent_search_clean"
DEFAULT_TABLES = REPO_ROOT / "data" / "tables"
DEFAULT_PLOTS = REPO_ROOT / "data" / "plots"

# Higher rank = preferred published document for patent-level stats.
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


def _stats(values: list[float | int]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "min": "", "max": "", "mean": ""}
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.fmean(values), 2),
    }


def _write_summary_csv(path: Path, metric: str, stats: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "n", "min", "max", "mean"])
        writer.writeheader()
        writer.writerow(
            {
                "metric": metric,
                "n": stats["n"],
                "min": stats["min"],
                "max": stats["max"],
                "mean": stats["mean"],
            }
        )


def _write_ipc_csv(path: Path, counts: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ipcr_classification", "n_patents"])
        writer.writeheader()
        for label, n in counts.most_common():
            writer.writerow({"ipcr_classification": label, "n_patents": n})


def collect_metrics(input_dir: Path) -> dict[str, Any]:
    claim_char_lengths: list[int] = []
    abstract_char_lengths: list[int] = []
    claims_per_patent: list[int] = []
    # Per patent: (num_claims, mean_claim_chars) for 2D hist; skip if no claims
    claim_chars_vs_num: list[tuple[int, float]] = []
    # Per patent: (num_claims, abstract_chars) when abstract is non-empty
    abstract_chars_vs_num: list[tuple[int, int]] = []
    ipc_counts: Counter[str] = Counter()
    n_patents = 0
    n_with_primary_claims = 0
    n_with_primary_abstract = 0

    for data in iter_shard_records(input_dir, include_open_jsonl=False):
        n_patents += 1

        for code in data.get("ipcrClassification") or []:
            if isinstance(code, str) and code.strip():
                ipc_counts[code.strip()] += 1

        docs_raw = data.get("publishedDocuments") or []
        docs = [d for d in docs_raw if isinstance(d, dict)]
        primary = _select_primary_document(docs)
        if primary is None:
            claims_per_patent.append(0)
            continue

        claims = primary.get("claims") if isinstance(primary.get("claims"), list) else []
        claim_texts = [c for c in claims if isinstance(c, str) and c.strip()]
        n_claims = len(claim_texts)
        claims_per_patent.append(n_claims)

        if n_claims:
            n_with_primary_claims += 1
            lengths = [len(c) for c in claim_texts]
            claim_char_lengths.extend(lengths)
            claim_chars_vs_num.append((n_claims, statistics.fmean(lengths)))

        abstract = primary.get("abstract") if isinstance(primary.get("abstract"), str) else ""
        abstract = abstract.strip()
        if abstract:
            n_with_primary_abstract += 1
            abstract_len = len(abstract)
            abstract_char_lengths.append(abstract_len)
            abstract_chars_vs_num.append((n_claims, abstract_len))

    return {
        "n_patents": n_patents,
        "n_with_primary_claims": n_with_primary_claims,
        "n_with_primary_abstract": n_with_primary_abstract,
        "claim_char_lengths": claim_char_lengths,
        "abstract_char_lengths": abstract_char_lengths,
        "claims_per_patent": claims_per_patent,
        "claim_chars_vs_num": claim_chars_vs_num,
        "abstract_chars_vs_num": abstract_chars_vs_num,
        "ipc_counts": ipc_counts,
    }


def write_tables(metrics: dict[str, Any], tables_dir: Path) -> list[Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    path = tables_dir / "chars_per_claim.csv"
    _write_summary_csv(path, "chars_per_claim", _stats(metrics["claim_char_lengths"]))
    written.append(path)

    path = tables_dir / "chars_per_abstract.csv"
    _write_summary_csv(
        path, "chars_per_abstract", _stats(metrics["abstract_char_lengths"])
    )
    written.append(path)

    path = tables_dir / "num_claims_per_patent.csv"
    _write_summary_csv(
        path, "num_claims_per_patent", _stats(metrics["claims_per_patent"])
    )
    written.append(path)

    path = tables_dir / "ipc_label_patent_counts.csv"
    _write_ipc_csv(path, metrics["ipc_counts"])
    written.append(path)

    return written


def write_plots(metrics: dict[str, Any], plots_dir: Path) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1) Claim char length histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    lengths = metrics["claim_char_lengths"]
    if lengths:
        ax.hist(lengths, bins=min(40, max(10, int(len(lengths) ** 0.5))), color="#3d5a80", edgecolor="white")
    ax.set_xlabel("Characters per claim")
    ax.set_ylabel("Count")
    ax.set_title("Claim character length")
    fig.tight_layout()
    path = plots_dir / "hist_claim_char_length.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 2) Num claims per patent histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    n_claims = metrics["claims_per_patent"]
    if n_claims:
        max_n = max(n_claims)
        bins = range(0, max_n + 2)
        ax.hist(n_claims, bins=bins, color="#ee6c4d", edgecolor="white", align="left")
    ax.set_xlabel("Number of claims (primary published document)")
    ax.set_ylabel("Number of patents")
    ax.set_title("Claims per patent")
    fig.tight_layout()
    path = plots_dir / "hist_num_claims_per_patent.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 3) 2D: mean claim char length vs num claims
    fig, ax = plt.subplots(figsize=(8, 5))
    pairs = metrics["claim_chars_vs_num"]
    if pairs:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        hb = ax.hexbin(xs, ys, gridsize=20, cmap="viridis", mincnt=1)
        fig.colorbar(hb, ax=ax, label="Patents")
    ax.set_xlabel("Number of claims")
    ax.set_ylabel("Mean characters per claim")
    ax.set_title("Claim length vs number of claims (per patent)")
    fig.tight_layout()
    path = plots_dir / "hist2d_claim_chars_vs_num_claims.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 4) Abstract char length histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    abs_lengths = metrics["abstract_char_lengths"]
    if abs_lengths:
        ax.hist(
            abs_lengths,
            bins=min(40, max(10, int(len(abs_lengths) ** 0.5))),
            color="#3d5a80",
            edgecolor="white",
        )
    ax.set_xlabel("Characters per abstract")
    ax.set_ylabel("Count")
    ax.set_title("Abstract character length")
    fig.tight_layout()
    path = plots_dir / "hist_abstract_char_length.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 5) 2D: abstract char length vs num claims
    fig, ax = plt.subplots(figsize=(8, 5))
    abs_pairs = metrics["abstract_chars_vs_num"]
    if abs_pairs:
        xs = [p[0] for p in abs_pairs]
        ys = [p[1] for p in abs_pairs]
        hb = ax.hexbin(xs, ys, gridsize=20, cmap="viridis", mincnt=1)
        fig.colorbar(hb, ax=ax, label="Patents")
    ax.set_xlabel("Number of claims")
    ax.set_ylabel("Characters per abstract")
    ax.set_title("Abstract length vs number of claims (per patent)")
    fig.tight_layout()
    path = plots_dir / "hist2d_abstract_chars_vs_num_claims.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 6) IPC label vs patent count (top 40 for readability)
    fig, ax = plt.subplots(figsize=(12, 6))
    counts: Counter[str] = metrics["ipc_counts"]
    top = counts.most_common(40)
    if top:
        labels = [item[0] for item in top]
        values = [item[1] for item in top]
        ax.bar(labels, values, color="#293241")
        ax.set_xlabel("IPC classification")
        ax.set_ylabel("Number of patents")
        ax.tick_params(axis="x", labelrotation=90)
        title_n = len(top)
        total = len(counts)
        suffix = f" (top {title_n} of {total})" if total > title_n else ""
        ax.set_title(f"IPC labels vs number of patents{suffix}")
    else:
        ax.set_title("IPC labels vs number of patents (no data)")
    fig.tight_layout()
    path = plots_dir / "ipc_label_patent_counts.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze patent_search_clean → data/tables CSVs and data/plots PNGs."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Clean interim JSONL.GZ dir (default: {DEFAULT_INPUT})",
    )
    p.add_argument(
        "--tables-dir",
        type=Path,
        default=DEFAULT_TABLES,
        help=f"CSV output dir (default: {DEFAULT_TABLES})",
    )
    p.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS,
        help=f"Plot output dir (default: {DEFAULT_PLOTS})",
    )
    p.add_argument(
        "--tables-only",
        action="store_true",
        help="Write CSVs only (skip plots)",
    )
    p.add_argument(
        "--plots-only",
        action="store_true",
        help="Write plots only (skip CSVs)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = args.input_dir
    if not input_dir.is_absolute():
        input_dir = REPO_ROOT / input_dir
    tables_dir = args.tables_dir if args.tables_dir.is_absolute() else REPO_ROOT / args.tables_dir
    plots_dir = args.plots_dir if args.plots_dir.is_absolute() else REPO_ROOT / args.plots_dir

    if not input_dir.is_dir():
        print(f"error: input_dir does not exist: {input_dir}", file=sys.stderr)
        return 1

    metrics = collect_metrics(input_dir)
    print(
        f"patents={metrics['n_patents']} "
        f"with_claims={metrics['n_with_primary_claims']} "
        f"with_abstract={metrics['n_with_primary_abstract']} "
        f"ipc_labels={len(metrics['ipc_counts'])}"
    )

    if not args.plots_only:
        for path in write_tables(metrics, tables_dir):
            print(f"wrote {path.relative_to(REPO_ROOT)}")

    if not args.tables_only:
        for path in write_plots(metrics, plots_dir):
            print(f"wrote {path.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
