#!/usr/bin/env python3
"""Histogram of distinct IPC codes per patent in patent_search_clean.

Counts symbols on ``ipcrClassification`` (normalized, de-duplicated). Writes
the full distribution to ``data/tables/ipc_codes_per_patent.csv`` and a bar
chart of 0, 1, 2, … through the 99th percentile to
``data/plots/hist_ipc_codes_per_patent.png``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scrape" / "src"))

from jsonl_gz import iter_shard_records  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "data" / "derived" / "patent_search_clean"
DEFAULT_TABLES = REPO_ROOT / "data" / "tables"
DEFAULT_PLOTS = REPO_ROOT / "data" / "plots"


def _normalize_ipc(code: str) -> str:
    return code.strip().upper().replace(" ", "")


def _n_ipc_codes(data: dict) -> int:
    seen: set[str] = set()
    for code in data.get("ipcrClassification") or []:
        if not isinstance(code, str):
            continue
        norm = _normalize_ipc(code)
        if norm:
            seen.add(norm)
    return len(seen)


def collect_histogram(input_dir: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for data in iter_shard_records(input_dir, include_open_jsonl=False):
        counts[_n_ipc_codes(data)] += 1
    return counts


def _percentile(hist: Counter[int], q: float) -> int:
    total = sum(hist.values())
    if total == 0:
        return 0
    target = (q / 100.0) * (total - 1)
    running = 0
    for k in range(0, max(hist) + 1):
        running += hist.get(k, 0)
        if running - 1 >= target:
            return k
    return max(hist)


def write_csv(path: Path, hist: Counter[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hi = max(hist) if hist else 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_ipc_codes", "n_patents"])
        writer.writeheader()
        for k in range(0, hi + 1):
            writer.writerow({"n_ipc_codes": k, "n_patents": hist.get(k, 0)})


def write_plot(path: Path, hist: Counter[int], *, xmax: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = list(range(0, xmax + 1))
    ys = [hist.get(k, 0) for k in xs]
    n_above = sum(v for k, v in hist.items() if k > xmax)
    n_patents = sum(hist.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(xs, ys, color="#293241", width=0.8, align="center")
    ax.set_xlabel("IPC codes per patent")
    ax.set_ylabel("Number of patents")
    ax.set_xticks(xs)
    title = "IPC codes per patent"
    if n_above:
        title += f" ({n_above:,} with more than {xmax} omitted)"
    elif n_patents:
        title += f" (n={n_patents:,})"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"patent_search_clean shard dir (default: {DEFAULT_INPUT})",
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
        help=f"PNG output dir (default: {DEFAULT_PLOTS})",
    )
    p.add_argument(
        "--xmax",
        type=int,
        default=None,
        help="Last bar on the chart (default: 99th percentile of codes per patent)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = args.input_dir if args.input_dir.is_absolute() else REPO_ROOT / args.input_dir
    tables_dir = args.tables_dir if args.tables_dir.is_absolute() else REPO_ROOT / args.tables_dir
    plots_dir = args.plots_dir if args.plots_dir.is_absolute() else REPO_ROOT / args.plots_dir

    hist = collect_histogram(input_dir)
    n = sum(hist.values())
    if n == 0:
        print(f"no records in {input_dir}", file=sys.stderr)
        return 1

    xmax = args.xmax if args.xmax is not None else _percentile(hist, 99)
    xmax = max(0, xmax)
    table_path = tables_dir / "ipc_codes_per_patent.csv"
    plot_path = plots_dir / "hist_ipc_codes_per_patent.png"
    write_csv(table_path, hist)
    write_plot(plot_path, hist, xmax=xmax)

    median = _percentile(hist, 50)
    mean = sum(k * v for k, v in hist.items()) / n
    print(f"patents={n:,} median={median} mean={mean:.2f} max={max(hist)}")
    print(f"plot bars 0–{xmax} → {plot_path}")
    print(f"full distribution → {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
