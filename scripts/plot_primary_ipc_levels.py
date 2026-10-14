#!/usr/bin/env python3
"""Primary IPC hierarchy-level counts for patent_search_clean.

Office ``primary_ipc`` is not always a leaf. This script looks each primary
up in the WIPO IPC catalog, records the scheme depth (section → n-dot
subgroup), and whether that place has narrower children.

Writes ``data/tables/primary_ipc_level_counts.csv`` and
``data/plots/primary_ipc_level_counts.png``.
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
sys.path.insert(0, str(REPO_ROOT / "instruction-generation" / "src"))

from ipc_lookup import IPCLookup, format_ipc_symbol  # noqa: E402
from jsonl_gz import iter_shard_records  # noqa: E402

DEFAULT_INPUT = REPO_ROOT / "data" / "derived" / "patent_search_clean"
DEFAULT_IPC = REPO_ROOT / "data" / "ipc-codes" / "ipc_codes_20260101.jsonl"
DEFAULT_TABLES = REPO_ROOT / "data" / "tables"
DEFAULT_PLOTS = REPO_ROOT / "data" / "plots"

# Scheme kind → display label, in hierarchy order.
LEVELS: tuple[tuple[str, str], ...] = (
    ("s", "section"),
    ("c", "class"),
    ("u", "subclass"),
    ("m", "main group"),
    *[(str(i), f"{i}-dot subgroup") for i in range(1, 10)],
    ("missing", "missing primary"),
    ("unknown", "not in 2026 catalog"),
)
_CATALOG_KINDS = frozenset(k for k, _ in LEVELS if k not in {"missing", "unknown"})


def canonicalize_ipc(code: str) -> str:
    """Match office symbols to catalog keys (strip spaces; drop trailing subgroup 0s).

    The 2026 JSONL stores ``C07K16/3`` for office ``C07K16/30``; ``/00`` is kept.
    """
    n = code.strip().upper().replace(" ", "")
    if not n:
        return ""
    if len(n) == 14 and n[4:].isdigit():
        n = format_ipc_symbol(n)
    if "/" in n:
        head, tail = n.split("/", 1)
        if tail != "00":
            n = f"{head}/{tail.rstrip('0') or '00'}"
    return n


def resolve_primary(raw: str, lookup: IPCLookup) -> tuple[str, str, bool | None]:
    """Return ``(kind, catalog_code, is_leaf)``.

    ``is_leaf`` is None when the place is missing or unknown to the catalog.
    """
    code = canonicalize_ipc(raw)
    if not code:
        return "missing", "", None
    entry = lookup.get(code)
    if entry is None or entry.kind not in _CATALOG_KINDS:
        return "unknown", code, None
    return entry.kind, entry.ipc_code, not bool(lookup.children_of(entry.ipc_code))


def collect(input_dir: Path, lookup: IPCLookup) -> dict[str, object]:
    patents_by_kind: Counter[str] = Counter()
    patents_leaf_by_kind: Counter[str] = Counter()
    patents_internal_by_kind: Counter[str] = Counter()
    codes_by_kind: dict[str, set[str]] = {k: set() for k, _ in LEVELS}
    codes_leaf_by_kind: dict[str, set[str]] = {k: set() for k, _ in LEVELS}
    codes_internal_by_kind: dict[str, set[str]] = {k: set() for k, _ in LEVELS}

    n = 0
    for data in iter_shard_records(input_dir, include_open_jsonl=False):
        n += 1
        kind, code, is_leaf = resolve_primary(str(data.get("primary_ipc") or ""), lookup)
        patents_by_kind[kind] += 1
        if code:
            codes_by_kind[kind].add(code)
        if is_leaf is True:
            patents_leaf_by_kind[kind] += 1
            if code:
                codes_leaf_by_kind[kind].add(code)
        elif is_leaf is False:
            patents_internal_by_kind[kind] += 1
            if code:
                codes_internal_by_kind[kind].add(code)

    return {
        "n": n,
        "patents_by_kind": patents_by_kind,
        "patents_leaf_by_kind": patents_leaf_by_kind,
        "patents_internal_by_kind": patents_internal_by_kind,
        "codes_by_kind": codes_by_kind,
        "codes_leaf_by_kind": codes_leaf_by_kind,
        "codes_internal_by_kind": codes_internal_by_kind,
    }


def _rows(stats: dict[str, object]) -> list[dict[str, object]]:
    patents: Counter[str] = stats["patents_by_kind"]  # type: ignore[assignment]
    leaf: Counter[str] = stats["patents_leaf_by_kind"]  # type: ignore[assignment]
    internal: Counter[str] = stats["patents_internal_by_kind"]  # type: ignore[assignment]
    codes: dict[str, set[str]] = stats["codes_by_kind"]  # type: ignore[assignment]
    codes_leaf: dict[str, set[str]] = stats["codes_leaf_by_kind"]  # type: ignore[assignment]
    codes_internal: dict[str, set[str]] = stats["codes_internal_by_kind"]  # type: ignore[assignment]
    n = int(stats["n"])  # type: ignore[arg-type]
    rows: list[dict[str, object]] = []
    for kind, label in LEVELS:
        n_patents = patents.get(kind, 0)
        if n_patents == 0 and kind not in {"missing", "unknown"}:
            continue
        rows.append(
            {
                "level": label,
                "kind": kind,
                "n_patents": n_patents,
                "pct_patents": (100.0 * n_patents / n) if n else 0.0,
                "n_patents_leaf": leaf.get(kind, 0),
                "n_patents_internal": internal.get(kind, 0),
                "n_unique_codes": len(codes.get(kind, ())),
                "n_unique_codes_leaf": len(codes_leaf.get(kind, ())),
                "n_unique_codes_internal": len(codes_internal.get(kind, ())),
            }
        )
    return rows


def write_csv(path: Path, stats: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "level",
        "kind",
        "n_patents",
        "pct_patents",
        "n_patents_leaf",
        "n_patents_internal",
        "n_unique_codes",
        "n_unique_codes_leaf",
        "n_unique_codes_internal",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in _rows(stats):
            out = dict(row)
            out["pct_patents"] = f"{float(row['pct_patents']):.4f}"
            writer.writerow(out)


def write_plot(path: Path, stats: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in _rows(stats) if int(r["n_patents"]) > 0]
    labels = [str(r["level"]) for r in rows]
    leaf = [int(r["n_patents_leaf"]) for r in rows]
    internal = [int(r["n_patents_internal"]) for r in rows]
    unresolved = [
        int(r["n_patents"]) - int(r["n_patents_leaf"]) - int(r["n_patents_internal"])
        for r in rows
    ]
    n = int(stats["n"])  # type: ignore[arg-type]
    n_leaf = sum(leaf)
    n_internal = sum(internal)
    xs = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(xs, leaf, color="#293241", label=f"Leaf ({n_leaf:,})")
    ax.bar(
        xs,
        internal,
        bottom=leaf,
        color="#3D5A80",
        label=f"Has narrower children ({n_internal:,})",
    )
    if any(unresolved):
        bottom = [a + b for a, b in zip(leaf, internal)]
        ax.bar(
            xs,
            unresolved,
            bottom=bottom,
            color="#98C1D9",
            label=f"Unresolved ({sum(unresolved):,})",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Number of patents")
    ax.set_xlabel("WIPO IPC 2026 scheme level of office primary_ipc")
    ax.set_title(
        f"Primary IPC depth (n={n:,}; leaf = no child place in catalog)"
    )
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
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
        "--ipc-jsonl",
        type=Path,
        default=DEFAULT_IPC,
        help=f"WIPO IPC catalog JSONL (default: {DEFAULT_IPC})",
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
    return p


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = _resolve(args.input_dir)
    ipc_jsonl = _resolve(args.ipc_jsonl)
    tables_dir = _resolve(args.tables_dir)
    plots_dir = _resolve(args.plots_dir)

    if not input_dir.is_dir():
        print(f"error: input dir missing: {input_dir}", file=sys.stderr)
        return 1
    if not ipc_jsonl.is_file():
        print(f"error: IPC catalog missing: {ipc_jsonl}", file=sys.stderr)
        return 1

    lookup = IPCLookup.from_jsonl(ipc_jsonl)
    print(f"Scanning {input_dir} (catalog entries={len(lookup):,})…", flush=True)
    stats = collect(input_dir, lookup)
    n = int(stats["n"])  # type: ignore[arg-type]
    if n == 0:
        print(f"no records in {input_dir}", file=sys.stderr)
        return 1

    table_path = tables_dir / "primary_ipc_level_counts.csv"
    plot_path = plots_dir / "primary_ipc_level_counts.png"
    write_csv(table_path, stats)
    write_plot(plot_path, stats)

    print(f"patents={n:,}")
    for row in _rows(stats):
        print(
            f"  {row['level']}: patents={row['n_patents']:,} "
            f"({float(row['pct_patents']):.2f}%) "
            f"leaf={row['n_patents_leaf']:,} internal={row['n_patents_internal']:,} "
            f"codes={row['n_unique_codes']:,}"
        )
    print(f"plot → {plot_path}")
    print(f"table → {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
