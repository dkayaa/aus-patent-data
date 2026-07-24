#!/usr/bin/env python3
"""Analyze patent_search_clean interim JSONL.GZ → CSV tables and plots.

Per-patent metrics use one primary published document (prefer B* over A*) so
A1/B2 versions of the same application are not double-counted.

Text lengths are measured in BERT WordPiece tokens (default:
``bert-base-uncased``, excluding special tokens).
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
DEFAULT_TOKENIZER = "bert-base-uncased"
BERT_MAX_POSITIONS = 512
_TOKENIZE_BATCH = 256

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


def _load_tokenizer(name: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required for BERT token counts. "
            "Install with: pip install -r requirements.txt"
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(name)
    # Allow counting full text lengths without the 512-position warning.
    tokenizer.model_max_length = int(1e9)
    return tokenizer


def _token_lengths(tokenizer: Any, texts: list[str]) -> list[int]:
    """WordPiece token counts excluding special tokens."""
    lengths: list[int] = []
    for i in range(0, len(texts), _TOKENIZE_BATCH):
        batch = texts[i : i + _TOKENIZE_BATCH]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


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


def _ipc_freq_of_freq(counts: Counter[str]) -> Counter[int]:
    """Map patents-per-code → how many IPC codes have that patent count."""
    return Counter(counts.values())


def _write_ipc_freq_of_freq_csv(path: Path, freq_of_freq: Counter[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["n_patents_per_code", "n_ipc_codes"]
        )
        writer.writeheader()
        for n_patents, n_codes in sorted(freq_of_freq.items()):
            writer.writerow(
                {"n_patents_per_code": n_patents, "n_ipc_codes": n_codes}
            )


def _write_ipc_tail_summary_csv(
    path: Path, counts: Counter[str], freq_of_freq: Counter[int]
) -> None:
    n_codes = len(counts)
    n_singleton = freq_of_freq.get(1, 0)
    n_le5 = sum(n for k, n in freq_of_freq.items() if k <= 5)
    total_assignments = sum(counts.values())
    singleton_assignments = n_singleton  # each singleton code → 1 patent assignment
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        rows = [
            ("n_ipc_codes", n_codes),
            ("n_singleton_codes", n_singleton),
            (
                "pct_singleton_codes",
                round(100.0 * n_singleton / n_codes, 2) if n_codes else 0,
            ),
            ("n_codes_with_le_5_patents", n_le5),
            (
                "pct_codes_with_le_5_patents",
                round(100.0 * n_le5 / n_codes, 2) if n_codes else 0,
            ),
            ("n_patent_ipc_assignments", total_assignments),
            ("n_singleton_assignments", singleton_assignments),
            (
                "pct_assignments_from_singletons",
                round(100.0 * singleton_assignments / total_assignments, 2)
                if total_assignments
                else 0,
            ),
        ]
        for metric, value in rows:
            writer.writerow({"metric": metric, "value": value})


def collect_metrics(input_dir: Path, tokenizer: Any) -> dict[str, Any]:
    claim_texts_all: list[str] = []
    # Parallel list: patent index into claims_per_patent / vs-num accumulators
    claim_patent_indices: list[int] = []
    abstract_texts: list[str] = []
    abstract_n_claims: list[int] = []

    claims_per_patent: list[int] = []
    ipc_counts: Counter[str] = Counter()
    n_patents = 0
    n_with_primary_claims = 0
    n_with_primary_abstract = 0

    for data in iter_shard_records(input_dir, include_open_jsonl=False):
        patent_i = n_patents
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
            claim_texts_all.extend(claim_texts)
            claim_patent_indices.extend([patent_i] * n_claims)

        abstract = primary.get("abstract") if isinstance(primary.get("abstract"), str) else ""
        abstract = abstract.strip()
        if abstract:
            n_with_primary_abstract += 1
            abstract_texts.append(abstract)
            abstract_n_claims.append(n_claims)

    claim_token_lengths = _token_lengths(tokenizer, claim_texts_all)
    abstract_token_lengths = _token_lengths(tokenizer, abstract_texts)

    # Mean claim tokens per patent (for 2D plot)
    tokens_by_patent: dict[int, list[int]] = {}
    for patent_i, n_tok in zip(claim_patent_indices, claim_token_lengths):
        tokens_by_patent.setdefault(patent_i, []).append(n_tok)
    claim_tokens_vs_num: list[tuple[int, float]] = [
        (claims_per_patent[i], statistics.fmean(toks))
        for i, toks in tokens_by_patent.items()
    ]
    abstract_tokens_vs_num: list[tuple[int, int]] = list(
        zip(abstract_n_claims, abstract_token_lengths)
    )

    return {
        "n_patents": n_patents,
        "n_with_primary_claims": n_with_primary_claims,
        "n_with_primary_abstract": n_with_primary_abstract,
        "claim_token_lengths": claim_token_lengths,
        "abstract_token_lengths": abstract_token_lengths,
        "claims_per_patent": claims_per_patent,
        "claim_tokens_vs_num": claim_tokens_vs_num,
        "abstract_tokens_vs_num": abstract_tokens_vs_num,
        "ipc_counts": ipc_counts,
        "tokenizer_name": getattr(tokenizer, "name_or_path", DEFAULT_TOKENIZER),
    }


def write_tables(metrics: dict[str, Any], tables_dir: Path) -> list[Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    path = tables_dir / "tokens_per_claim.csv"
    _write_summary_csv(
        path, "tokens_per_claim", _stats(metrics["claim_token_lengths"])
    )
    written.append(path)

    path = tables_dir / "tokens_per_abstract.csv"
    _write_summary_csv(
        path, "tokens_per_abstract", _stats(metrics["abstract_token_lengths"])
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

    freq_of_freq = _ipc_freq_of_freq(metrics["ipc_counts"])
    path = tables_dir / "ipc_code_frequency_of_frequencies.csv"
    _write_ipc_freq_of_freq_csv(path, freq_of_freq)
    written.append(path)

    path = tables_dir / "ipc_code_tail_summary.csv"
    _write_ipc_tail_summary_csv(path, metrics["ipc_counts"], freq_of_freq)
    written.append(path)

    return written


def _percentile_cap(values: list[float | int], q: float = 99.0) -> float | None:
    """Upper display bound at percentile ``q``; None if empty."""
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    # quantiles(..., n=100) → cut points at 1%, 2%, …, 99%.
    cuts = statistics.quantiles(values, n=100, method="inclusive")
    idx = max(0, min(len(cuts) - 1, int(q) - 1))
    return float(cuts[idx])


def _clip_for_display(
    values: list[float | int], *, q: float = 99.0
) -> tuple[list[float | int], float | None, int]:
    """Keep values ≤ Pq for plotting; return (clipped, cap, n_above)."""
    cap = _percentile_cap(values, q=q)
    if cap is None:
        return [], None, 0
    clipped = [v for v in values if v <= cap]
    n_above = len(values) - len(clipped)
    return clipped, cap, n_above


def _clip_pairs_for_display(
    pairs: list[tuple[float | int, float | int]],
    *,
    q: float = 99.0,
) -> tuple[list[tuple[float | int, float | int]], str]:
    """Clip both axes of 2D pairs to their respective Pq bounds."""
    if not pairs:
        return [], ""
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    x_cap = _percentile_cap(xs, q=q)
    y_cap = _percentile_cap(ys, q=q)
    assert x_cap is not None and y_cap is not None
    clipped = [(x, y) for x, y in pairs if x <= x_cap and y <= y_cap]
    n_out = len(pairs) - len(clipped)
    note = (
        f" (axes ≤ P{q:g}: x≤{x_cap:g}, y≤{y_cap:g}"
        + (f"; {n_out} outside" if n_out else "")
        + ")"
    )
    return clipped, note


def _maybe_mark_bert_limit(ax: Any, *, xmax: float | None) -> None:
    if xmax is not None and xmax >= BERT_MAX_POSITIONS:
        ax.axvline(
            BERT_MAX_POSITIONS,
            color="#9b2226",
            linestyle="--",
            linewidth=1.2,
            label=f"BERT max {BERT_MAX_POSITIONS}",
        )
        ax.legend(loc="upper right", fontsize=8)


def write_plots(metrics: dict[str, Any], plots_dir: Path) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    q = 99.0
    tok_name = metrics.get("tokenizer_name", DEFAULT_TOKENIZER)

    # 1) Claim token length histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    lengths, cap, n_above = _clip_for_display(metrics["claim_token_lengths"], q=q)
    if lengths:
        ax.hist(
            lengths,
            bins=min(40, max(10, int(len(lengths) ** 0.5))),
            color="#3d5a80",
            edgecolor="white",
        )
    ax.set_xlabel(f"BERT tokens per claim ({tok_name})")
    ax.set_ylabel("Count")
    title = "Claim token length"
    if cap is not None:
        title += f" (≤ P{q:g}={cap:g}"
        if n_above:
            title += f"; {n_above} above omitted"
        title += ")"
    ax.set_title(title)
    _maybe_mark_bert_limit(ax, xmax=cap)
    fig.tight_layout()
    path = plots_dir / "hist_claim_token_length.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 2) Num claims per patent histogram (exclude 0: dominates scale; P99-cap the rest)
    fig, ax = plt.subplots(figsize=(8, 5))
    n_claims_raw = [n for n in metrics["claims_per_patent"] if n > 0]
    n_zero = len(metrics["claims_per_patent"]) - len(n_claims_raw)
    n_claims, cap, n_above = _clip_for_display(n_claims_raw, q=q)
    if n_claims and cap is not None:
        bin_max = max(1, int(cap))
        bins = range(1, bin_max + 2)
        ax.hist(n_claims, bins=bins, color="#ee6c4d", edgecolor="white", align="left")
        ax.set_xlim(0.5, bin_max + 0.5)
    ax.set_xlabel("Number of claims (primary published document)")
    ax.set_ylabel("Number of patents")
    title = "Claims per patent"
    bits: list[str] = []
    if n_zero:
        bits.append(f"{n_zero} with 0 claims omitted")
    if cap is not None:
        bits.append(f"≤ P{q:g}={cap:g}")
        if n_above:
            bits.append(f"{n_above} above omitted")
    if bits:
        title += " (" + "; ".join(bits) + ")"
    ax.set_title(title)
    fig.tight_layout()
    path = plots_dir / "hist_num_claims_per_patent.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 3) 2D: mean claim tokens vs num claims
    fig, ax = plt.subplots(figsize=(8, 5))
    pairs, note = _clip_pairs_for_display(metrics["claim_tokens_vs_num"], q=q)
    if pairs:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        hb = ax.hexbin(xs, ys, gridsize=20, cmap="viridis", mincnt=1)
        fig.colorbar(hb, ax=ax, label="Patents")
    ax.set_xlabel("Number of claims")
    ax.set_ylabel(f"Mean BERT tokens per claim ({tok_name})")
    ax.set_title(f"Claim tokens vs number of claims (per patent){note}")
    fig.tight_layout()
    path = plots_dir / "hist2d_claim_tokens_vs_num_claims.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 4) Abstract token length histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    abs_lengths, cap, n_above = _clip_for_display(
        metrics["abstract_token_lengths"], q=q
    )
    if abs_lengths:
        ax.hist(
            abs_lengths,
            bins=min(40, max(10, int(len(abs_lengths) ** 0.5))),
            color="#3d5a80",
            edgecolor="white",
        )
    ax.set_xlabel(f"BERT tokens per abstract ({tok_name})")
    ax.set_ylabel("Count")
    title = "Abstract token length"
    if cap is not None:
        title += f" (≤ P{q:g}={cap:g}"
        if n_above:
            title += f"; {n_above} above omitted"
        title += ")"
    ax.set_title(title)
    _maybe_mark_bert_limit(ax, xmax=cap)
    fig.tight_layout()
    path = plots_dir / "hist_abstract_token_length.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # 5) 2D: abstract tokens vs num claims
    fig, ax = plt.subplots(figsize=(8, 5))
    abs_pairs, note = _clip_pairs_for_display(metrics["abstract_tokens_vs_num"], q=q)
    if abs_pairs:
        xs = [p[0] for p in abs_pairs]
        ys = [p[1] for p in abs_pairs]
        hb = ax.hexbin(xs, ys, gridsize=20, cmap="viridis", mincnt=1)
        fig.colorbar(hb, ax=ax, label="Patents")
    ax.set_xlabel("Number of claims")
    ax.set_ylabel(f"BERT tokens per abstract ({tok_name})")
    ax.set_title(f"Abstract tokens vs number of claims (per patent){note}")
    fig.tight_layout()
    path = plots_dir / "hist2d_abstract_tokens_vs_num_claims.png"
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

    # 7) Frequency-of-frequencies: long tail (esp. singleton IPC codes)
    fig, ax = plt.subplots(figsize=(8, 5))
    freq_of_freq = _ipc_freq_of_freq(counts)
    if freq_of_freq:
        xs = sorted(freq_of_freq.keys())
        ys = [freq_of_freq[x] for x in xs]
        ax.bar(xs, ys, color="#293241", width=0.8, align="center")
        ax.set_yscale("log")
        n_singleton = freq_of_freq.get(1, 0)
        n_codes = sum(freq_of_freq.values())
        pct = 100.0 * n_singleton / n_codes if n_codes else 0.0
        ax.set_title(
            f"IPC code frequency-of-frequencies "
            f"({n_singleton} singletons, {pct:.1f}% of {n_codes} codes)"
        )
    else:
        ax.set_title("IPC code frequency-of-frequencies (no data)")
    ax.set_xlabel("Patents per IPC code")
    ax.set_ylabel("Number of IPC codes (log scale)")
    fig.tight_layout()
    path = plots_dir / "hist_ipc_code_patent_frequency.png"
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
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=f"Hugging Face tokenizer name (default: {DEFAULT_TOKENIZER})",
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

    print(f"loading tokenizer={args.tokenizer}")
    tokenizer = _load_tokenizer(args.tokenizer)
    metrics = collect_metrics(input_dir, tokenizer)
    print(
        f"patents={metrics['n_patents']} "
        f"with_claims={metrics['n_with_primary_claims']} "
        f"with_abstract={metrics['n_with_primary_abstract']} "
        f"ipc_labels={len(metrics['ipc_counts'])} "
        f"tokenizer={metrics['tokenizer_name']}"
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
