"""Sample application_numbers with a per-primary_ipc cap (IPC reasoning universe)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ipc_lookup import IPCLookup  # noqa: E402
from patents import iter_patent_texts  # noqa: E402


def _normalize_ipc(code: str) -> str:
    return code.strip().upper().replace(" ", "")


def _section(code: str) -> str:
    return code[:1] if code else "?"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Sample eligible patents for ipc_reasoning with a max fraction "
            "per primary_ipc symbol (default 1% of --target). Requires a WIPO "
            "definition_statement (same rule as IPCReasoningTask.eligible)."
        )
    )
    p.add_argument(
        "--patents-dir",
        type=Path,
        default=REPO_ROOT / "data" / "derived" / "patent_search_clean",
    )
    p.add_argument(
        "--ipc-jsonl",
        type=Path,
        default=REPO_ROOT / "data" / "ipc-codes" / "ipc_codes_20260101.jsonl",
    )
    p.add_argument(
        "--target",
        type=int,
        default=10_000,
        help="Desired number of application_numbers (default 10000)",
    )
    p.add_argument(
        "--max-per-symbol-frac",
        type=float,
        default=0.005,
        help="Max share of --target from any single primary_ipc (default 0.005 = 0.5%)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "derived"
        / "instruction_generation"
        / "_samples",
        help="Directory for ids txt + manifest (default _samples/)",
    )
    p.add_argument(
        "--name",
        default="ipc_reasoning_10k_cap0_5pct",
        help="Basename for output files",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.target < 1:
        print("error: --target must be >= 1", file=sys.stderr)
        return 1
    if not (0 < args.max_per_symbol_frac <= 1):
        print("error: --max-per-symbol-frac must be in (0, 1]", file=sys.stderr)
        return 1

    patents_dir = (
        args.patents_dir
        if args.patents_dir.is_absolute()
        else REPO_ROOT / args.patents_dir
    )
    ipc_jsonl = (
        args.ipc_jsonl if args.ipc_jsonl.is_absolute() else REPO_ROOT / args.ipc_jsonl
    )
    if not patents_dir.is_dir():
        print(f"error: patents dir missing: {patents_dir}", file=sys.stderr)
        return 1
    if not ipc_jsonl.is_file():
        print(f"error: IPC catalog missing: {ipc_jsonl}", file=sys.stderr)
        return 1

    lookup = IPCLookup.from_jsonl(ipc_jsonl)
    cap = max(1, int(args.target * args.max_per_symbol_frac))
    print(
        f"Scanning {patents_dir} (WIPO IPC entries={len(lookup)}; "
        f"target={args.target}; per-symbol cap={cap} "
        f"= {args.max_per_symbol_frac:.2%} of target)…",
        flush=True,
    )

    eligible: list[tuple[str, str]] = []
    n_no_ipc = 0
    n_unknown_ipc = 0
    n_no_definition = 0
    for patent in iter_patent_texts(patents_dir):
        code = _normalize_ipc(patent.primary_ipc)
        if not code:
            n_no_ipc += 1
            continue
        entry = lookup.get(code)
        if entry is None:
            n_unknown_ipc += 1
            continue
        # Match IPCReasoningTask.eligible: need WIPO definition_statement.
        if not entry.definition_statement:
            n_no_definition += 1
            continue
        eligible.append((patent.application_number, code))

    print(
        f"Eligible (claims+abstract+known primary_ipc+definition): {len(eligible)} "
        f"(skipped no_ipc={n_no_ipc} unknown_ipc={n_unknown_ipc} "
        f"no_definition={n_no_definition})",
        flush=True,
    )

    rng = random.Random(args.seed)
    rng.shuffle(eligible)

    selected: list[tuple[str, str]] = []
    per_symbol: Counter[str] = Counter()
    n_rejected_cap = 0
    for app, code in eligible:
        if len(selected) >= args.target:
            break
        if per_symbol[code] >= cap:
            n_rejected_cap += 1
            continue
        selected.append((app, code))
        per_symbol[code] += 1

    section_counts: Counter[str] = Counter()
    for _, code in selected:
        section_counts[_section(code)] += 1

    out_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else REPO_ROOT / args.output_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_path = out_dir / f"{args.name}.txt"
    man_path = out_dir / f"{args.name}.manifest.json"

    ids = [app for app, _ in selected]
    ids_path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    top_symbols = per_symbol.most_common(20)
    manifest: dict[str, Any] = {
        "name": args.name,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": args.seed,
        "target": args.target,
        "n_selected": len(selected),
        "reached_target": len(selected) >= args.target,
        "max_per_symbol_frac": args.max_per_symbol_frac,
        "max_per_symbol": cap,
        "n_eligible": len(eligible),
        "n_skipped_no_ipc": n_no_ipc,
        "n_skipped_unknown_ipc": n_unknown_ipc,
        "n_skipped_no_definition": n_no_definition,
        "n_walk_rejected_at_cap": n_rejected_cap,
        "n_unique_symbols": len(per_symbol),
        "section_counts": dict(sorted(section_counts.items())),
        "top_symbols": [{"ipc": c, "n": n} for c, n in top_symbols],
        "patents_dir": str(patents_dir),
        "ipc_jsonl": str(ipc_jsonl),
        "ids_file": str(ids_path),
    }
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"Selected {len(selected)}/{args.target} apps, "
        f"{len(per_symbol)} symbols, sections={dict(section_counts)}",
        flush=True,
    )
    print(f"Wrote {ids_path}", flush=True)
    print(f"Wrote {man_path}", flush=True)
    if len(selected) < args.target:
        print(
            "warning: could not reach --target under the per-symbol cap; "
            "lower the cap frac or target, or expand the patent dump.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
