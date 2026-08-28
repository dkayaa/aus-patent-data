#!/usr/bin/env python3
"""Run MiniCheck faithfulness on ipc_reasoning only (additive, non-gating).

Design:
  atomicize → drop META (alignment / empty) *before* scoring → combined doc
  → three-way band (SUPPORTED / UNDECIDED / UNSUPPORTED).

Writes:
  data/derived/instruction_generation_validation/ipc_reasoning/faithfulness/
    summaries.jsonl.gz
    sentences.jsonl.gz
  reports/faithfulness_ipc.md
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dataset-validation" / "src"))
sys.path.insert(0, str(REPO_ROOT / "instruction-generation" / "src"))

_NLTK = REPO_ROOT / ".venv" / "nltk_data"
if _NLTK.is_dir():
    import os

    os.environ.setdefault("NLTK_DATA", str(_NLTK))

from faithfulness import (  # noqa: E402
    DEFAULT_SUPPORT_HIGH,
    DEFAULT_SUPPORT_LOW,
    FaithfulnessScorer,
)
from ipc_checks import parse_ipc_output  # noqa: E402
from ipc_lookup import IPCLookup  # noqa: E402
from io_util import iter_task_records  # noqa: E402
from semantic import SemanticScorer, load_embedding_registry  # noqa: E402
from task_metrics import claims_for_terms, score_record  # noqa: E402
from terms_coverage import TermsCoverageScorer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR = REPO_ROOT / "data" / "derived" / "instruction_generation" / "ipc_reasoning"
OUT_DIR = (
    REPO_ROOT
    / "data"
    / "derived"
    / "instruction_generation_validation"
    / "ipc_reasoning"
    / "faithfulness"
)
REPORT = REPO_ROOT / "reports" / "faithfulness_ipc.md"
IPC_JSONL = REPO_ROOT / "data" / "ipc-codes" / "ipc_codes_20260101.jsonl"
COS_FLOOR = 0.15
ROUGE_FLOOR = 0.02


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - spread) / denom, (centre + spread) / denom)


def _pct(k: int, n: int) -> str:
    return "n/a" if n == 0 else f"{100.0 * k / n:.1f}%"


def _ci_str(k: int, n: int) -> str:
    lo, hi = wilson_ci(k, n)
    return f"[{100 * lo:.1f}%, {100 * hi:.1f}%]"


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    try:
        from scipy.stats import spearmanr

        r, _ = spearmanr(xs, ys)
        return float(r) if r == r else None
    except Exception:  # noqa: BLE001
        return None


def _kendall(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    try:
        from scipy.stats import kendalltau

        r, _ = kendalltau(xs, ys)
        return float(r) if r == r else None
    except Exception:  # noqa: BLE001
        return None


def _write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    p.add_argument("--support-high", type=float, default=DEFAULT_SUPPORT_HIGH)
    p.add_argument("--support-low", type=float, default=DEFAULT_SUPPORT_LOW)
    p.add_argument(
        "--score-halves",
        action="store_true",
        help="Also score claims/def alone (diagnostic; 3x MiniCheck calls)",
    )
    args = p.parse_args()

    input_dir = args.input_dir if args.input_dir.is_absolute() else REPO_ROOT / args.input_dir
    if not input_dir.is_dir():
        print(f"error: missing input {input_dir}", file=sys.stderr)
        return 1

    try:
        import minicheck  # noqa: F401
    except ImportError as exc:
        print(
            "MiniCheck could not be imported on this machine.\n"
            "Install with:\n"
            '  .venv/bin/pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"\n'
            f"ImportError: {exc}",
            file=sys.stderr,
        )
        return 1

    ipc_lookup = IPCLookup.from_jsonl(IPC_JSONL)
    terms = TermsCoverageScorer()
    faith = FaithfulnessScorer(
        cache_dir=REPO_ROOT / "ckpts",
        batch_size=8,
        support_high=float(args.support_high),
        support_low=float(args.support_low),
        score_halves=bool(args.score_halves),
        terms=terms,
    )

    registry = load_embedding_registry()
    semantic = SemanticScorer(registry["nomic"], batch_size=8)
    floors = {
        "semantic_cosine_min": COS_FLOOR,
        "rouge_l_f1_min": ROUGE_FLOOR,
        "mrc_best_span_f1_min": 0.5,
        "ipc_wipo_cosine_min": 0.55,
        "ipc_wipo_rouge_l_f1_min": 0.08,
        "ipc_wipo_rouge_l_f1_max": 0.60,
        "ipc_claims_cosine_min": 0.50,
        "abstract_cosine_min": 0.40,
    }

    records = []
    for rec in iter_task_records(input_dir):
        if not rec.get("task"):
            rec = {**rec, "task": "ipc_reasoning"}
        records.append(rec)
        if args.limit is not None and len(records) >= args.limit:
            break
    logger.info("Scoring faithfulness on %d ipc_reasoning records", len(records))

    summaries: list[dict[str, Any]] = []
    sentence_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    sent_state_counts: Counter[str] = Counter()
    t0 = time.perf_counter()

    for i, rec in enumerate(records):
        app = str(rec.get("application_number") or "")
        meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
        code = str(meta.get("primary_ipc") or "").strip()
        if not code:
            parsed, _ = parse_ipc_output(str(rec.get("output") or ""))
            code = parsed or ""
        entry = ipc_lookup.get(code) if code else None
        definition = (entry.definition_statement if entry else None) or ""
        if not definition:
            logger.warning("skip %s: no WIPO definition for code=%r", app, code)
            continue

        claims = claims_for_terms(rec) or ""
        _, body = parse_ipc_output(str(rec.get("output") or ""))
        if not body:
            logger.warning("skip %s: could not parse justification", app)
            continue

        fr = faith.score_justification(
            application_number=app,
            justification=body,
            claims=claims,
            definition=definition,
        )
        for s in fr.sentences:
            sent_state_counts[s.state] += 1

        validation = score_record(
            rec,
            semantic=semantic,
            floors=floors,
            ipc_lookup=ipc_lookup,
            terms=terms,
            faithfulness=None,
        )
        scores = validation.get("scores") or {}

        summary = {
            "application_number": app,
            "primary_ipc": code,
            "programmatic_passed": bool(validation.get("passed")),
            "failed_rules": list(validation.get("failed_rules") or []),
            "semantic_cosine": scores.get("semantic_cosine"),
            "terms_coverage": scores.get("terms_coverage"),
            **faith.summary_dict(fr),
            "claims": claims,
            "definition": definition,
            "justification": body,
        }
        summaries.append(summary)
        sentence_rows.append(faith.sentence_detail_dict(fr))
        examples.append(summary)

        if (i + 1) % 10 == 0:
            logger.info("  %d/%d …", i + 1, len(records))

    elapsed = time.perf_counter() - t0
    n_scored_rec = len(summaries)
    per = elapsed / n_scored_rec if n_scored_rec else 0.0
    hours_40k = (per * 40000) / 3600.0
    lo, hi = float(args.support_low), float(args.support_high)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl_gz(OUT_DIR / "summaries.jsonl.gz", summaries)
    _write_jsonl_gz(OUT_DIR / "sentences.jsonl.gz", sentence_rows)

    n_atoms = sum(sent_state_counts.values())
    n_scored_atoms = (
        sent_state_counts["SUPPORTED"]
        + sent_state_counts["UNDECIDED"]
        + sent_state_counts["UNSUPPORTED"]
    )
    lines: list[str] = []
    lines.append("# Faithfulness audit — ipc_reasoning\n")
    lines.append(
        "MiniCheck-Flan-T5-Large (`lytang/MiniCheck-Flan-T5-Large`, MIT). "
        "Justifications are **atomicised**. **META** atoms (alignment markers "
        "or zero claim terms) are dropped *before* MiniCheck. Remaining atoms "
        f"are scored on the **combined** claims+definition document with bands "
        f"**SUPPORTED** P≥{hi}, **UNDECIDED** [{lo},{hi}), **UNSUPPORTED** P<{lo}. "
        "**Non-gating**.\n"
    )
    lines.append(
        "Wrong-bridge (true claim facts, wrong classifying feature) is "
        "**out of scope** — expert audit. Undecided atoms are the natural "
        "expert-review band.\n"
    )
    lines.append(
        f"Scored **{n_scored_rec}** records, **{n_atoms}** atoms "
        f"({sent_state_counts['META']} META skipped, **{n_scored_atoms}** MiniChecked). "
        f"Wall-clock **{elapsed:.1f}s** ({per:.2f}s/record); "
        f"extrapolated to 40,000 records ≈ **{hours_40k:.1f} hours**.\n"
    )

    lines.append("## Atom-level state distribution\n")
    lines.append("| state | n | rate (all atoms) | rate (scored only) | Wilson 95% CI (all) |")
    lines.append("|-------|---|------------------|--------------------|---------------------|")
    for state in ("SUPPORTED", "UNDECIDED", "UNSUPPORTED", "META"):
        k = sent_state_counts[state]
        scored_pct = (
            _pct(k, n_scored_atoms)
            if state != "META" and n_scored_atoms
            else "—"
        )
        lines.append(
            f"| {state} | {k} | {_pct(k, n_atoms)} | {scored_pct} | {_ci_str(k, n_atoms)} |"
        )
    lines.append("")

    lines.append("## Record-level presence\n")
    n_full = sum(
        1
        for s in summaries
        if int(s["n_scored"] or 0) > 0
        and int(s["n_supported"] or 0) == int(s["n_scored"] or 0)
        and int(s["n_undecided"] or 0) == 0
        and int(s["n_unsupported"] or 0) == 0
    )
    n_any_unsup = sum(1 for s in summaries if int(s["n_unsupported"] or 0) > 0)
    n_any_und = sum(1 for s in summaries if int(s["n_undecided"] or 0) > 0)
    n_any_meta = sum(1 for s in summaries if int(s["n_meta"] or 0) > 0)
    n_any_sup = sum(1 for s in summaries if int(s["n_supported"] or 0) > 0)

    lines.append("| property | n | rate | Wilson 95% CI |")
    lines.append("|----------|---|------|---------------|")
    lines.append(
        f"| fully supported (scored atoms all SUPPORTED) | {n_full} | "
        f"{_pct(n_full, n_scored_rec)} | {_ci_str(n_full, n_scored_rec)} |"
    )
    lines.append(
        f"| ≥1 SUPPORTED | {n_any_sup} | "
        f"{_pct(n_any_sup, n_scored_rec)} | {_ci_str(n_any_sup, n_scored_rec)} |"
    )
    lines.append(
        f"| ≥1 UNDECIDED | {n_any_und} | "
        f"{_pct(n_any_und, n_scored_rec)} | {_ci_str(n_any_und, n_scored_rec)} |"
    )
    lines.append(
        f"| ≥1 UNSUPPORTED | {n_any_unsup} | "
        f"{_pct(n_any_unsup, n_scored_rec)} | {_ci_str(n_any_unsup, n_scored_rec)} |"
    )
    lines.append(
        f"| ≥1 META | {n_any_meta} | "
        f"{_pct(n_any_meta, n_scored_rec)} | {_ci_str(n_any_meta, n_scored_rec)} |"
    )
    lines.append("")

    rates = [
        float(s["faithfulness_rate"])
        for s in summaries
        if s["faithfulness_rate"] is not None
    ]
    und_rates = [
        float(s["undecided_rate"])
        for s in summaries
        if s["undecided_rate"] is not None
    ]
    mean_rate = statistics.mean(rates) if rates else 0.0
    mean_und = statistics.mean(und_rates) if und_rates else 0.0
    lines.append(
        f"Mean `faithfulness_rate` (n_supported/n_scored): **{mean_rate:.3f}**\n"
    )
    lines.append(
        f"Mean `undecided_rate` (n_undecided/n_scored): **{mean_und:.3f}**\n"
    )

    lines.append("## Cross-tabulation: programmatic PASS with unresolved scored atoms\n")
    pass_unresolved = [
        s
        for s in summaries
        if s["programmatic_passed"]
        and (
            int(s["n_unsupported"] or 0) > 0 or int(s["n_undecided"] or 0) > 0
        )
    ]
    n_prog_pass = sum(1 for s in summaries if s["programmatic_passed"])
    pass_unsup = sum(1 for s in pass_unresolved if int(s["n_unsupported"] or 0) > 0)
    pass_und = sum(1 for s in pass_unresolved if int(s["n_undecided"] or 0) > 0)

    lines.append(f"Programmatic passes: **{n_prog_pass}/{n_scored_rec}**\n")
    lines.append(
        f"Of those, **{len(pass_unresolved)}** "
        f"({_pct(len(pass_unresolved), n_prog_pass)} of passes) have "
        "≥1 UNSUPPORTED or UNDECIDED scored atom.\n"
    )
    lines.append("| among those | n |")
    lines.append("|-------------|---|")
    lines.append(f"| contains UNSUPPORTED | {pass_unsup} |")
    lines.append(f"| contains UNDECIDED | {pass_und} |")
    lines.append("")

    lines.append("## Correlation: faithfulness_rate vs nomic cosine / terms_coverage\n")
    pairs_cos = [
        (float(s["faithfulness_rate"]), float(s["semantic_cosine"]))
        for s in summaries
        if s["faithfulness_rate"] is not None and s["semantic_cosine"] is not None
    ]
    pairs_tc = [
        (float(s["faithfulness_rate"]), float(s["terms_coverage"]))
        for s in summaries
        if s["faithfulness_rate"] is not None and s["terms_coverage"] is not None
    ]
    lines.append("| metric | n | Spearman | Kendall |")
    lines.append("|--------|---|----------|---------|")
    if pairs_cos:
        xs, ys = zip(*pairs_cos)
        sp, kd = _spearman(list(xs), list(ys)), _kendall(list(xs), list(ys))
        lines.append(
            f"| nomic cosine | {len(pairs_cos)} | "
            f"{'n/a' if sp is None else f'{sp:.3f}'} | "
            f"{'n/a' if kd is None else f'{kd:.3f}'} |"
        )
    if pairs_tc:
        xs, ys = zip(*pairs_tc)
        sp, kd = _spearman(list(xs), list(ys)), _kendall(list(xs), list(ys))
        lines.append(
            f"| terms_coverage | {len(pairs_tc)} | "
            f"{'n/a' if sp is None else f'{sp:.3f}'} | "
            f"{'n/a' if kd is None else f'{kd:.3f}'} |"
        )
    lines.append("")

    lines.append("## Fifteen lowest-faithfulness examples (among scored atoms)\n")
    ranked = sorted(
        [s for s in examples if s["faithfulness_rate"] is not None],
        key=lambda s: (
            float(s["faithfulness_rate"]),
            -int(s["n_unsupported"] or 0),
            -int(s["n_undecided"] or 0),
        ),
    )
    by_app = {r["application_number"]: r for r in sentence_rows}
    for rank, s in enumerate(ranked[:15], start=1):
        app = s["application_number"]
        detail = by_app.get(app) or {"sentences": []}
        lines.append(
            f"### {rank}. {app} (faithfulness_rate={s['faithfulness_rate']:.3f}; "
            f"undecided={s['n_undecided']}; unsupported={s['n_unsupported']}; "
            f"meta={s['n_meta']}; "
            f"programmatic={'PASS' if s['programmatic_passed'] else 'FAIL'})\n"
        )
        lines.append(f"**IPC:** `{s['primary_ipc']}`\n")
        lines.append("**Claims:**\n```\n" + (s["claims"] or "") + "\n```\n")
        lines.append("**WIPO definition:**\n```\n" + (s["definition"] or "") + "\n```\n")
        lines.append("**Justification:**\n```\n" + (s["justification"] or "") + "\n```\n")
        lines.append("**Atoms:**\n")
        for j, sent in enumerate(detail.get("sentences") or [], start=1):
            if sent.get("scored"):
                lines.append(
                    f"{j}. **[{sent['state']}]** "
                    f"(combined={sent['support_combined']}/"
                    f"{sent['support_combined_prob']:.3f}, "
                    f"claim_terms={sent['n_claim_terms']})\n"
                )
            else:
                lines.append(
                    f"{j}. **[{sent['state']}:{sent.get('meta_reason')}]** "
                    f"(not scored, claim_terms={sent['n_claim_terms']})\n"
                )
            lines.append(f"   {sent['sentence']}\n")
        lines.append("")

    lines.append("## Runtime\n")
    lines.append(
        f"- Wall-clock: {elapsed:.1f}s for {n_scored_rec} records "
        f"({per:.2f}s/record)\n"
        f"- Extrapolation to 40,000 records: ≈ {hours_40k:.1f} hours\n"
    )
    lines.append("## Limits\n")
    lines.append(
        "- META filter is a style heuristic (markers + claim-term hits), not a "
        "semantic judge.\n"
        "- UNDECIDED is for expert review; do not treat it as failure or success.\n"
        "- Wrong-bridge remains out of scope for MiniCheck.\n"
    )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT}", flush=True)
    print(f"Wrote {OUT_DIR / 'summaries.jsonl.gz'}", flush=True)
    print(f"Wrote {OUT_DIR / 'sentences.jsonl.gz'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
