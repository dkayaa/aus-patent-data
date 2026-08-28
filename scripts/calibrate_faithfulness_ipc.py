#!/usr/bin/env python3
"""Calibrate MiniCheck on a human-labeled good/bad ipc justification pool.

Scores each pool sentence against claims, definition, and combined documents
(diagnostic three-way). Production path (see faithfulness.py):

  atomicize → drop META before MiniCheck → combined doc → bands
  SUPPORTED P≥0.7 / UNDECIDED mid / UNSUPPORTED P<0.3

Does not change Mode 1 pass/fail.

  .venv/bin/python scripts/calibrate_faithfulness_ipc.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "dataset-validation" / "src"))

_NLTK = REPO_ROOT / ".venv" / "nltk_data"
if _NLTK.is_dir():
    import os

    os.environ.setdefault("NLTK_DATA", str(_NLTK))

from faithfulness import FaithfulnessScorer, combined_document  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_POOL = (
    REPO_ROOT / "dataset-validation" / "config" / "faithfulness_calibration.jsonl"
)
DEFAULT_OUT = (
    REPO_ROOT
    / "data"
    / "derived"
    / "instruction_generation_validation"
    / "ipc_reasoning"
    / "faithfulness"
    / "calibration_results.jsonl"
)


def load_pool(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("label") not in ("good", "bad"):
                raise SystemExit(f"{path}:{line_no}: label must be good|bad")
            for key in ("claims", "definition", "sentence", "id"):
                if not str(row.get(key) or "").strip():
                    raise SystemExit(f"{path}:{line_no}: missing {key}")
            rows.append(row)
    return rows


def score_row(scorer: FaithfulnessScorer, row: dict[str, Any]) -> dict[str, Any]:
    claims = row["claims"].strip()
    definition = row["definition"].strip()
    sentence = row["sentence"].strip()
    docs = {
        "claims": claims,
        "definition": definition,
        "combined": combined_document(claims=claims, definition=definition),
    }
    out: dict[str, Any] = {
        "id": row["id"],
        "label": row["label"],
        "difficulty": row.get("difficulty"),
        "failure_mode": row.get("failure_mode"),
        "application_number": row.get("application_number"),
        "ipc": row.get("ipc"),
        "sentence": sentence,
        "why": row.get("why"),
    }
    for name, doc in docs.items():
        labels, probs = scorer._score_pairs([doc], [sentence])
        out[f"support_{name}"] = labels[0]
        out[f"prob_{name}"] = probs[0]
    return out


def _gap_table(
    lines: list[str],
    goods: list[dict[str, Any]],
    bads: list[dict[str, Any]],
    title: str,
) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append(
        f"good={len(goods)}  bad={len(bads)}"
    )
    if not goods or not bads:
        lines.append("_skipped (need both good and bad)_")
        lines.append("")
        return
    lines.append("")
    lines.append(
        "| doc | good mean P | bad mean P | good support=1 | bad support=1 | gap (good−bad P) |"
    )
    lines.append(
        "|-----|-------------|------------|----------------|---------------|------------------|"
    )
    for doc in ("claims", "definition", "combined"):
        g_p = sum(r[f"prob_{doc}"] for r in goods) / len(goods)
        b_p = sum(r[f"prob_{doc}"] for r in bads) / len(bads)
        g_r = sum(r[f"support_{doc}"] for r in goods) / len(goods)
        b_r = sum(r[f"support_{doc}"] for r in bads) / len(bads)
        lines.append(
            f"| {doc} | {g_p:.3f} | {b_p:.3f} | {100 * g_r:.0f}% | "
            f"{100 * b_r:.0f}% | {g_p - b_p:+.3f} |"
        )
    min_good = min(r["prob_combined"] for r in goods)
    max_bad = max(r["prob_combined"] for r in bads)
    if min_good > max_bad:
        lines.append("")
        lines.append(
            f"**Separates cleanly on P(combined):** min(good)={min_good:.3f} "
            f"> max(bad)={max_bad:.3f}"
        )
    else:
        lines.append("")
        lines.append(
            f"**Overlap on P(combined):** min(good)={min_good:.3f} ≤ "
            f"max(bad)={max_bad:.3f}"
        )
    lines.append("")


def summarize(results: list[dict[str, Any]]) -> str:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_label[r["label"]].append(r)

    goods = by_label["good"]
    bads = by_label["bad"]
    hard_bads = [r for r in bads if r.get("difficulty") == "hard"]
    easy_bads = [r for r in bads if r.get("difficulty") == "easy"]

    lines = ["# Faithfulness calibration", ""]
    lines.append(
        f"n={len(results)}  good={len(goods)}  bad={len(bads)} "
        f"(hard={len(hard_bads)}, easy={len(easy_bads)})"
    )
    lines.append("")
    lines.append(
        "Easy bads are cartoon topic mismatches (floor only). "
        "Hard bads are realistic: invented ops, definition stuffing, "
        "wrong-feature link, neighbouring class, empty meta."
    )
    lines.append("")
    lines.append(
        "Caveat: pool uses trimmed claims/definitions and shorter sentences "
        "than production — passing here does not prove production readiness."
    )
    lines.append("")

    _gap_table(lines, goods, bads, "All negatives (inflated by easy bads)")
    _gap_table(
        lines,
        goods,
        hard_bads,
        "Hard negatives only (the capability test)",
    )
    _gap_table(lines, goods, easy_bads, "Easy negatives only (floor)")

    lines.append("## Per-item (sorted by combined prob)")
    lines.append("")
    lines.append(
        "| id | label | difficulty | failure_mode | claims | def | "
        "combined | P(c) | P(d) | P(x) |"
    )
    lines.append(
        "|----|-------|------------|--------------|--------|-----|"
        "----------|------|------|------|"
    )
    for r in sorted(results, key=lambda x: -x["prob_combined"]):
        fm = r.get("failure_mode") or "—"
        lines.append(
            f"| `{r['id']}` | {r['label']} | {r.get('difficulty') or '—'} | "
            f"{fm} | {r['support_claims']} | {r['support_definition']} | "
            f"{r['support_combined']} | {r['prob_claims']:.3f} | "
            f"{r['prob_definition']:.3f} | {r['prob_combined']:.3f} |"
        )

    ranked = sorted(results, key=lambda x: -x["prob_combined"])
    good_ranks = [i + 1 for i, r in enumerate(ranked) if r["label"] == "good"]
    if good_ranks:
        lines.append("")
        lines.append(
            f"Mean rank of good items by P(combined) (1=best): "
            f"{sum(good_ranks) / len(good_ranks):.1f} / {len(results)}"
        )

    # Spotlight the two wrong-bridge cases
    spotlight = [
        r
        for r in results
        if r["id"]
        in ("bad_pv_right_facts_wrong_link", "bad_eat_neighbouring_code")
    ]
    if spotlight:
        lines.append("")
        lines.append("## Spotlight: true facts, wrong bridge")
        lines.append("")
        for r in spotlight:
            lines.append(
                f"- `{r['id']}`: claims={r['support_claims']}/{r['prob_claims']:.3f}, "
                f"def={r['support_definition']}/{r['prob_definition']:.3f}, "
                f"combined={r['support_combined']}/{r['prob_combined']:.3f}"
            )

    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "- **Floor fail:** easy bads not clearly below goods → stop; tool broken."
    )
    lines.append(
        "- **Capability fail:** hard bads overlap goods on combined → MiniCheck "
        "can't police realistic IPC justifications."
    )
    lines.append(
        "- If combined separates but claims/def halves do not, the four-state "
        "split design was the bug, not the model."
    )
    lines.append(
        "- Soft threshold ~0.5 on P(combined) separated goods from caught "
        "failures in this pool; production now uses bands 0.3/0.7 plus "
        "decide-first META (alignment markers / empty claim terms)."
    )
    lines.append(
        "- `wrong_feature_link` staying high is expected: faithfulness ≠ "
        "correct IPC reasoning — expert audit."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "reports" / "faithfulness_calibration.md",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "ckpts",
    )
    args = ap.parse_args()

    rows = load_pool(args.pool)
    logger.info("Loaded %d calibration items from %s", len(rows), args.pool)

    scorer = FaithfulnessScorer(cache_dir=args.cache_dir)
    results = [score_row(scorer, row) for row in rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %s", args.out)

    report = summarize(results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    logger.info("Wrote %s", args.report)
    print(report)


if __name__ == "__main__":
    main()
