#!/usr/bin/env python3
"""Prepare claims.csv, run PatentBERT claim-level inference, write predictions.csv[.gz]."""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "classification" / "config" / "patentbert.yaml"
PATENTBERT_DIR = Path(__file__).resolve().parent / "patentbert"
CLASSIFIER = PATENTBERT_DIR / "patent_classifier.py"
CHECKPOINT_PREFIX = "model.ckpt-181172"

# predict_result.txt lines: "LABEL (0.85), LABEL2 (0.42)"
_PRED_TOKEN_RE = re.compile(r"^\s*(.+?)\s+\(([0-9]*\.?[0-9]+)\)\s*$")

# Patent claims can exceed the default 128 KiB csv field limit.
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _first_label(label_file: Path) -> str:
    """First CPC code in labels_group_id.tsv (skips header row ``id\\ttitle``)."""
    with label_file.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i == 0:
                continue  # header
            if not row:
                continue
            label = row[0].strip()
            if label:
                return label
    raise RuntimeError(f"No labels found in {label_file}")


def _sanitize_tsv_field(value: str) -> str:
    """Flatten to a single TSV cell (PatentBERT reads with quotechar=None)."""
    return (
        value.replace("\t", " ")
        .replace("\r", " ")
        .replace("\n", " ")
        .strip()
    )


def _open_text_read(path: Path) -> TextIO:
    if path.suffix == ".gz" or path.name.endswith(".csv.gz") or path.name.endswith(".tsv.gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _open_text_write(path: Path) -> TextIO:
    if path.suffix == ".gz" or str(path).endswith(".csv.gz") or str(path).endswith(".tsv.gz") or str(path).endswith(".txt.gz"):
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def _gzip_replace(path: Path) -> Path:
    """Gzip ``path`` → ``path.gz`` and remove the uncompressed file."""
    gz_path = Path(str(path) + ".gz")
    with path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    path.unlink()
    return gz_path


def prepare_input_tsv(
    claims_csv: Path,
    output_dir: Path,
    *,
    placeholder_group_id: str,
    max_predictions: int | None,
    use_gzip: bool = False,
) -> tuple[Path, Path, int]:
    """Write input.tsv (plain; TF1 needs it) + row_map.csv[.gz]."""
    output_dir.mkdir(parents=True, exist_ok=True)
    input_tsv = output_dir / "input.tsv"
    row_map_path = output_dir / ("row_map.csv.gz" if use_gzip else "row_map.csv")

    claim_seq_by_app: dict[str, int] = defaultdict(int)
    n_rows = 0
    placeholder = _sanitize_tsv_field(placeholder_group_id)

    with _open_text_read(claims_csv) as inf, input_tsv.open(
        "w", encoding="utf-8", newline="\n"
    ) as tsv_f, _open_text_write(row_map_path) as map_f:
        reader = csv.DictReader(inf)
        required = {"application_number", "claim"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError(
                f"{claims_csv} must have columns application_number, claim; "
                f"got {reader.fieldnames}"
            )

        # Manual TSV: patent_classifier uses csv.reader(..., quotechar=None).
        tsv_f.write("group_ids\tid\tdate\ttext\n")

        map_w = csv.DictWriter(
            map_f,
            fieldnames=["row_index", "application_number", "claim_seq", "id"],
        )
        map_w.writeheader()

        for row in reader:
            if max_predictions is not None and n_rows >= max_predictions:
                break
            app_no = (row.get("application_number") or "").strip()
            text = (row.get("claim") or "").strip()
            if not app_no or not text:
                continue

            claim_seq_by_app[app_no] += 1
            claim_seq = claim_seq_by_app[app_no]
            row_id = f"{app_no}__{claim_seq}"
            text_flat = _sanitize_tsv_field(text)

            tsv_f.write(f"{placeholder}\t{row_id}\t1970-01-01\t{text_flat}\n")
            map_w.writerow(
                {
                    "row_index": n_rows,
                    "application_number": app_no,
                    "claim_seq": claim_seq,
                    "id": row_id,
                }
            )
            n_rows += 1

    if n_rows == 0:
        raise RuntimeError(f"No claim rows written from {claims_csv}")

    return input_tsv, row_map_path, n_rows


def parse_pred_line(line: str) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    scores: list[str] = []
    line = line.strip()
    if not line:
        return labels, scores
    for part in line.split(", "):
        m = _PRED_TOKEN_RE.match(part)
        if not m:
            continue
        labels.append(m.group(1).strip())
        scores.append(m.group(2))
    return labels, scores


def write_predictions_csv(
    row_map_path: Path,
    pred_result_path: Path,
    predictions_csv: Path,
) -> int:
    with _open_text_read(row_map_path) as map_f:
        row_map = list(csv.DictReader(map_f))

    pred_lines: list[str] = []
    if pred_result_path.is_file():
        with _open_text_read(pred_result_path) as pred_f:
            pred_lines = pred_f.read().splitlines()

    n = min(len(row_map), len(pred_lines))
    with _open_text_write(predictions_csv) as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=[
                "application_number",
                "claim_seq",
                "id",
                "predicted_group_ids",
                "predicted_scores",
            ],
        )
        writer.writeheader()
        for i in range(n):
            labels, scores = parse_pred_line(pred_lines[i])
            writer.writerow(
                {
                    "application_number": row_map[i]["application_number"],
                    "claim_seq": row_map[i]["claim_seq"],
                    "id": row_map[i]["id"],
                    "predicted_group_ids": ",".join(labels),
                    "predicted_scores": ",".join(scores),
                }
            )
    return n


def _check_model_assets(model_dir: Path) -> None:
    init_ckpt = model_dir / CHECKPOINT_PREFIX
    required = [
        Path(str(init_ckpt) + ".meta"),
        Path(str(init_ckpt) + ".index"),
        Path(str(init_ckpt) + ".data-00000-of-00001"),
        model_dir / "labels_group_id.tsv",
        model_dir / "vocab.txt",
        model_dir / "bert_config.json",
    ]
    missing = [p for p in required if not p.is_file()]
    if missing:
        sys.exit(
            "Missing model files:\n  "
            + "\n  ".join(str(p) for p in missing)
            + f"\n\nRun: python scripts/download_patentbert.py --model-dir {model_dir}"
        )


def run_classifier(
    *,
    model_dir: Path,
    test_file: Path,
    pred_result_file: Path,
    multi_hot_threshold: float,
    predict_batch_size: int,
    number_of_predictions: int,
    do_lower_case: bool,
) -> int:
    _check_model_assets(model_dir)
    if not CLASSIFIER.is_file():
        sys.exit(f"Missing classifier: {CLASSIFIER}")

    init_ckpt = model_dir / CHECKPOINT_PREFIX
    label_file = model_dir / "labels_group_id.tsv"
    vocab_file = model_dir / "vocab.txt"
    config_file = model_dir / "bert_config.json"
    data_dir = test_file.parent
    test_basename = test_file.name

    # Classifier defaults reuse_tf_record=True; always rebuild for current input.tsv.
    stale = model_dir / "predict.tf_record"
    if stale.is_file():
        stale.unlink()

    env = os_environ_with_patentbert_path()
    cmd = [
        sys.executable,
        str(CLASSIFIER),
        f"--multi_hot_threshold={multi_hot_threshold}",
        f"--test_file={test_basename}",
        f"--predict_batch_size={predict_batch_size}",
        f"--vocab_file={vocab_file}",
        f"--bert_config_file={config_file}",
        f"--init_checkpoint={init_ckpt}",
        f"--pred_result_file={pred_result_file}",
        f"--number_of_predictions={number_of_predictions}",
        f"--label_file={label_file}",
        f"--data_dir={data_dir}",
        f"--output_dir={model_dir}",
        f"--do_lower_case={'True' if do_lower_case else 'False'}",
        "--reuse_tf_record=False",
    ]
    print("command:", " ".join(cmd))
    return subprocess.call(cmd, env=env)


def os_environ_with_patentbert_path() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PATENTBERT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to patentbert.yaml",
    )
    parser.add_argument(
        "--claims-csv",
        type=Path,
        default=None,
        help="Override paths.claims_csv",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Override paths.model_dir",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override paths.output_dir",
    )
    parser.add_argument(
        "--max-predictions",
        type=int,
        default=None,
        help="Cap claim rows (overrides infer.number_of_predictions)",
    )
    parser.add_argument(
        "--multi-hot-threshold",
        type=float,
        default=None,
        help="Override infer.multi_hot_threshold",
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=None,
        help="Override infer.predict_batch_size",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write input.tsv + row_map.csv[.gz] and exit (no TF1 inference)",
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        help=(
            "Write compressed outputs: row_map.csv.gz, predictions.csv.gz, "
            "predict_result.txt.gz, input.tsv.gz (input.tsv stays plain until after TF1)"
        ),
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config.resolve())
    paths = cfg.get("paths") or {}
    infer = cfg.get("infer") or {}

    claims_csv = _resolve_path(
        args.claims_csv or paths.get("claims_csv") or "data/interim/patent_search_text/claims.csv"
    )
    model_dir = _resolve_path(
        args.model_dir or paths.get("model_dir") or "classification/models/patentbert"
    )
    output_dir = _resolve_path(
        args.output_dir or paths.get("output_dir") or "data/interim/patentbert"
    )
    use_gzip = bool(args.gzip or infer.get("gzip", False))

    max_predictions = args.max_predictions
    if max_predictions is None:
        cfg_cap = infer.get("number_of_predictions")
        max_predictions = int(cfg_cap) if cfg_cap is not None else None

    threshold = (
        float(args.multi_hot_threshold)
        if args.multi_hot_threshold is not None
        else float(infer.get("multi_hot_threshold", 0.3))
    )
    batch_size = (
        int(args.predict_batch_size)
        if args.predict_batch_size is not None
        else int(infer.get("predict_batch_size", 8))
    )
    do_lower_case = bool(infer.get("do_lower_case", True))

    label_file = model_dir / "labels_group_id.tsv"
    placeholder = infer.get("placeholder_group_id")
    if placeholder:
        placeholder_group_id = str(placeholder).strip()
    else:
        if not label_file.is_file():
            sys.exit(
                f"Missing {label_file}; run: python scripts/download_patentbert.py "
                f"--model-dir {model_dir}"
            )
        placeholder_group_id = _first_label(label_file)

    if not claims_csv.is_file():
        sys.exit(f"Claims CSV not found: {claims_csv}")

    input_tsv, row_map_path, n_rows = prepare_input_tsv(
        claims_csv,
        output_dir,
        placeholder_group_id=placeholder_group_id,
        max_predictions=max_predictions,
        use_gzip=use_gzip,
    )
    print(f"Prepared {n_rows} claim rows → {input_tsv} (+ {row_map_path.name})")

    if args.prepare_only:
        if use_gzip and input_tsv.is_file():
            gz = _gzip_replace(input_tsv)
            print(f"Compressed → {gz}")
        return 0

    # TF1 classifier writes a plain text file only.
    pred_result = output_dir / "predict_result.txt"
    # Input TSV already truncated when max_predictions set; pass -1 to classify all rows.
    rc = run_classifier(
        model_dir=model_dir,
        test_file=input_tsv,
        pred_result_file=pred_result,
        multi_hot_threshold=threshold,
        predict_batch_size=batch_size,
        number_of_predictions=-1,
        do_lower_case=do_lower_case,
    )
    if rc != 0:
        return rc

    predictions_csv = output_dir / (
        "predictions.csv.gz" if use_gzip else "predictions.csv"
    )
    n_written = write_predictions_csv(row_map_path, pred_result, predictions_csv)
    print(f"Wrote {n_written} predictions → {predictions_csv}")

    if use_gzip:
        if pred_result.is_file():
            gz_pred = _gzip_replace(pred_result)
            print(f"Compressed → {gz_pred}")
        if input_tsv.is_file():
            gz_tsv = _gzip_replace(input_tsv)
            print(f"Compressed → {gz_tsv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
