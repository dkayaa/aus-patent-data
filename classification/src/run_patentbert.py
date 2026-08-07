#!/usr/bin/env python3
"""Stream claims CSV in chunks, run PatentBERT per chunk, append predictions.

The upstream TF1 classifier loads its TSV entirely into memory. For large corpora
(~500k rows) this script never feeds it more than ``chunk_size`` rows at a time.
"""

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
from typing import Any, Iterator, TextIO

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "classification" / "config" / "patentbert.yaml"
PATENTBERT_DIR = Path(__file__).resolve().parent / "patentbert"
CLASSIFIER = PATENTBERT_DIR / "patent_classifier.py"
CHECKPOINT_PREFIX = "model.ckpt-181172"
DEFAULT_CHUNK_SIZE = 1000
PRED_FIELDS = [
    "application_number",
    "claim_seq",
    "id",
    "predicted_group_ids",
    "predicted_scores",
]
ROW_MAP_FIELDS = ["row_index", "application_number", "claim_seq", "id"]

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


def _gzip_replace(path: Path) -> Path:
    """Gzip ``path`` → ``path.gz`` and remove the uncompressed file."""
    gz_path = Path(str(path) + ".gz")
    with path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    path.unlink()
    return gz_path


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


def iter_claim_rows(
    claims_csv: Path,
    *,
    max_predictions: int | None,
) -> Iterator[dict[str, str]]:
    """Yield claim dicts with application_number, claim_seq, id, text."""
    claim_seq_by_app: dict[str, int] = defaultdict(int)
    n_rows = 0
    with _open_text_read(claims_csv) as inf:
        reader = csv.DictReader(inf)
        required = {"application_number", "claim"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError(
                f"{claims_csv} must have columns application_number, claim; "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            if max_predictions is not None and n_rows >= max_predictions:
                break
            app_no = (row.get("application_number") or "").strip()
            text = (row.get("claim") or "").strip()
            if not app_no or not text:
                continue
            claim_seq_by_app[app_no] += 1
            claim_seq = claim_seq_by_app[app_no]
            yield {
                "application_number": app_no,
                "claim_seq": str(claim_seq),
                "id": f"{app_no}__{claim_seq}",
                "text": _sanitize_tsv_field(text),
            }
            n_rows += 1


def iter_chunks(
    rows: Iterator[dict[str, str]],
    chunk_size: int,
) -> Iterator[list[dict[str, str]]]:
    chunk: list[dict[str, str]] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def write_chunk_tsv(
    chunk: list[dict[str, str]],
    input_tsv: Path,
    *,
    placeholder_group_id: str,
) -> None:
    placeholder = _sanitize_tsv_field(placeholder_group_id)
    with input_tsv.open("w", encoding="utf-8", newline="\n") as tsv_f:
        tsv_f.write("group_ids\tid\tdate\ttext\n")
        for row in chunk:
            tsv_f.write(
                f"{placeholder}\t{row['id']}\t1970-01-01\t{row['text']}\n"
            )


def append_row_map(
    map_f: TextIO,
    chunk: list[dict[str, str]],
    *,
    start_index: int,
    write_header: bool,
) -> None:
    writer = csv.DictWriter(map_f, fieldnames=ROW_MAP_FIELDS)
    if write_header:
        writer.writeheader()
    for i, row in enumerate(chunk):
        writer.writerow(
            {
                "row_index": start_index + i,
                "application_number": row["application_number"],
                "claim_seq": row["claim_seq"],
                "id": row["id"],
            }
        )


def append_predictions(
    pred_f: TextIO,
    chunk: list[dict[str, str]],
    pred_lines: list[str],
    *,
    write_header: bool,
) -> int:
    writer = csv.DictWriter(pred_f, fieldnames=PRED_FIELDS)
    if write_header:
        writer.writeheader()
    n = min(len(chunk), len(pred_lines))
    for i in range(n):
        labels, scores = parse_pred_line(pred_lines[i])
        writer.writerow(
            {
                "application_number": chunk[i]["application_number"],
                "claim_seq": chunk[i]["claim_seq"],
                "id": chunk[i]["id"],
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
    return subprocess.call(cmd, env=env)


def os_environ_with_patentbert_path() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PATENTBERT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def prepare_only_stream(
    claims_csv: Path,
    output_dir: Path,
    *,
    placeholder_group_id: str,
    max_predictions: int | None,
    chunk_size: int,
    use_gzip: bool,
) -> int:
    """Stream-write full input.tsv + row_map (disk OK; TF is not invoked)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    input_tsv = output_dir / "input.tsv"
    row_map_path = output_dir / "row_map.csv"
    placeholder = _sanitize_tsv_field(placeholder_group_id)
    n_rows = 0
    with input_tsv.open("w", encoding="utf-8", newline="\n") as tsv_f, row_map_path.open(
        "w", encoding="utf-8", newline=""
    ) as map_f:
        tsv_f.write("group_ids\tid\tdate\ttext\n")
        map_w = csv.DictWriter(map_f, fieldnames=ROW_MAP_FIELDS)
        map_w.writeheader()
        for chunk_idx, chunk in enumerate(
            iter_chunks(
                iter_claim_rows(claims_csv, max_predictions=max_predictions),
                chunk_size,
            ),
            start=1,
        ):
            for row in chunk:
                tsv_f.write(
                    f"{placeholder}\t{row['id']}\t1970-01-01\t{row['text']}\n"
                )
                map_w.writerow(
                    {
                        "row_index": n_rows,
                        "application_number": row["application_number"],
                        "claim_seq": row["claim_seq"],
                        "id": row["id"],
                    }
                )
                n_rows += 1
            print(f"prepare chunk {chunk_idx}: wrote {len(chunk)} rows (total {n_rows})")
    if n_rows == 0:
        raise RuntimeError(f"No claim rows written from {claims_csv}")
    print(f"Prepared {n_rows} claim rows → {input_tsv} (+ {row_map_path.name})")
    if use_gzip:
        print(f"Compressed → {_gzip_replace(input_tsv)}")
        print(f"Compressed → {_gzip_replace(row_map_path)}")
    return n_rows


def run_streaming_inference(
    claims_csv: Path,
    output_dir: Path,
    model_dir: Path,
    *,
    placeholder_group_id: str,
    max_predictions: int | None,
    chunk_size: int,
    multi_hot_threshold: float,
    predict_batch_size: int,
    do_lower_case: bool,
    use_gzip: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_tsv = output_dir / "input.tsv"
    chunk_pred = output_dir / "predict_result.chunk.txt"
    row_map_path = output_dir / "row_map.csv"
    predictions_path = output_dir / "predictions.csv"
    pred_result_path = output_dir / "predict_result.txt"

    # Fresh outputs each run.
    for p in (row_map_path, predictions_path, pred_result_path, chunk_pred):
        if p.is_file():
            p.unlink()
    for p in (
        Path(str(row_map_path) + ".gz"),
        Path(str(predictions_path) + ".gz"),
        Path(str(pred_result_path) + ".gz"),
        Path(str(input_tsv) + ".gz"),
    ):
        if p.is_file():
            p.unlink()

    total_rows = 0
    total_written = 0
    chunk_idx = 0

    with row_map_path.open("w", encoding="utf-8", newline="") as map_f, predictions_path.open(
        "w", encoding="utf-8", newline=""
    ) as pred_f, pred_result_path.open("w", encoding="utf-8", newline="\n") as raw_f:
        for chunk in iter_chunks(
            iter_claim_rows(claims_csv, max_predictions=max_predictions),
            chunk_size,
        ):
            chunk_idx += 1
            start_index = total_rows
            write_chunk_tsv(chunk, input_tsv, placeholder_group_id=placeholder_group_id)
            append_row_map(
                map_f,
                chunk,
                start_index=start_index,
                write_header=(chunk_idx == 1),
            )
            map_f.flush()

            print(
                f"chunk {chunk_idx}: inferring {len(chunk)} rows "
                f"(rows {start_index + 1}-{start_index + len(chunk)}; "
                f"running total after = {start_index + len(chunk)})"
            )
            rc = run_classifier(
                model_dir=model_dir,
                test_file=input_tsv,
                pred_result_file=chunk_pred,
                multi_hot_threshold=multi_hot_threshold,
                predict_batch_size=predict_batch_size,
                number_of_predictions=-1,
                do_lower_case=do_lower_case,
            )
            if rc != 0:
                print(f"error: classifier failed on chunk {chunk_idx} (exit {rc})", file=sys.stderr)
                return rc

            pred_lines = chunk_pred.read_text(encoding="utf-8").splitlines()
            n = append_predictions(
                pred_f,
                chunk,
                pred_lines,
                write_header=(chunk_idx == 1),
            )
            pred_f.flush()
            for line in pred_lines[:n]:
                raw_f.write(line.rstrip("\n") + "\n")
            raw_f.flush()

            total_rows += len(chunk)
            total_written += n
            print(
                f"chunk {chunk_idx}: wrote {n}/{len(chunk)} predictions "
                f"(completed {total_written} rows)"
            )

    if total_rows == 0:
        raise RuntimeError(f"No claim rows from {claims_csv}")

    if chunk_pred.is_file():
        chunk_pred.unlink()
    if input_tsv.is_file():
        input_tsv.unlink()

    print(f"Done: {total_written} predictions → {predictions_path}")
    if use_gzip:
        print(f"Compressed → {_gzip_replace(predictions_path)}")
        print(f"Compressed → {_gzip_replace(pred_result_path)}")
        print(f"Compressed → {_gzip_replace(row_map_path)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--claims-csv", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--max-predictions",
        type=int,
        default=None,
        help="Cap claim rows (overrides infer.number_of_predictions)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=f"Rows per TF1 invoke (default: config or {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument("--multi-hot-threshold", type=float, default=None)
    parser.add_argument("--predict-batch-size", type=int, default=None)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stream-write full input.tsv + row_map.csv and exit (no TF1)",
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        help="Gzip final outputs (predictions / predict_result / row_map)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config.resolve())
    paths = cfg.get("paths") or {}
    infer = cfg.get("infer") or {}

    claims_csv = _resolve_path(
        args.claims_csv or paths.get("claims_csv") or "data/derived/patent_search_text/claims.csv"
    )
    model_dir = _resolve_path(
        args.model_dir or paths.get("model_dir") or "classification/models/patentbert"
    )
    output_dir = _resolve_path(
        args.output_dir or paths.get("output_dir") or "data/derived/patentbert"
    )
    use_gzip = bool(args.gzip or infer.get("gzip", False))

    max_predictions = args.max_predictions
    if max_predictions is None:
        cfg_cap = infer.get("number_of_predictions")
        max_predictions = int(cfg_cap) if cfg_cap is not None else None

    chunk_size = args.chunk_size
    if chunk_size is None:
        chunk_size = int(infer.get("chunk_size") or DEFAULT_CHUNK_SIZE)
    if chunk_size < 1:
        parser.error("--chunk-size must be >= 1")

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

    if args.prepare_only:
        prepare_only_stream(
            claims_csv,
            output_dir,
            placeholder_group_id=placeholder_group_id,
            max_predictions=max_predictions,
            chunk_size=chunk_size,
            use_gzip=use_gzip,
        )
        return 0

    return run_streaming_inference(
        claims_csv,
        output_dir,
        model_dir,
        placeholder_group_id=placeholder_group_id,
        max_predictions=max_predictions,
        chunk_size=chunk_size,
        multi_hot_threshold=threshold,
        predict_batch_size=batch_size,
        do_lower_case=do_lower_case,
        use_gzip=use_gzip,
    )


if __name__ == "__main__":
    raise SystemExit(main())
