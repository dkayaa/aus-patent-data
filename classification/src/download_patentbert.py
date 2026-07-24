#!/usr/bin/env python3
"""Download PatentBERT CPC-subclass checkpoint + BERT base vocab/config."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "classification" / "models" / "patentbert"
_UA = "Mozilla/5.0 (compatible; aus-patent-data/1.0)"

# Released fine-tuned PatentBERT assets (Google Drive file IDs from PatentBERT.ipynb)
DRIVE_FILES = {
    "model.ckpt-181172.data-00000-of-00001": "1CjV7F4lBjPQnssx6Au4pPXjC_QVVSv7D",
    "model.ckpt-181172.index": "1VH_ByE2txMkNQJXlWWXQ-mAnP3D9IAd-",
    "model.ckpt-181172.meta": "1QGDkoN6MNGkh7xMu7-J8BUCUYo5y_lkO",
    "labels_group_id.tsv": "1oxfu-6gsKehXWOHTCv-uOrlUYfOGIcyI",
}

OPTIONAL_DRIVE_FILES = {
    "data.2015.tsv": "1S2KK7zqKwm3Op_KwHZNoP7xuoa81qJ9e",
}

# Official GCS zip is often 403 now; prefer Hugging Face mirrors for vocab/config.
BERT_BASE_ZIP_URL = (
    "https://storage.googleapis.com/bert_models/2018_10_18/"
    "uncased_L-12_H-768_A-12.zip"
)
HF_VOCAB_URL = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt"
HF_CONFIG_URL = "https://huggingface.co/bert-base-uncased/resolve/main/config.json"
BERT_EXTRACT_NAMES = ("vocab.txt", "bert_config.json")

# Keys expected by patentbert/modeling.BertConfig (ignore HF-only extras).
_BERT_CONFIG_KEYS = (
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "hidden_act",
    "hidden_dropout_prob",
    "attention_probs_dropout_prob",
    "max_position_embeddings",
    "type_vocab_size",
    "initializer_range",
)


def _download_drive(file_id: str, dest_path: Path) -> None:
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is required: pip install gdown")
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {dest_path.name} ...")
    gdown.download(url, str(dest_path), quiet=False)


def _download_url(url: str, dest_path: Path) -> None:
    print(f"Downloading {dest_path.name} from {url} ...")
    req = Request(url, headers={"User-Agent": _UA})
    with urlopen(req) as resp, dest_path.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _write_bert_config_from_hf(hf_config_path: Path, dest: Path) -> None:
    raw = json.loads(hf_config_path.read_text(encoding="utf-8"))
    slim = {k: raw[k] for k in _BERT_CONFIG_KEYS if k in raw}
    dest.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")


def _ensure_bert_vocab_config(model_dir: Path) -> None:
    vocab = model_dir / "vocab.txt"
    config = model_dir / "bert_config.json"
    if vocab.is_file() and config.is_file():
        print("BERT vocab/config already present.")
        return

    # Prefer Hugging Face (GCS bert_models zip commonly returns 403).
    try:
        if not vocab.is_file():
            _download_url(HF_VOCAB_URL, vocab)
        if not config.is_file():
            hf_cfg = model_dir / "hf_config.json"
            _download_url(HF_CONFIG_URL, hf_cfg)
            _write_bert_config_from_hf(hf_cfg, config)
            if hf_cfg.is_file():
                hf_cfg.unlink()
        print("BERT vocab/config ready (Hugging Face bert-base-uncased).")
        return
    except Exception as exc:
        print(f"Hugging Face download failed ({exc}); trying official BERT zip ...")

    zip_path = model_dir / "uncased_L-12_H-768_A-12.zip"
    if not zip_path.is_file() or zip_path.stat().st_size < 1_000_000:
        if zip_path.is_file():
            zip_path.unlink()
        _download_url(BERT_BASE_ZIP_URL, zip_path)

    print("Extracting vocab.txt and bert_config.json ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        for wanted in BERT_EXTRACT_NAMES:
            match = next((n for n in names if n.endswith(wanted)), None)
            if match is None:
                raise RuntimeError(f"Could not find {wanted} in BERT zip")
            with zf.open(match) as src, (model_dir / wanted).open("wb") as dst:
                dst.write(src.read())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory to store checkpoint + labels + vocab/config",
    )
    parser.add_argument(
        "--with-data-2015",
        action="store_true",
        help="Also download data.2015.tsv (large; optional eval set)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist",
    )
    args = parser.parse_args(argv)

    model_dir = args.model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    files = dict(DRIVE_FILES)
    if args.with_data_2015:
        files.update(OPTIONAL_DRIVE_FILES)

    for name, file_id in files.items():
        dest = model_dir / name
        if dest.is_file() and not args.force:
            print(f"Exists, skip: {name}")
            continue
        _download_drive(file_id, dest)

    _ensure_bert_vocab_config(model_dir)
    print(f"Done. Model dir: {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
