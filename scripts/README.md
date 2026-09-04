# scripts

Optional one-shot and end-to-end runners that glue stages together.

Prefer thin wrappers that call into `scrape/`, `classification/`, `instruction-generation/`, `dataset-validation/`, and `evaluation/` modules rather than duplicating logic here.

| Script | Calls | Purpose |
|--------|-------|---------|
| `fetch_patent_search.py` | `scrape/src/patent_search.py` | Enrich application numbers via Patent Search API → `part-*.jsonl.gz` + `fetched_ids.txt` |
| `clean_patent_search.py` | `scrape/src/clean_patent_search.py` | Reshape raw shards → `patent_search_clean/part-*.jsonl.gz` (parsed claims) |
| `pack_patent_search_json_to_jsonl.py` | `scrape/src/pack_patent_search_json.py` | One-shot: legacy `{application_number}.json` → `part-*.jsonl.gz` |
| `summarize_patent_search.py` | (standalone; uses `jsonl_gz`) | Summarize field coverage in a patent_search shard folder |
| `analyze_patent_search_clean.py` | (standalone; uses `jsonl_gz`) | Stats + plots from clean shards → `data/tables/`, `data/plots/` |
| `export_patent_text_csvs.py` | (standalone; uses `jsonl_gz`) | Clean shards → `abstracts.csv` + `first_claims.csv` + `claims.csv` under `data/derived/patent_search_text/` |
| `download_patentbert.py` | `classification/src/download_patentbert.py` | Fetch PatentBERT checkpoint → `classification/models/patentbert/` |
| `run_patentbert.py` | `classification/src/run_patentbert.py` | Claim-level CPC-subclass inference → `data/derived/patentbert/predictions.csv` |
| `split_eval_data.py` | `evaluation/src/split.py` | Freeze temporal train/val/test + 3-shot exemplars → `data/derived/evaluation/splits/` |
| `run_baselines.py` | `evaluation/src/run_baseline.py` | OpenRouter zero-shot / 3-shot on frozen test IDs |
| `score_baselines.py` | `evaluation/src/score.py` | Automatic IPC / abstract scores → `data/derived/evaluation/scores/` |
| `prepare_sft_data.py` | `sft/src/prepare.py` | Flat SFT JSONL from frozen eval splits → `data/derived/sft/` |
| `run_sft.py` | `sft/src/run_train.py` | Per-dataset QLoRA (CUDA) → `data/derived/sft/runs/` |
| `sample_ipc_apps.py` | `instruction-generation/src/sample_ipc_apps.py` | Stratified app list (per-`primary_ipc` cap) for `--only-ids` |

### PatentBERT (requires TF1 env — not root `.venv`)

On Apple Silicon / Python 3.13, `tensorflow==1.15.5` is unavailable via pip. Use the conda osx-64 recipe in `classification/README.md`.

```bash
# after activating patentbert-tf1 (see classification/README.md)
pip install -r classification/requirements-patentbert.txt
python scripts/download_patentbert.py
python scripts/run_patentbert.py --max-predictions 5
python scripts/run_patentbert.py --gzip --chunk-size 1000   # stream TF1 in chunks
python scripts/run_patentbert.py --gzip
```

### Export text CSVs for classification

```bash
# First N applications only
python scripts/export_patent_text_csvs.py --max-records 1000

# Full corpus
python scripts/export_patent_text_csvs.py

# Gzip-compressed CSVs (*.csv.gz)
python scripts/export_patent_text_csvs.py --gzip
```

Writes from the primary published document:

- `abstracts.csv` / `.csv.gz` — 1 row/app
- `first_claims.csv` / `.csv.gz` — 1 row/app (first claim only)
- `claims.csv` / `.csv.gz` — 1 row/claim

### Pack legacy per-file JSON

```bash
python scripts/pack_patent_search_json_to_jsonl.py \
  --input-dir data/derived/patent_search
# Replace existing shards if re-packing:
python scripts/pack_patent_search_json_to_jsonl.py \
  --input-dir /Volumes/T7/patent-aus/data/derived/patent_search --force
```
