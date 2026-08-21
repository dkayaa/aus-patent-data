# data

Dataset artifacts and seed dumps (not pipeline source code).

## Subdirs

| Subdir | Meaning |
|--------|---------|
| `raw/` | Base IP Rapid dumps and raw API payloads from scrape |
| `derived/` | Joined / cleaned / partially labeled tables |
| `processed/` | Final labeled dataset for analysis / release |
| `tables/` | Analysis CSV exports (e.g. from `scripts/analyze_patent_search_clean.py`) |
| `plots/` | Analysis plot PNGs |

## Base seed (checked in)

| File | Source | Role |
|------|--------|------|
| `raw/application-toy.csv` | Toy subset of the IP Rapid application dump (data.gov) | Seed application list for development and scraper prototyping |

### `application-toy.csv` columns

Metadata / lifecycle fields only (no patent body text):

- `ip_right_type`, `application_number`, `ip_right_sub_type`, `status`
- `application_date`, `earliest_filed_date`, `priority_date`
- `gained_registration_status_date`, `gained_enforceable_status_date`, `enforceable_from_date`, `deemed_retired_date`

Scrapers use `application_number` (and related IDs) to pull semantic text from the IP Australia API. Full IP Rapid dumps under `raw/` (and other bulk artifacts) are stored with **Git LFS** — see root `.gitattributes`.

## Rules

- Dataset binaries/tables under `data/` go through Git LFS (not normal git blobs). Run `git lfs install` once locally before adding or pulling them.
- Prefer regenerating enriched and labeled files from `scrape/` + `classification/` over hand-editing.
- Update this README when schema or row counts change.

## Derived (scrape / pipeline outputs)

| Path | Producer | Role |
|------|----------|------|
| Configured by `scrape/config/patent_search.yaml` → `paths.output_dir` (default: `data/derived/patent_search/`) | `scrape/src/patent_search.py` | Raw Patent Search API responses as `part-*.jsonl.gz` + `fetched_ids.txt` |
| `derived/patent_search_clean/` (or `scrape/config/clean_patent_search.yaml` → `paths.output_dir`) | `scrape/src/clean_patent_search.py` | Flattened records with parsed claims (`part-*.jsonl.gz`) |
| `derived/patent_search_text/` | `scripts/export_patent_text_csvs.py` | Classification-oriented CSVs: `abstracts.csv`, `first_claims.csv`, `claims.csv` |
| `derived/patentbert/` | `scripts/run_patentbert.py` | Claim-level PatentBERT CPC-subclass predictions (`input.tsv[.gz]`, `row_map.csv[.gz]`, `predict_result.txt[.gz]`, `predictions.csv[.gz]`) |
| `derived/instruction_generation/_pools/` | `scripts/generate_instruction_data.py` | Shared Evol-Instruct instruction phrasings (not per-model) |
| `derived/instruction_generation/{model_slug}/` | `scripts/generate_instruction_data.py` | Per-generator SFT JSONL: `manifest.json` + `<task>/part-*.jsonl.gz` + `done_ids.txt` |
| `derived/instruction_generation_validation/{model_slug}/` | `scripts/validate_instruction_data.py`, `scripts/judge_instruction_data.py` | Mode 1 `passed/`/`rejected/` + Mode 2 `llm_judge/` per task |
| `derived/evaluation/splits/{model_slug}/` | `scripts/split_eval_data.py` | Frozen train/val/test JSONL + `split_manifest.json` + `exemplars.json` |
| `derived/evaluation/predictions/{system_slug}/` | `scripts/run_baselines.py` | OpenRouter zero-shot / 3-shot predictions per task |
| `derived/evaluation/scores/` | `scripts/score_baselines.py` | Per-system `report.json` + `summary.json` |

### `patent_search` / `patent_search_clean` storage

Batched JSONL.GZ shards (`part-NNNNN.jsonl.gz`), one compact JSON object per line. Fetch also maintains `fetched_ids.txt` for resume. Open (uncompressed) `part-*.jsonl` may exist while a fetch shard is filling.

### `patent_search_clean` schema

One JSONL line per application:

- `application_number`, `fetched_at`, `ipRightStatusCode`, `inventionTitle`
- `primary_ipc` (classification with lowest `sequenceNumber`; else first listed)
- `ipcrClassification` (list of classification strings)
- `patentApplicationType`, `filedDate`, `priorityDate`, `expiryDate`
- `applicant`, `inventors` (name lists)
- `publishedDocuments[]`: `documentTypeCode`, `fileName`, `abstract`, `claims` (list of numbered claim strings), `claims_parse_ok`
- `summary.json` in the same folder: run counts + full `claims_parse_failures` list for triage

## Schema (enriched / final)

TBD once labels are fixed.

## Regeneration

```bash
export IP_AUSTRALIA_CLIENT_ID='...'
export IP_AUSTRALIA_CLIENT_SECRET='...'
python scripts/fetch_patent_search.py
python scripts/clean_patent_search.py
python scripts/analyze_patent_search_clean.py
```

See `scrape/README.md` for config and idempotency notes.

## Analysis outputs

From `patent_search_clean` (primary published document per patent: prefer B* over A*):

| Path | Contents |
|------|----------|
| `tables/tokens_per_claim.csv` | min / max / mean claim length in BERT WordPiece tokens |
| `tables/tokens_per_abstract.csv` | min / max / mean abstract length in BERT WordPiece tokens |
| `tables/num_claims_per_patent.csv` | min / max / mean claims per patent |
| `tables/ipc_label_patent_counts.csv` | IPC code → patent count |
| `tables/ipc_code_frequency_of_frequencies.csv` | patents-per-code → number of IPC codes |
| `tables/ipc_code_tail_summary.csv` | singleton / long-tail summary stats |
| `plots/hist_claim_token_length.png` | Claim token-length histogram (`bert-base-uncased`) |
| `plots/hist_num_claims_per_patent.png` | Claims-per-patent histogram |
| `plots/hist2d_claim_tokens_vs_num_claims.png` | Mean claim tokens vs num claims |
| `plots/hist_abstract_token_length.png` | Abstract token-length histogram |
| `plots/hist2d_abstract_tokens_vs_num_claims.png` | Abstract tokens vs num claims |
| `plots/ipc_label_patent_counts.png` | Top IPC labels by patent count |
| `plots/hist_ipc_code_patent_frequency.png` | Frequency-of-frequencies (long-tail / singletons) |

Length plots use BERT WordPiece token counts (default tokenizer `bert-base-uncased`, excluding special tokens). Override with `--tokenizer`.

### AI / IPC subset

Filter to patents with ≥1 matching IPC prefix and write under separate dirs (does not overwrite the full-corpus outputs):

```bash
.venv/bin/python scripts/analyze_patent_search_clean.py --ipc-prefix G06N
# → data/tables/ipc_G06N/, data/plots/ipc_G06N/

.venv/bin/python scripts/analyze_patent_search_clean.py --ipc-prefix G06N --ipc-prefix G06V
# → data/tables/ipc_G06N+G06V/, data/plots/ipc_G06N+G06V/
```
