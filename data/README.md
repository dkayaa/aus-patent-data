# data

Dataset artifacts and seed dumps (not pipeline source code).

## Subdirs

| Subdir | Meaning |
|--------|---------|
| `raw/` | Base IP Rapid dumps and raw API payloads from scrape |
| `interim/` | Joined / cleaned / partially labeled tables |
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

## Interim (scrape outputs)

| Path | Producer | Role |
|------|----------|------|
| Configured by `scrape/config/patent_search.yaml` → `paths.output_dir` (default: `data/interim/patent_search/`) | `scrape/src/patent_search.py` | Raw Patent Search API responses as `part-*.jsonl.gz` + `fetched_ids.txt` |
| `interim/patent_search_clean/` (or `scrape/config/clean_patent_search.yaml` → `paths.output_dir`) | `scrape/src/clean_patent_search.py` | Flattened records with parsed claims (`part-*.jsonl.gz`) |

### `patent_search` / `patent_search_clean` storage

Batched JSONL.GZ shards (`part-NNNNN.jsonl.gz`), one compact JSON object per line. Fetch also maintains `fetched_ids.txt` for resume. Open (uncompressed) `part-*.jsonl` may exist while a fetch shard is filling.

### `patent_search_clean` schema

One JSONL line per application:

- `application_number`, `fetched_at`, `ipRightStatusCode`, `inventionTitle`
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
| `tables/chars_per_claim.csv` | min / max / mean claim character length |
| `tables/chars_per_abstract.csv` | min / max / mean abstract character length |
| `tables/num_claims_per_patent.csv` | min / max / mean claims per patent |
| `tables/ipc_label_patent_counts.csv` | IPC code → patent count |
| `plots/hist_claim_char_length.png` | Claim length histogram |
| `plots/hist_num_claims_per_patent.png` | Claims-per-patent histogram |
| `plots/hist2d_claim_chars_vs_num_claims.png` | Mean claim chars vs num claims |
| `plots/ipc_label_patent_counts.png` | Top IPC labels by patent count |
