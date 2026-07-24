# aus-patent-data

Research repository for constructing an Australian patent dataset. This repo is the **build recipe** for the dataset that supports the associated research paper.

## Pipeline (actual)

1. **Base identifiers** — start from an IP Rapid / data.gov application dump (toy sample checked in under `data/raw/`).
2. **Scrape** — use those application numbers to fetch **semantic textual data** from the IP Australia API into `data/`.
3. **Classification** — label / categorize the enriched records.
4. **Release artifacts** — final tables under `data/processed/`.

## For Cursor agents

Read this section before changing anything.

### Purpose

Work is organized by **pipeline stage**, not by language or framework:

| Stage | Directory | Responsibility |
|-------|-----------|----------------|
| Base + artifacts | `data/` | Seed dumps (`raw/`), scrape outputs, interim joins, final dataset. |
| Enrich | `scrape/` | Call IP Australia (and related) APIs to fetch semantic text for known application numbers. Does not invent the application list. |
| Label | `classification/` | Apply taxonomies, rules, or models. No API fetching of patent text. |

### Hard rules

1. **Do not mix stages.** API clients and downloaders stay in `scrape/`. Labeling stays in `classification/`.
2. **Scrape is enrichment, not discovery.** The application universe comes from IP Rapid (or a full dump later). Scrapers take `application_number` (and related IDs) from `data/raw/` and fetch text/metadata from IP Australia.
3. **Pipeline direction:** `data/raw` (base) → `scrape/` → enriched artifacts in `data/` → `classification/` → `data/interim/` / `data/processed/`.
4. **Toy vs bulk via Git LFS.** Dataset files under `data/` (`*.csv`, `*.json`, archives, etc.) are tracked with **Git LFS** (see `.gitattributes`). Pointers live in git; blobs live in LFS storage. Install with `git lfs install` before cloning/pulling data.
5. **Prefer importable modules over notebooks** for anything that must be reproducible for the paper.
6. **Document sources and regeneration** in each stage README (what is read, what is written, how to run).
7. **Paper alignment.** Prefer names that map to methods: base dump, API enrichment, labeling, dataset release.

### Where to put new work

| If you are… | Put it in… |
|-------------|------------|
| Adding IP Australia API clients / text fetchers | `scrape/src/`, config in `scrape/config/` |
| Documenting or placing a new base dump / toy sample | `data/raw/` + update `data/README.md` |
| Defining or changing labels / taxonomy | `classification/schemas/` |
| Implementing labeling logic or models | `classification/src/` |
| Writing pipeline glue / one-shot runners | `scripts/` |
| Changing project-wide deps | root `pyproject.toml` / `requirements.txt` (when added) |

### Expected data flow

```text
data/raw/application-toy.csv   (IP Rapid seed: IDs + status/dates)
            │
            ▼
         scrape/   ── IP Australia API: semantic / textual fields ──►  data/raw/ or data/interim/
            │
            ▼
    classification/  ── labels ──►  data/interim/  ──►  data/processed/
```

Each stage should use **explicit paths** (config or CLI args).

### Out of scope (for now)

- Treating scrape as “download the whole IP Rapid dump” (that dump is the *input*; API text is the scrape target).
- Mixing scrape + classification into a single `src/` tree.
- Committing large dataset files as normal git blobs (use Git LFS patterns in `.gitattributes` instead).

## Repository layout

```text
aus-patent-data/
├── scrape/                 # IP Australia API enrichment
│   ├── src/
│   ├── config/
│   └── README.md
├── classification/         # labeling & taxonomy
│   ├── src/
│   ├── schemas/
│   ├── models/
│   └── README.md
├── data/
│   ├── raw/                # base dumps + (later) raw API payloads
│   │   └── application-toy.csv
│   ├── interim/
│   ├── processed/
│   └── README.md
├── scripts/
├── .gitignore
└── README.md
```

## Status

- Base toy sample: `data/raw/application-toy.csv` (IP Rapid-style application rows).
- Patent Search API enrichment: `scrape/src/patent_search.py` → `part-*.jsonl.gz` shards under the configured `patent_search` output dir.
- Patent Search clean: `scrape/src/clean_patent_search.py` → mirrored `part-*.jsonl.gz` under `data/interim/patent_search_clean/`.
- Classification: PatentBERT claim-level CPC-subclass inference (`classification/src/run_patentbert.py` → `data/interim/patentbert/`).

## Reproduction (partial)

```bash
pip install -r requirements.txt
export IP_AUSTRALIA_CLIENT_ID='...'
export IP_AUSTRALIA_CLIENT_SECRET='...'
python scripts/fetch_patent_search.py
python scripts/clean_patent_search.py
python scripts/analyze_patent_search_clean.py
python scripts/export_patent_text_csvs.py

# PatentBERT needs a separate TF1 / Python 3.7–3.8 env (not root .venv).
# On Apple Silicon see classification/README.md (conda osx-64 + tensorflow=1.15).
pip install -r classification/requirements-patentbert.txt
python scripts/download_patentbert.py
python scripts/run_patentbert.py
```

See `scrape/README.md` for config (client credentials → JWT, `max_responses`, backoff) and idempotent re-runs. See `classification/README.md` for PatentBERT.
