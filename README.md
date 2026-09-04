# aus-patent-data

Research repository for constructing an Australian patent dataset. This repo is the **build recipe** for the dataset that supports the associated research paper.

## Pipeline (actual)

1. **Base identifiers** — start from an IP Rapid / data.gov application dump (toy sample checked in under `data/raw/`).
2. **Scrape** — use those application numbers to fetch **semantic textual data** from the IP Australia API into `data/`.
3. **Classification** — label / categorize the enriched records.
4. **Instruction generation** — synthesize instruction-tuning JSONL from cleaned patents (optional peer stage).
5. **Dataset validation** — Mode 1 programmatic checks and optional Mode 2 LLM-as-a-judge on that JSONL.
6. **Eval** — freeze a temporal test split and run OpenRouter zero-shot / 3-shot baselines (no training).
7. **Release artifacts** — final tables under `data/processed/`.

## For Cursor agents

Read this section before changing anything.

### Purpose

Work is organized by **pipeline stage**, not by language or framework:

| Stage | Directory | Responsibility |
|-------|-----------|----------------|
| Base + artifacts | `data/` | Seed dumps (`raw/`), scrape outputs, derived joins, final dataset. |
| Enrich | `scrape/` | Call IP Australia (and related) APIs to fetch semantic text for known application numbers. Does not invent the application list. |
| Label | `classification/` | Apply taxonomies, rules, or models. No API fetching of patent text. |
| Instruction SFT | `instruction-generation/` | Synthetic instruction-tuning JSONL from cleaned patents + IPC catalog via LLM (local or OpenRouter). |
| Dataset validation | `dataset-validation/` | Mode 1 programmatic checks + optional cheap LLM judge + Mode 2 LLM-as-a-judge (sample of Mode 1 `passed/`). |
| Eval | `evaluation/` | Frozen temporal test split of Mode 1 `passed/` + OpenRouter zero-shot / 3-shot baselines + automatic scores. |
| Train SFT | `sft/` | Flat SFT JSONL from frozen eval splits + per-dataset QLoRA (TRL/PEFT, CUDA). Does not invent splits. |

### Hard rules

1. **Do not mix stages.** API clients and downloaders stay in `scrape/`. Labeling stays in `classification/`. Synthetic SFT generation stays in `instruction-generation/`. Validation scoring stays in `dataset-validation/`. Baseline eval stays in `evaluation/`. Student QLoRA stays in `sft/`.
2. **Scrape is enrichment, not discovery.** The application universe comes from IP Rapid (or a full dump later). Scrapers take `application_number` (and related IDs) from `data/raw/` and fetch text/metadata from IP Australia.
3. **Pipeline direction:** `data/raw` (base) → `scrape/` → enriched artifacts in `data/` → (`classification/` and/or `instruction-generation/` → `dataset-validation/` → `evaluation/` → `sft/`) → `data/derived/` / `data/processed/`.
4. **Toy vs bulk via Git LFS.** Dataset files under `data/` (`*.csv`, `*.json`, archives, etc.) are tracked with **Git LFS** (see `.gitattributes`). Pointers live in git; blobs live in LFS storage. Install with `git lfs install` before cloning/pulling data.
5. **Prefer importable modules over notebooks** for anything that must be reproducible for the paper.
6. **Document sources and regeneration** in each stage README (what is read, what is written, how to run).
7. **Paper alignment.** Prefer names that map to methods: base dump, API enrichment, labeling, instruction generation, dataset release.

### Where to put new work

| If you are… | Put it in… |
|-------------|------------|
| Adding IP Australia API clients / text fetchers | `scrape/src/`, config in `scrape/config/` |
| Documenting or placing a new base dump / toy sample | `data/raw/` + update `data/README.md` |
| Defining or changing labels / taxonomy | `classification/schemas/` |
| Implementing labeling logic or models | `classification/src/` |
| Building synthetic instruction-tuning data | `instruction-generation/src/`, config in `instruction-generation/config/` |
| Scoring Mode 1 / Mode 2 on generated SFT | `dataset-validation/src/`, config in `dataset-validation/config/` |
| Frozen test splits and OpenRouter baselines | `evaluation/src/`, config in `evaluation/config/` |
| Prepare SFT JSONL / QLoRA train | `sft/src/`, config in `sft/config/` (`requirements-sft.txt` on GPU) |
| Writing pipeline glue / one-shot runners | `scripts/` |
| Changing project-wide deps | root `pyproject.toml` / `requirements.txt` (when added) |

### Expected data flow

```text
data/raw/application-toy.csv   (IP Rapid seed: IDs + status/dates)
            │
            ▼
         scrape/   ── IP Australia API: semantic / textual fields ──►  data/derived/
            │
            ├─► classification/          ── labels / models ──► data/derived/
            └─► instruction-generation/ ── synthetic SFT JSONL ──► data/derived/instruction_generation/{model_slug}/
                        │
                        ▼
              dataset-validation/ ── Mode 1 passed/ + optional cheap_judge + Mode 2 sample ──► data/derived/instruction_generation_validation/
                        │
                        ▼
                  evaluation/ ── frozen splits + OpenRouter baselines ──► data/derived/evaluation/
                        │
                        ▼
                      sft/ ── prepare flat JSONL + QLoRA (inherits splits) ──► data/derived/sft/
```

Each stage should use **explicit paths** (config or CLI args).

### Out of scope (for now)

- Treating scrape as “download the whole IP Rapid dump” (that dump is the *input*; API text is the scrape target).
- Mixing scrape + classification + instruction-generation into a single `src/` tree.
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
├── instruction-generation/ # synthetic instruction-tuning JSONL
│   ├── src/
│   ├── config/
│   └── README.md
├── dataset-validation/     # Mode 1 programmatic + Mode 2 LLM-as-a-judge
│   ├── src/
│   ├── config/
│   └── README.md
├── evaluation/             # frozen splits + OpenRouter baselines
│   ├── src/
│   ├── config/
│   └── README.md
├── sft/                    # prepare flat SFT + QLoRA train (CUDA)
│   ├── src/
│   ├── config/
│   └── README.md
├── methodologies/          # numbered method notes (01 generation, 02 validation, 03 eval)
├── data/
│   ├── raw/                # base dumps + (later) raw API payloads
│   │   └── application-toy.csv
│   ├── derived/
│   ├── processed/
│   └── README.md
├── scripts/
├── .gitignore
└── README.md
```

## Status

- Base toy sample: `data/raw/application-toy.csv` (IP Rapid-style application rows).
- Patent Search API enrichment: `scrape/src/patent_search.py` → `part-*.jsonl.gz` shards under the configured `patent_search` output dir.
- Patent Search clean: `scrape/src/clean_patent_search.py` → mirrored `part-*.jsonl.gz` under `data/derived/patent_search_clean/`.
- Classification: PatentBERT claim-level CPC-subclass inference (`classification/src/run_patentbert.py` → `data/derived/patentbert/`).
- Instruction generation: synthetic SFT JSONL (`instruction-generation/` → `data/derived/instruction_generation/{model_slug}/`).
- Dataset validation: Mode 1 `scripts/validate_instruction_data.py`; Mode 2 sample judge `scripts/judge_instruction_data.py` → `data/derived/instruction_generation_validation/{model_slug}/`.
- Eval: frozen splits `scripts/split_eval_data.py`; OpenRouter baselines `scripts/run_baselines.py`; scores `scripts/score_baselines.py` → `data/derived/evaluation/`.
- SFT: prepare from splits `scripts/prepare_sft_data.py`; QLoRA `scripts/run_sft.py` → `data/derived/sft/` (GPU: `requirements-sft.txt`).

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

# Instruction-tuning JSONL (local Llama OpenAI-compatible server by default).
python scripts/generate_instruction_data.py --task ipc_reasoning --limit 20
```

See `scrape/README.md` for config (client credentials → JWT, `max_responses`, backoff) and idempotent re-runs. See `classification/README.md` for PatentBERT. See `instruction-generation/README.md` for LLM provider swap (local / OpenRouter). See `evaluation/README.md` for frozen splits and OpenRouter baselines.
