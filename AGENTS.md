# AGENTS

Guidance for Cursor agents working in this repository.

## Python environment

- **`.venv/` at the repo root is the authoritative Python environment.**
- Prefer `.venv/bin/python` and `.venv/bin/pip` for installs and script runs.
- Do not invent a second venv, rely on a global interpreter, or use pyenv/system Python when `.venv` exists.
- If `.venv` is missing, create it with `python3 -m venv .venv` and install from `requirements.txt` before running pipeline code.

## Pipeline stages

Work is organized by **pipeline stage**, not by language or framework. See also the root `README.md` “For Cursor agents” section.

| Stage | Directory | Responsibility |
|-------|-----------|----------------|
| Base + artifacts | `data/` | Seed dumps (`raw/`), scrape outputs, interim joins, final dataset. |
| Enrich | `scrape/` | Call IP Australia (and related) APIs to fetch semantic text for known application numbers. Does not invent the application list. |
| Label | `classification/` | Apply taxonomies, rules, or models. No API fetching of patent text. |
| Instruction SFT | `instruction-generation/` | Build synthetic instruction-tuning JSONL from cleaned patents + IPC catalog via an LLM (local OpenAI-compatible server or OpenRouter). No IP Australia scraping; not taxonomy labeling. |

### Hard rules

1. **Do not mix stages.** API clients and downloaders stay in `scrape/`. Labeling stays in `classification/`. Synthetic SFT generation stays in `instruction-generation/`.
2. **Scrape is enrichment, not discovery.** The application universe comes from IP Rapid (or a full dump later).
3. **Pipeline direction:** `data/raw` → `scrape/` → `data/interim/patent_search_clean` → (`classification/` and/or `instruction-generation/`) → further `data/interim/` / `data/processed/`.
4. Seed task methodology notes live under `methodologies/seed-instruction-data-generation/`; runnable generation code lives under `instruction-generation/`. Evolved-instruction methods can live alongside under `methodologies/` as they are added.
