# classification

**Stage:** labeling and enrichment of already-fetched records.

Reads base + API-enriched tables from `data/`, applies taxonomies / rules / models, and writes labeled outputs to `data/interim/` and `data/processed/`.

## Rules

- No IP Australia (or other patent-source) API fetching here — that belongs in `scrape/`.
- Keep label definitions in `schemas/` separate from application code in `src/`.
- Large model weights belong under `models/` (gitignored) or external storage; document how to obtain them.

## Layout

- `src/` — labeling pipelines, heuristics, model inference
- `schemas/` — taxonomy and label definitions (cite-friendly)
- `models/` — trained artifacts or pointers (bulk files ignored)

## Inputs / outputs

- **Reads:** enriched tables from `data/raw/` and/or `data/interim/` (after scrape), plus the base application dump as needed
- **Writes:** `data/interim/`, `data/processed/`

## How to run

TBD once classifiers exist.
