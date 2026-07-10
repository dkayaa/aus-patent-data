# scrape

**Stage:** enrich known applications with semantic / textual data from IP Australia.

Does **not** define the application universe. That comes from the IP Rapid / data.gov dump (see `data/raw/application-toy.csv` for the toy seed).

## Role

1. Read application identifiers from a base file under `data/raw/` (e.g. `application_number`).
2. Call the IP Australia API (and related endpoints as needed) to fetch abstracts, descriptions, claims, or other semantic text.
3. Write raw API responses or normalized text tables into `data/raw/` and/or `data/interim/` (paths decided per fetcher; document them here when added).

## Rules

- No classification, labeling, or taxonomy logic here.
- Prefer configurable endpoints, rate limits, auth, and input/output paths under `config/`.
- Document each API (URL, auth, ToS/legal notes, fields returned) before adding a fetcher.
- Be resilient to rate limits and partial failures; prefer resumable runs keyed by `application_number`.

## Layout

- `src/` — API clients and fetch utilities
- `config/` — endpoints, credentials env keys, rate limits, path settings

## Inputs / outputs

- **Reads:** `data/raw/` base dumps (start with `application-toy.csv`)
- **Writes:** enriched semantic payloads / tables under `data/raw/` or `data/interim/` (TBD per scraper)

## How to run

TBD once scrapers exist.
