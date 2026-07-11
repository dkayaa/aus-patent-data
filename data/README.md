# data

Dataset artifacts and seed dumps (not pipeline source code).

## Subdirs

| Subdir | Meaning |
|--------|---------|
| `raw/` | Base IP Rapid dumps and raw API payloads from scrape |
| `interim/` | Joined / cleaned / partially labeled tables |
| `processed/` | Final labeled dataset for analysis / release |

## Base seed (checked in)

| File | Source | Role |
|------|--------|------|
| `raw/application-toy.csv` | Toy subset of the IP Rapid application dump (data.gov) | Seed application list for development and scraper prototyping |

### `application-toy.csv` columns

Metadata / lifecycle fields only (no patent body text):

- `ip_right_type`, `application_number`, `ip_right_sub_type`, `status`
- `application_date`, `earliest_filed_date`, `priority_date`
- `gained_registration_status_date`, `gained_enforceable_status_date`, `enforceable_from_date`, `deemed_retired_date`

Scrapers use `application_number` (and related IDs) to pull semantic text from the IP Australia API. Full IP Rapid dumps may be added later under `raw/` and should remain gitignored if large.

## Rules

- Small toys/samples may be committed; bulk dumps and large API caches are gitignored.
- Prefer regenerating enriched and labeled files from `scrape/` + `classification/` over hand-editing.
- Update this README when schema or row counts change.

## Interim (scrape outputs)

| Path | Producer | Role |
|------|----------|------|
| Configured by `scrape/config/patent_search.yaml` → `paths.output_dir` (default external: `/Volumes/T7/patent-aus/data/interim/patent_search/`) | `scrape/src/patent_search.py` | Raw Patent Search API responses |

## Schema (enriched / final)

TBD once API fields and labels are fixed.

## Regeneration

```bash
export IP_AUSTRALIA_CLIENT_ID='...'
export IP_AUSTRALIA_CLIENT_SECRET='...'
python scripts/fetch_patent_search.py
```

See `scrape/README.md` for config and idempotency notes.
