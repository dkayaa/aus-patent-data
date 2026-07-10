# scripts

Optional one-shot and end-to-end runners that glue stages together.

Prefer thin wrappers that call into `scrape/` and `classification/` modules rather than duplicating logic here.

| Script | Calls | Purpose |
|--------|-------|---------|
| `fetch_patent_search.py` | `scrape/src/patent_search.py` | Enrich toy/base application numbers via Patent Search API → `data/interim/patent_search/` |
