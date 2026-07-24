# scripts

Optional one-shot and end-to-end runners that glue stages together.

Prefer thin wrappers that call into `scrape/` and `classification/` modules rather than duplicating logic here.

| Script | Calls | Purpose |
|--------|-------|---------|
| `fetch_patent_search.py` | `scrape/src/patent_search.py` | Enrich toy/base application numbers via Patent Search API → `data/interim/patent_search/` |
| `clean_patent_search.py` | `scrape/src/clean_patent_search.py` | Reshape raw Patent Search interim JSON → `data/interim/patent_search_clean/` (parsed claims) |
| `summarize_patent_search.py` | (standalone) | Summarize field coverage in a patent_search interim folder |
