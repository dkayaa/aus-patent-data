# scripts

Optional one-shot and end-to-end runners that glue stages together.

Prefer thin wrappers that call into `scrape/` and `classification/` modules rather than duplicating logic here.

| Script | Calls | Purpose |
|--------|-------|---------|
| `fetch_patent_search.py` | `scrape/src/patent_search.py` | Enrich application numbers via Patent Search API → `part-*.jsonl.gz` + `fetched_ids.txt` |
| `clean_patent_search.py` | `scrape/src/clean_patent_search.py` | Reshape raw shards → `patent_search_clean/part-*.jsonl.gz` (parsed claims) |
| `pack_patent_search_json_to_jsonl.py` | `scrape/src/pack_patent_search_json.py` | One-shot: legacy `{application_number}.json` → `part-*.jsonl.gz` |
| `summarize_patent_search.py` | (standalone; uses `jsonl_gz`) | Summarize field coverage in a patent_search shard folder |
| `analyze_patent_search_clean.py` | (standalone; uses `jsonl_gz`) | Stats + plots from clean shards → `data/tables/`, `data/plots/` |

### Pack legacy per-file JSON

```bash
python scripts/pack_patent_search_json_to_jsonl.py \
  --input-dir data/interim/patent_search
# Replace existing shards if re-packing:
python scripts/pack_patent_search_json_to_jsonl.py \
  --input-dir /Volumes/T7/patent-aus/data/interim/patent_search --force
```
