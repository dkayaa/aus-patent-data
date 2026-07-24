# scrape

**Stage:** enrich known applications with semantic / textual data from IP Australia.

Does **not** define the application universe. That comes from the IP Rapid / data.gov dump (see `data/raw/application-toy.csv` for the toy seed).

## Role

1. Read application identifiers from a base file under `data/raw/` (e.g. `application_number`).
2. Call the IP Australia API (and related endpoints as needed) to fetch abstracts, descriptions, claims, or other semantic text.
3. Write raw API responses into `data/interim/` as batched `part-*.jsonl.gz` shards.

## Rules

- No classification, labeling, or taxonomy logic here.
- Prefer configurable endpoints, rate limits, auth, and input/output paths under `config/`.
- Document each API (URL, auth, ToS/legal notes, fields returned) before adding a fetcher.
- Be resilient to rate limits and partial failures; prefer resumable runs keyed by `application_number`.

## Layout

- `src/` — API clients and fetch utilities
- `config/` — endpoints, credentials env keys, rate limits, path settings

## APIs

### Australian Patent Search API

| | |
|---|---|
| Portal | [Australian Patent Search API](https://portal.api.ipaustralia.gov.au/s/communityapi/a082w00000TJfb7AAD/developersaustralianpatentsearchapi) |
| Production base | `https://production.api.ipaustralia.gov.au/public/australian-patent-search-api/v1` |
| Endpoint used | `GET /patent/{ipRightIdentifier}` |
| Auth | OAuth2 `client_credentials` → JWT via [External Token API](https://production.api.ipaustralia.gov.au/public/external-token-api/v1/access_token); send as `Authorization: Bearer …` |
| Config | `config/patent_search.yaml` |
| Module | `src/patent_search.py` |

The sibling [Design Search API](https://portal.api.ipaustralia.gov.au/s/communityapi/a082w000000LObRAAW/developersaustraliandesignsearchapi) uses the same portal pattern (`GET /design/{id}`) but is not used here — the toy seed is patents only.

## Inputs / outputs

| | Path |
|---|---|
| **Reads** | `paths.input_csv` in config (e.g. `data/raw/application-from-2000.csv`) (`application_number`) |
| **Writes** | `paths.output_dir` — `part-NNNNN.jsonl.gz` shards, open `part-NNNNN.jsonl` while a shard is filling, and `fetched_ids.txt` |

Each JSONL line wraps the API JSON (compact, one object per line):

```json
{"application_number":"2022901535","fetched_at":"2026-07-10T01:00:00Z","response":{}}
```

**Idempotency / resume:** application numbers listed in `fetched_ids.txt` are skipped. On startup, any open `part-*.jsonl` is also scanned so a crash between line write and id append does not cause refetch. Re-runs do not duplicate completed IDs.

**Legacy per-file JSON:** if you still have `{application_number}.json` files, pack them once with `python scripts/pack_patent_search_json_to_jsonl.py --input-dir …` (see `scripts/README.md`).

## Config knobs (`config/patent_search.yaml`)

| Key | Meaning |
|---|---|
| `auth.token_url` | External Token API URL |
| `auth.client_id_env` / `auth.client_secret_env` | Env vars for OAuth client credentials |
| `fetch.shard_size` | Records per finalized `part-*.jsonl.gz` (default `1000`) |
| `fetch.max_responses` | Optional cap on **new** fetches this run (`null` = no cap) |
| `fetch.max_requests_per_minute` | Request cap (default `500`) |
| `fetch.rate_limit_headroom` | Fraction of that cap to use (default `0.9` → ~450/min, ~0.133s spacing) |
| `fetch.backoff.*` | Exponential backoff on 429/5xx/network errors only |
| `paths.output_dir` | Where shards are written (absolute or repo-relative; created if missing) |
| `paths.input_csv` / `application_number_column` | Seed CSV and ID column |

On each run with pending work, the script POSTs `client_id` / `client_secret` to the token URL, then uses the returned `access_token` as `Authorization: Bearer …`. Requests are paced under `max_requests_per_minute`; retries use separate exponential backoff.

## How to run

```bash
pip install -r requirements.txt
export IP_AUSTRALIA_CLIENT_ID='...'
export IP_AUSTRALIA_CLIENT_SECRET='...'   # or copy .env.example → .env and load it

# Full seed set (skips IDs already in fetched_ids.txt)
python scripts/fetch_patent_search.py

# Cap new fetches for a smoke test
python scripts/fetch_patent_search.py --max-responses 5 -v
```

## Cleaner (reshape interim)

Post-process raw Patent Search caches into a flatter analysis-ready interim set (no API calls).

| | |
|---|---|
| Config | `config/clean_patent_search.yaml` |
| Module | `src/clean_patent_search.py` |
| **Reads** | `paths.input_dir` (default `data/interim/patent_search/part-*.jsonl.gz`) |
| **Writes** | Mirrored `paths.output_dir/part-*.jsonl.gz` plus `summary.json` |

Each cleaned line keeps status/title/IPC/parties/dates plus `publishedDocuments[]` with `abstract` and parsed `claims` (`claims_parse_ok` flags OCR/split failures). Outputs are always rewritten (deterministic, cheap). The run also writes `summary.json` in the output dir with counts and the full `claims_parse_failures` list.

```bash
python scripts/clean_patent_search.py
python scripts/clean_patent_search.py --limit 5 -v
```
