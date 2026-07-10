# scrape

**Stage:** enrich known applications with semantic / textual data from IP Australia.

Does **not** define the application universe. That comes from the IP Rapid / data.gov dump (see `data/raw/application-toy.csv` for the toy seed).

## Role

1. Read application identifiers from a base file under `data/raw/` (e.g. `application_number`).
2. Call the IP Australia API (and related endpoints as needed) to fetch abstracts, descriptions, claims, or other semantic text.
3. Write raw API responses into `data/interim/` (one JSON file per application for the patent search fetcher).

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
| **Reads** | `data/raw/application-toy.csv` (`application_number`) |
| **Writes** | `data/interim/patent_search/{application_number}.json` |

Each output file wraps the API JSON:

```json
{
  "application_number": "2022901535",
  "fetched_at": "2026-07-10T01:00:00Z",
  "response": { }
}
```

**Idempotency:** if `{application_number}.json` already exists, that row is skipped. Re-runs do not refetch or duplicate.

## Config knobs (`config/patent_search.yaml`)

| Key | Meaning |
|---|---|
| `auth.token_url` | External Token API URL |
| `auth.client_id_env` / `auth.client_secret_env` | Env vars for OAuth client credentials |
| `fetch.max_responses` | Optional cap on **new** fetches this run (`null` = no cap) |
| `fetch.max_requests_per_minute` | Published API cap (default `600`) |
| `fetch.rate_limit_headroom` | Fraction of that cap to use (default `0.9` → ~540/min, ~0.111s spacing) |
| `fetch.backoff.*` | Exponential backoff on 429/5xx/network errors only |
| `paths.*` | Input CSV, column name, output directory |

On each run with pending work, the script POSTs `client_id` / `client_secret` to the token URL, then uses the returned `access_token` as `Authorization: Bearer …`. Requests are paced under `max_requests_per_minute`; retries use separate exponential backoff.

## How to run

```bash
pip install -r requirements.txt
export IP_AUSTRALIA_CLIENT_ID='...'
export IP_AUSTRALIA_CLIENT_SECRET='...'   # or copy .env.example → .env and load it

# Full toy set (skips any already-written interim files)
python scripts/fetch_patent_search.py

# Cap new fetches for a smoke test
python scripts/fetch_patent_search.py --max-responses 5 -v
```
