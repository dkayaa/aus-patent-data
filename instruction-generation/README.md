# instruction-generation

Synthetic **instruction-tuning** JSONL from cleaned Australian patents and the WIPO IPC catalog. Peer stage to `classification/` (labeling); this stage does **not** scrape IP Australia.

Task methodologies: `methodologies/01-instruction-data-generation/`.

## Inputs

| Path | Role |
|------|------|
| `data/derived/patent_search_clean/` | Cleaned patent shards (`part-*.jsonl.gz`) |
| `data/ipc-codes/ipc_codes_20260101.jsonl` | IPC titles / definitions for IPC reasoning |

## Outputs

`data/derived/instruction_generation/{model_slug}/<task>/part-*.jsonl.gz` plus `done_ids.txt` for resume. The slug is derived from `--model` (e.g. `anthropic/claude-sonnet-4.6` → `anthropic-claude-sonnet-4.6`, `llama3.1:8b` → `llama3.1-8b`). Each generator also gets `manifest.json`. Instruction pools stay shared at `_pools/`.

Alpaca-style records: `task`, `application_number`, `instruction`, `input`, `output`, `meta`.

### Tasks

| `--task` | Method |
|----------|--------|
| `ipc_reasoning` | Instruction pool + LLM justification of `primary_ipc` (skip if no WIPO definition) |
| `abstract_drafting` | Instruction pool; claims → official abstract |
| `mrc` | *(retired from eval/SFT)* Instruction pool + LLM extractive Q/A; keep only for legacy holdings |

## LLM providers

Configured in `config/instruction_generation.yaml` under `llm:`:

- **`provider: local`** (default) — OpenAI-compatible server, e.g. llama.cpp / vLLM / Ollama at `http://127.0.0.1:8080/v1`. Set `OPENAI_API_KEY` to any non-empty value if required.
- **`provider: openrouter`** — `https://openrouter.ai/api/v1` with `OPENROUTER_API_KEY`.

Same client code path; swap via YAML or CLI flags.

## Run

```bash
# from repo root, with .venv active
pip install -r requirements.txt

# local Llama (default config) → data/derived/instruction_generation/llama3.1-8b/
.venv/bin/python scripts/generate_instruction_data.py --task ipc_reasoning --limit 20

# all three tasks
.venv/bin/python scripts/generate_instruction_data.py --all --limit 50

# OpenRouter → data/derived/instruction_generation/anthropic-claude-sonnet-4.6/
export OPENROUTER_API_KEY='...'
.venv/bin/python scripts/generate_instruction_data.py \
  --task mrc --provider openrouter --model anthropic/claude-sonnet-4.6 --limit 10

# Parallel OpenRouter calls (IPC). Abstract drafting is local and stays fast.
# --limit is successful writes this run; already-written ids in done_ids.txt are skipped.
# If one task is already complete, pass --task rather than --all.
.venv/bin/python scripts/generate_instruction_data.py --task ipc_reasoning --limit 1000 \
  --provider openrouter --model meta-llama/llama-3.3-70b-instruct --workers 12 --temperature 0.0

# Stratified expand: cap any single primary_ipc at 0.5% of target (requires WIPO definition)
.venv/bin/python scripts/sample_ipc_apps.py --target 10000 --max-per-symbol-frac 0.005
.venv/bin/python scripts/generate_instruction_data.py --task ipc_reasoning \
  --provider openrouter --model qwen/qwen3-235b-a22b-2507 --workers 12 --temperature 0.0 \
  --only-ids data/derived/instruction_generation/_samples/ipc_reasoning_10k_cap0_5pct.txt \
  --limit 10000
```
Instruction pools (diversified phrasings for a fixed task) are generated once per task and cached under `data/derived/instruction_generation/_pools/`. Reused across generators so smoke tests are comparable. Rebuild one task's pool with `--rebuild-pool` (requires `--task`; overwrites `_pools/<task>.json`). See `methodologies/01-instruction-data-generation/` for the full workflow.
