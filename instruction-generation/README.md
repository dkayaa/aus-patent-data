# instruction-generation

Synthetic **instruction-tuning** JSONL from cleaned Australian patents and the WIPO IPC catalog. Peer stage to `classification/` (labeling); this stage does **not** scrape IP Australia.

Seed task methodologies: `methodologies/seed-instruction-data-generation/`.

## Inputs

| Path | Role |
|------|------|
| `data/interim/patent_search_clean/` | Cleaned patent shards (`part-*.jsonl.gz`) |
| `data/ipc-codes/ipc_codes_20260101.jsonl` | IPC titles / definitions for legal reasoning |

## Outputs

`data/interim/instruction_generation/<task>/part-*.jsonl.gz` plus `done_ids.txt` for resume.

Alpaca-style records: `task`, `application_number`, `instruction`, `input`, `output`, `meta`.

### Tasks

| `--task` | Method |
|----------|--------|
| `legal_reasoning` | LLM justification of `primary_ipc` grounded in WIPO catalog |
| `abstract_drafting` | Evol-Instruct instruction pool; claims → official abstract |
| `patent_drafting` | Evol-Instruct instruction pool; abstract → claim 1 |
| `mrc` | LLM extractive Q/A over claims |

## LLM providers

Configured in `config/instruction_generation.yaml` under `llm:`:

- **`provider: local`** (default) — OpenAI-compatible server, e.g. llama.cpp / vLLM / Ollama at `http://127.0.0.1:8080/v1`. Set `OPENAI_API_KEY` to any non-empty value if required.
- **`provider: openrouter`** — `https://openrouter.ai/api/v1` with `OPENROUTER_API_KEY`.

Same client code path; swap via YAML or CLI flags.

## Run

```bash
# from repo root, with .venv active
pip install -r requirements.txt

# local Llama (default config)
.venv/bin/python scripts/generate_instruction_data.py --task legal_reasoning --limit 20

# all four tasks
.venv/bin/python scripts/generate_instruction_data.py --all --limit 50

# OpenRouter
export OPENROUTER_API_KEY='...'
.venv/bin/python scripts/generate_instruction_data.py \
  --task mrc --provider openrouter --model meta-llama/llama-3.1-8b-instruct --limit 10
```

Evol-Instruct pools are generated once per task and cached under `data/interim/instruction_generation/_pools/`.
