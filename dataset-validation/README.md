# dataset-validation

Validation of seed instruction JSONL after `instruction-generation/`.

| Mode | What | Entry |
|------|------|--------|
| **1** Programmatic | Schema/IPC (incl. WIPO catalog), ROUGE-L, MRC best-span F1, Nomic Embed cosine (IPC vs WIPO definition; abstract vs claims) | `scripts/validate_instruction_data.py` |
| **2** LLM-as-a-judge | Sample Mode 1 `passed/` rows; pointwise 1–5 scores via OpenRouter | `scripts/judge_instruction_data.py` |

Methodology: [`methodologies/02-dataset-validation/`](../methodologies/02-dataset-validation/).

## Mode 1 — programmatic

```bash
.venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/validate_instruction_data.py --task ipc_reasoning
.venv/bin/python scripts/validate_instruction_data.py --all --limit 100
.venv/bin/python scripts/validate_instruction_data.py --all \
  --generator anthropic/claude-sonnet-4.6

# lexical/structural only (no embedding download)
.venv/bin/python scripts/validate_instruction_data.py --task mrc --skip-semantic
```

Reads `data/derived/instruction_generation/{model_slug}/<task>/`. If several generators exist, pass `--generator` (model id or slug). With one generator, that one is used.

Outputs under `data/derived/instruction_generation_validation/{model_slug}/<task>/`:

| Path | Content |
|------|---------|
| `passed/part-*.jsonl.gz` | Kept rows + `meta.validation.scores` |
| `rejected/part-*.jsonl.gz` | Failures + `meta.validation.failed_rules` |
| `report.json` | Counts and mean scores |

Config: [`config/programmatic.yaml`](config/programmatic.yaml).

## Mode 2 — LLM-as-a-judge (sample, not full corpus)

Grades a **deterministic sample** of Mode 1 survivors (`sample_size: 50` per task by default; `--limit` overrides). Not a 500k full-pass.

```bash
set -a && source .env && set +a

.venv/bin/python scripts/judge_instruction_data.py --task ipc_reasoning
.venv/bin/python scripts/judge_instruction_data.py --all --limit 50
.venv/bin/python scripts/judge_instruction_data.py --task mrc \
  --generator anthropic/claude-sonnet-4.6 \
  --provider openrouter --model anthropic/claude-sonnet-4.6
```

Requires `OPENROUTER_API_KEY` in the environment (load `.env` as above). `--generator` selects whose Mode 1 `passed/` rows to grade (same default as Mode 1: the only generator dir). `--model` is the *judge* LLM. Resume via `{model_slug}/<task>/llm_judge/done_ids.txt`.

| Path | Content |
|------|---------|
| `llm_judge/passed/` | `pass: true` + `meta.llm_judge` |
| `llm_judge/rejected/` | failed grades |
| `llm_judge/report.json` | n_judged, mean score, pass rate, failure tags |
| `llm_judge/done_ids.txt` | resume set |

Config: [`config/llm_judge.yaml`](config/llm_judge.yaml). Rubrics: [`src/judge_prompts.py`](src/judge_prompts.py).
