# dataset-validation

Validation of seed instruction JSONL after `instruction-generation/`.

| Mode | What | Entry |
|------|------|--------|
| **1** Programmatic | Schema/IPC (WIPO catalog), ROUGE-L, MRC best-span F1, Nomic cosine (IPC vs WIPO + claims; abstract vs claims pairing) | `scripts/validate_instruction_data.py` |
| **2** LLM-as-a-judge | Sample Mode 1 `passed/` rows; pointwise 1–5; `pass` from `score >= 4`; closed tags | `scripts/judge_instruction_data.py` |

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

`pass` is derived in code as `score >= pass_score_min` (default 4). Failure tags are a closed set per task. IPC payloads include WIPO title + definition from the catalog. Same Accept definition as Mode 3 (`methodologies/02-dataset-validation/`).

```bash
set -a && source .env && set +a

.venv/bin/python scripts/judge_instruction_data.py --task ipc_reasoning
.venv/bin/python scripts/judge_instruction_data.py --all --limit 50
.venv/bin/python scripts/judge_instruction_data.py --task mrc \
  --generator anthropic/claude-sonnet-4.6 \
  --provider openrouter --model anthropic/claude-sonnet-4.6

# Concurrent OpenRouter calls (same --workers pattern as instruction generation)
.venv/bin/python scripts/judge_instruction_data.py --all --limit 50 \
  --generator meta-llama/llama-3.3-70b-instruct --workers 12

# Re-grade a frozen ID list (calibration sample; skips seed shuffle)
.venv/bin/python scripts/judge_instruction_data.py --task ipc_reasoning \
  --generator meta-llama/llama-3.3-70b-instruct \
  --ids-file data/derived/instruction_generation_validation/meta-llama-llama-3.3-70b-instruct/ipc_reasoning/llm_judge_v0/done_ids.txt \
  --workers 12
```

Requires `OPENROUTER_API_KEY` in the environment (load `.env` as above). `--generator` selects whose Mode 1 `passed/` rows to grade (same default as Mode 1: the only generator dir). `--model` is the *judge* LLM. Resume via `{model_slug}/<task>/llm_judge/done_ids.txt`. `--ids-file` requires a single `--task`.

| Path | Content |
|------|---------|
| `llm_judge/passed/` | `pass: true` + `meta.llm_judge` |
| `llm_judge/rejected/` | failed grades |
| `llm_judge/report.json` | n_judged, mean score, pass rate, closed-set failure tags |
| `llm_judge/done_ids.txt` | resume / pinned sample set |
| `llm_judge_v0/` | archived pre-protocol grades (untouched) |

Config: [`config/llm_judge.yaml`](config/llm_judge.yaml). Rubrics: [`src/judge_prompts.py`](src/judge_prompts.py).

## Mode 2 calibration hook

```bash
.venv/bin/python scripts/calibrate_llm_judge.py \
  --generator meta-llama/llama-3.3-70b-instruct
.venv/bin/python scripts/calibrate_llm_judge.py \
  --generator meta-llama/llama-3.3-70b-instruct --export-csv
```

Prints score histograms, pass rates, and tag counts. If `{model_slug}/human_audit.jsonl` exists, also writes Cohen’s κ, a confusion matrix, and a `pass_score_min` sweep to `{model_slug}/llm_judge_calibration.json`. `--export-csv` writes a **blind** audit spreadsheet (no judge scores).
