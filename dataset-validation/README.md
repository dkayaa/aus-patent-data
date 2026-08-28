# dataset-validation

Validation of seed instruction JSONL after `instruction-generation/`.

| Mode | What | Entry |
|------|------|--------|
| **1** Programmatic | Schema/IPC (WIPO catalog), ROUGE-L, MRC best-span F1, configurable embedding cosine (Nomic by default), Terms Coverage | `scripts/validate_instruction_data.py` |
| **Cheap judge** (optional) | Same rubric as Mode 2; Claude Haiku 4.5; **all** Mode 1 `passed/` (or pinned IDs). Not a training filter until cascade enrichment is positive. | `scripts/cheap_judge_instruction_data.py` |
| **2** LLM-as-a-judge | Sample Mode 1 `passed/` rows; pointwise 1–5; `pass` from `score >= 4`; closed tags | `scripts/judge_instruction_data.py` |

Methodology: [`methodologies/02-dataset-validation/`](../methodologies/02-dataset-validation/).

## Mode 1 — programmatic

```bash
.venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/validate_instruction_data.py --task ipc_reasoning
.venv/bin/python scripts/validate_instruction_data.py --all --limit 100
.venv/bin/python scripts/validate_instruction_data.py --all \
  --generator anthropic/claude-sonnet-4.6

# choose embedding model (keys in config/embedding_models.yaml)
.venv/bin/python scripts/validate_instruction_data.py --task mrc --embedding-model nomic

# lexical/structural only (no embedding download)
.venv/bin/python scripts/validate_instruction_data.py --task mrc --skip-semantic
```

Reads `data/derived/instruction_generation/{model_slug}/<task>/`. If several generators exist, pass `--generator` (model id or slug). With one generator, that one is used.

The default embedding model is **`nomic`** (`nomic-ai/nomic-embed-text-v1.5`,
8192-token window). The registry in
[`config/embedding_models.yaml`](config/embedding_models.yaml) also provides
`granite`, `granite_small`, and the truncated `minilm` baseline. It records each
model's expected runtime context length, trust setting, and required document
prefix; a context-length mismatch fails validation rather than silently
truncating.

Switch models with `--embedding-model granite` (or `granite_small` / `minilm`)
without editing thresholds.

Faithfulness (ipc_reasoning only, MiniCheck-Flan-T5-Large, **non-gating**):

```bash
.venv/bin/pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
# calibrate first (human good/bad pool) before trusting faithfulness_rate
.venv/bin/python scripts/calibrate_faithfulness_ipc.py
# report → reports/faithfulness_calibration.md
.venv/bin/python scripts/run_faithfulness_ipc.py
# report → reports/faithfulness_ipc.md
```

Design (post-calibration): **atomicise** → drop **META** (alignment markers /
zero claim terms) *before* MiniCheck → score **combined** doc → three-way
band (`SUPPORTED` P≥0.7, `UNDECIDED` mid, `UNSUPPORTED` P<0.3). Wrong-bridge
is out of scope — expert audit (undecided band is the natural review set).

Calibration pool: [`config/faithfulness_calibration.jsonl`](config/faithfulness_calibration.jsonl)
(16 labeled sentences; `difficulty=hard|easy`).

Terms Coverage is an additive, non-gating metric for `abstract_drafting` and
`ipc_reasoning`. It measures how many technical claim terms are retained in the
generated text, using the full claims without an embedding window. Configure it
under `terms_coverage` in
[`config/programmatic.yaml`](config/programmatic.yaml); deterministic
boilerplate exclusions live in
[`config/terms_boilerplate.yaml`](config/terms_boilerplate.yaml).

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
  --ids-file data/derived/instruction_generation_validation/meta-llama-llama-3.3-70b-instruct/ipc_reasoning/llm_judge/done_ids.txt \
  --workers 12
```

Requires `OPENROUTER_API_KEY` in the environment (load `.env` as above). `--generator` selects whose Mode 1 `passed/` rows to grade (same default as Mode 1: the only generator dir). `--model` is the *judge* LLM. Resume via `{model_slug}/<task>/llm_judge/done_ids.txt`. `--ids-file` requires a single `--task`.

| Path | Content |
|------|---------|
| `llm_judge/passed/` | `pass: true` + `meta.llm_judge` |
| `llm_judge/rejected/` | failed grades |
| `llm_judge/report.json` | n_judged, mean score, pass rate, closed-set failure tags |
| `llm_judge/done_ids.txt` | resume / pinned sample set |

Config: [`config/llm_judge.yaml`](config/llm_judge.yaml). Rubrics: [`src/judge_prompts.py`](src/judge_prompts.py).

## Cheap judge (optional cascade screen)

Same prompts and `pass_score_min` as Mode 2. Writes `{task}/cheap_judge/`, never `llm_judge/`. Default YAML grades **every** Mode 1 survivor (`sample_size: null`) with OpenRouter `anthropic/claude-haiku-4.5`. Pin the Mode 2 IDs first to measure enrichment.

```bash
# OpenRouter Claude Haiku 4.5 (default cheap_judge.yaml)
set -a && source .env && set +a
.venv/bin/python scripts/cheap_judge_instruction_data.py --task ipc_reasoning \
  --generator meta-llama/llama-3.3-70b-instruct \
  --ids-file data/derived/instruction_generation_validation/meta-llama-llama-3.3-70b-instruct/ipc_reasoning/llm_judge/done_ids.txt

# Full Mode 1 passed/ for one task
.venv/bin/python scripts/cheap_judge_instruction_data.py --task ipc_reasoning \
  --generator meta-llama/llama-3.3-70b-instruct

# Local Ollama instead
.venv/bin/python scripts/cheap_judge_instruction_data.py --task ipc_reasoning \
  --provider local --model llama3.1:8b --base-url http://127.0.0.1:11434/v1 --workers 1
```

Then:

```bash
.venv/bin/python scripts/calibrate_llm_judge.py \
  --generator meta-llama/llama-3.3-70b-instruct --task ipc_reasoning
```

Writes `{model_slug}/cheap_judge_cascade.json` when both `cheap_judge/` and `llm_judge/` exist. Hinge: `frontier_pass_rate_given_cheap_pass` vs `frontier_pass_rate` (`enrichment`). Do **not** feed Mode 2 from `cheap_judge/passed/` unless enrichment is clearly positive. Do not train on cheap-pass shards.

Equivalent to `scripts/judge_instruction_data.py --config dataset-validation/config/cheap_judge.yaml …`. `--full` forces a full corpus pass even with the Mode 2 YAML (expensive on Sonnet).

## Mode 2 calibration hook

```bash
.venv/bin/python scripts/calibrate_llm_judge.py \
  --generator meta-llama/llama-3.3-70b-instruct
.venv/bin/python scripts/calibrate_llm_judge.py \
  --generator meta-llama/llama-3.3-70b-instruct --export-csv
```

Prints score histograms, pass rates, and tag counts. If `{task}/cheap_judge/` exists, also writes cheap↔frontier cascade stats to `{model_slug}/cheap_judge_cascade.json`. If `{model_slug}/human_audit.jsonl` exists, also writes Cohen’s κ, a confusion matrix, and a `pass_score_min` sweep to `{model_slug}/llm_judge_calibration.json`. `--export-csv` writes a **blind** audit spreadsheet (no judge scores).
