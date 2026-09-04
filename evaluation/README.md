# evaluation

Held-out **baseline eval** for instruction tasks. Reads frozen Mode 1 `passed/` rows, joins `filedDate`, writes train/val/test, then runs OpenRouter systems zero-shot and 3-shot on the same test IDs and scores them with automatic metrics.

This stage does **not** scrape IP Australia, generate SFT gold, run Mode 1/2 validation, or train.

## What it measures

Paired baselines on a frozen test set: a small instruct student, Llama 3.1 8B, the Llama 3.3 70B teacher, and GPT-4o, each with and without 3 in-context examples. Later SFT should reuse the same test `application_number`s.

**How to read scores**

- **IPC code** vs office `primary_ipc`: unbiased for every system.
- **Abstract** vs office gold: a real ceiling for teacher and GPT.
- **IPC justification / MRC** vs seed `output`: 3.3-written gold, so teacher vs gold is self-agreement; GPT/students matching that wording is not independent quality. Few-shot gold is also 3.3-written (format leak).

## Layout

```text
evaluation/
  config/baselines.yaml
  src/split.py
  src/run_baseline.py
  src/score.py
  README.md
```

Outputs under `data/derived/evaluation/`:

```text
splits/{generator_slug}/
  split_manifest.json
  exemplars.json
  <task>/{train,val,test}.jsonl.gz
predictions/{system_slug}/{prompting}/{task}/part-*.jsonl.gz
scores/{system_slug}/{prompting}/{task}/report.json
scores/summary.json
```

## Splits

Two holds, both required:

| Leak | Rule |
|------|------|
| SFT | Same `application_number` is never in both train and test (assignment is global across IPC / abstract / MRC). |
| Pretraining | Test filings have `filedDate >= 2024-01-01`. One test set for all systems. Qwen does not get a later cutoff. |

Undated apps cannot enter test (`n_missing_date` in the manifest). If the post-cutoff pool is under ~10% of unique apps, the whole pool is test; otherwise a 10% sample is drawn **from the pool only** (seed 42). Train/val is the remainder, 80/20. Test is never padded with older patents.

```bash
# Primary seed generator (Mode 1 passed/): Qwen3-235B
.venv/bin/python scripts/split_eval_data.py --generator qwen/qwen3-235b-a22b-2507
```

## Systems

Temperature **0**. Default `--workers 12`. Reuses `instruction-generation/src/llm.py` (OpenRouter).

| Role | OpenRouter model |
|------|------------------|
| Student A | `qwen/qwen-2.5-7b-instruct` |
| Student B | `meta-llama/llama-3.1-8b-instruct` |
| Teacher | `meta-llama/llama-3.3-70b-instruct` |
| Frontier | `openai/gpt-4o` |

Each system × two regimes on the **same test IDs** (8 jobs):

| Regime | Prompt |
|--------|--------|
| `zeroshot` | Test `instruction` + `input` only. No gold. |
| `fewshot_k3` | k=3 frozen train chats, then the test user turn. |

Exemplars come from **train only**, one frozen list per task for every system. Exemplar `input` is truncated to `fewshot.exemplar_input_chars` (default 4000). `k_effective` is stored when train has fewer than 3 rows.

```bash
export OPENROUTER_API_KEY='...'
.venv/bin/python scripts/run_baselines.py --all --generator qwen/qwen3-235b-a22b-2507
# --prompting zeroshot|fewshot|all (default all)
# --system qwen/qwen-2.5-7b-instruct --task mrc --limit 5
```

Resume via `done_ids.txt` under each prediction dir. `max_tokens`: 1024 IPC/abstract, 256 MRC.

## Scoring

Reuses Mode 1 parsers, **not** Mode 1 pass/fail floors. Nomic is loaded once per score run.

| Task | Headline | Also |
|------|----------|------|
| IPC | Exact code vs `primary_ipc` (strict schema) | Lenient exact (gold symbol anywhere in the output); hierarchical match; format-valid rate; ROUGE-L + Nomic cosine of justification vs gold justification |
| Abstract | ROUGE-L vs gold abstract | Nomic cosine vs gold abstract and vs claims; compression; empty rate |
| MRC | Token-F1 vs gold answer | Exact match; answer in claims; empty rate. No cosine |

Empty generation = fail.

```bash
.venv/bin/python scripts/score_baselines.py --generator qwen/qwen3-235b-a22b-2507
```

## Downstream SFT

QLoRA prepare/train lives in [`sft/`](../sft/). It inherits these frozen splits (no new membership). Same generator slug end-to-end for fair comparison with baselines.

## Deferred

PatentBERT (IPC code-only classifier), post-SFT generate + score, BERTScore/BLEU, LLM-as-a-judge on predictions. Same frozen test IDs later.
