# Fine-tuning on tasks
---

## Objective
Adapt one or more open-source instruction-capable LLMs on the validated Australian patent SFT corpus so they improve on the three seed tasks (IPC reasoning, abstract drafting, MRC) relative to their untuned base and to external baselines.

## Inputs
* Prepared flat JSONL from `scripts/prepare_sft_data.py` → `data/derived/sft/{generator_slug}/<dataset>/` (inherits frozen `evaluation/splits/` membership; do not re-split).
* Source rows are Mode 1 `passed/` material already assigned by `scripts/split_eval_data.py` (global `application_number` across tasks; temporal test pool; seed 42).
* Alpaca-style fields: `instruction`, `input`, `output` (plus `dataset` / `task`, `application_number`, `meta`).

## Student models (candidates)
Prefer openly weights-available models that fit local or modest cloud GPU budgets, e.g.:
* Small/mid instruct bases (≈1B–8B): Llama / Qwen / Mistral instruct variants of similar size for fair head-to-heads.
* Optional larger student if compute allows (same recipe, not a different methodology).

Document exact Hugging Face IDs, tokenizer, and chat template in the run config when implemented.

## Training modes

### Per-dataset SFT (implemented)
Train a specialist adapter via `scripts/run_sft.py --dataset …` on one of:
* `ipc_reasoning_full` — Classification + Justification
* `ipc_reasoning_classification_only` — Classification line only (ablation)
* `abstract_drafting` / `mrc` — pass-through gold / teacher Q–A

Default stack: HF + PEFT QLoRA + TRL on CUDA (`sft/config/sft.yaml`; student default `meta-llama/Llama-3.1-8B-Instruct`). Train on train, validate on val, never fit on test.

### Multi-task SFT
Pool datasets (stratified or proportional sampling). Out of scope for the current `sft/` runners; one checkpoint as a later ablation unless specialists clearly win.

### Parameter-efficient default
Prefer **LoRA / QLoRA** adapters over full fine-tunes for reproducibility and cost:
* Rank / alpha / target modules recorded in config
* Same max sequence length and packing policy across students when comparing
* Early stopping or fixed epoch budget on a held-out **validation** slice (not the final test set)

## Task-aware formatting
* Keep the generation-time instruction pools: students see diverse phrasings of the same task.
* Prompt format: chat template of the base model; train on assistant completion of `output` given system (optional) + user (`instruction` + `input`).
* **ipc_reasoning:** target remains `Classification: <IPC>\nJustification: …` so classification accuracy is measurable.
* **mrc:** short extractive answers; avoid length penalties that favor verbose drafting styles.

## Splits and leakage
* **Do not invent splits in `sft/`.** Reuse `evaluation/splits/{generator_slug}/` so the same `application_number` never appears in both train and test (assignment is global across IPC / abstract / MRC), and the temporal holdout matches baselines (`filedDate >=` YAML cutoff, currently `2024-01-01`, seed 42).
* IPC-section stratification is desirable where possible but is **not** implemented in `evaluation/src/split.py` today; prepare does not add it.
* Freeze split manifests before any student training; prepare copies seed/cutoff into each dataset `manifest.json` when present.
* Use one `generator_slug` end-to-end for SFT vs baseline tables.

## Non-goals
* Continued pretraining on raw patent dumps (domain CPT) unless added as a separate ablation.
* RLHF / preference optimization in the first pass — SFT only unless human preference data is collected later.
* Mixing scrape or classification pipelines into the trainer; this stage consumes frozen JSONL only.
* Changing the eval temporal cutoff or adding IPC-section stratification inside `prepare_sft_data` (inherit `evaluation/` YAML / `split.py` as-is).

## Runnable
See [`sft/README.md`](../../sft/README.md): `scripts/prepare_sft_data.py`, `scripts/run_sft.py`, `requirements-sft.txt`.
