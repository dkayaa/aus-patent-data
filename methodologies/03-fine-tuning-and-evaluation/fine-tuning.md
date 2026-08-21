# Fine-tuning on tasks
---

## Objective
Adapt one or more open-source instruction-capable LLMs on the validated Australian patent SFT corpus so they improve on the three seed tasks (IPC reasoning, abstract drafting, MRC) relative to their untuned base and to external baselines.

## Inputs
* Train / validation rows from `data/derived/instruction_generation_validation/{model_slug}/<task>/passed/` (and any human-audited or judge-filtered subsets once frozen).
* Alpaca-style fields: `instruction`, `input`, `output` (plus `task`, `application_number`, `meta` for stratification and leakage control).

## Student models (candidates)
Prefer openly weights-available models that fit local or modest cloud GPU budgets, e.g.:
* Small/mid instruct bases (≈1B–8B): Llama / Qwen / Mistral instruct variants of similar size for fair head-to-heads.
* Optional larger student if compute allows (same recipe, not a different methodology).

Document exact Hugging Face IDs, tokenizer, and chat template in the run config when implemented.

## Training modes

### Per-task SFT
Train a specialist checkpoint on one task’s JSONL. Best for isolating whether the data helps that capability; report three specialists when comparing task-wise.

### Multi-task SFT
Pool all three tasks (stratified or proportional sampling). One checkpoint serves as the primary “patent instruction” model for the paper unless specialists clearly win.

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
* Split by `application_number` (or patent family) so the same filing never appears in both train and test.
* Stratify by task and, where possible, IPC section (IPC reasoning) / length buckets (abstract drafting).
* Freeze split manifests under `data/derived/` (or later `data/processed/`) before any student training.

## Non-goals
* Continued pretraining on raw patent dumps (domain CPT) unless added as a separate ablation.
* RLHF / preference optimization in the first pass — SFT only unless human preference data is collected later.
* Mixing scrape or classification pipelines into the trainer; this stage consumes frozen JSONL only.
