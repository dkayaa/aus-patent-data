# Baselining
---

## Objective
Establish reference performance on the same held-out test prompts used for fine-tuned students, so gains are attributable to SFT data rather than to an easy eval set or a weak comparison point.

## Baseline classes

### 1. Untuned student base
Same open-source checkpoint as the SFT student, **without** our adapters / full fine-tune. Run in the same decoding settings (temperature, max tokens, stop rules). This is the primary “did our data help?” control.

### 2. Zero-shot / few-shot prompting (no weight updates)
* **Zero-shot:** task instruction + patent `input` only.
* **Few-shot:** k exemplars drawn from the **train** split only (never test), same task; keep k small (e.g. 1–3) and truncate long claims so context fits.
Apply to both the student base and any API baselines that support long context.

### 3. Strong API / frontier references (optional but useful)
One frontier chat model via OpenRouter (or equivalent), evaluated only as an **upper reference**, not as a trained student. Same prompts and decoding constraints where possible. Do not treat API scores as a fair compute-matched baseline.

### 4. Task-specific published or domain baselines (where they exist)
Use when a clear, citable system maps onto a task:
* **legal_reasoning / IPC-like labeling:** PatentBERT (or similar CPC/IPC classifiers already used in `classification/`) as a **classification-only** baseline on primary IPC / subclass — not a free-text justification baseline.
* **MRC-style extractive QA:** report classic extractive metrics against any off-the-shelf open MRC model only if inputs can be aligned fairly; otherwise omit rather than force a mismatch.
* **Drafting tasks:** typically no single public AU-claims↔abstract model; rely on untuned + frontier references plus automatic metrics.

Document each baseline’s license, access date, and exact model ID.

## Evaluation conditions (shared with students)
* Identical test manifests (`application_number` + `task`).
* Identical generation hyperparameters for generative baselines of the same size class.
* No access to gold `output` except for few-shot exemplars from train.
* For legal reasoning, baselines that only emit a code are scored on **code accuracy**; free-text justification metrics apply only when a justification is produced.

## What to report per baseline
* Model / method name and size (or API tier)
* Prompting regime (zero-shot / few-shot / SFT)
* Whether LoRA or full weights (N/A for APIs)
* Aggregate metrics from [comparison.md](comparison.md), by task

## Non-goals
* Training proprietary closed models on our JSONL.
* Claiming SOTA against systems that used different patents, languages, or IPC editions without noting the mismatch.
