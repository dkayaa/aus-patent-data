# Comparison and metrics
---

## Objective
Compare fine-tuned students against each other and against baselines on a frozen held-out test set, using a consistent metric suite that matches each task’s success criteria (classification fidelity, drafting fidelity, extractive QA, and optional human / LLM judge quality).

## Protocol
1. Load the frozen test split (no train leakage; same prompts for every system).
2. Generate one prediction per example under fixed decoding settings.
3. Score with the automatic metrics below; optionally score a stratified subsample with LLM-as-a-judge and/or human raters (reuse rubrics from [`02-dataset-validation`](../02-dataset-validation/)).
4. Report **per-task** tables (primary) and an optional multi-task average only when the multi-task student is the headline model.
5. Prefer paired comparisons on the same items (e.g. win rates, bootstrap CIs) over bare point estimates when claiming improvement.

## Shared reporting conventions
* **Primary split:** test only for final numbers; validation only for model selection / early stopping.
* **Decode once** at temperature 0 (or low) for automatic metrics; note if a second temperature is used for diversity ablations.
* Truncation / max-new-tokens must not silently empty outputs; count empty generations as failures.
* Where gold text is long, apply the same tokenization and length caps used in Mode 1 validation for ROUGE / embedding scores.

---

## Automatic metrics by task

### IPC reasoning & classification

For the **classification-only vs classification + justification** SFT contrast (label accuracy as the discovery metric), see the related-work note [`related_works/label-vs-rationale-sft.md`](../../related_works/label-vs-rationale-sft.md).

| Metric | What it measures | Notes |
|--------|------------------|--------|
| **Exact IPC accuracy** | Predicted `Classification` code == gold `primary_ipc` (normalized) | Strict headline: requires the SFT two-line schema |
| **Lenient exact IPC accuracy** | Gold `primary_ipc` appears anywhere in the output (normalized; spaces in symbols ignored) | Format-robust; markdown / examiner memos still count. Report next to strict exact and format-valid |
| **Hierarchical IPC accuracy** | Match at section / class / subclass / group | Softer view when full symbol is hard; still uses the parsed `Classification` line |
| **Justification ROUGE-L F1** | Lexical overlap of justification body vs gold justification (and optionally vs claims) | Complements code accuracy; not sufficient alone |
| **Justification semantic cosine** | Nomic Embed cosine vs gold / vs claims | Same family as Mode 1 semantic checks |
| **Format validity rate** | Output parses as `Classification` + `Justification` | Surface errors separate from content errors |

PatentBERT-style baselines: report IPC/subclass accuracy only; mark generative metrics N/A.

### Abstract drafting

| Metric | What it measures | Notes |
|--------|------------------|--------|
| **ROUGE-1 / ROUGE-2 / ROUGE-L** | n-gram / LCS overlap vs gold abstract | Standard summarization suite |
| **BLEU** | Precision-oriented n-gram overlap vs gold | Secondary; sensitive to length |
| **BERTScore (F1)** | Token-level contextual similarity vs gold | Captures paraphrase better than ROUGE |
| **Semantic cosine** | Sentence embedding similarity vs gold abstract (and vs claims for faithfulness) | Align with Mode 1 pair definitions |
| **Length / compression stats** | `|pred|`, `|pred|/|claims|` vs gold distribution | Detect verbosity collapse or copy-all |

### Machine reading comprehension (MRC) — retired

MRC is **out of the active eval/SFT path**. Legacy metrics below remain for historical notes only; do not require new MRC generation for paper runs.

| Metric | What it measures | Notes |
|--------|------------------|--------|
| **Exact match (EM)** | Normalized string equality with gold answer | Strict |
| **Token-F1** | Token overlap F1 vs gold | Primary soft metric (SQuAD-style) |
| **Answer containment** | Pred (or gold) span appears in claims | Faithfulness / extractiveness |
| **Has-answer rate** | Non-empty, on-topic replies | Separates refusals / empties |

---

## Cross-cutting qualitative metrics

### LLM-as-a-judge (sample)
Pointwise 1–5 (+ pass/fail) on model **predictions** using task rubrics adapted from Mode 2 (judge sees instruction, input, and model output; optionally gold for reference-based grading when comparing systems). Report mean score, pass rate, and top failure tags **by system × task**. Mitigate position and verbosity bias (fixed rubric, temperature 0, no generator identity in the prompt).

### Human evaluation (sample)
Blind pairwise or Likert ratings on a stratified subsample (correctness, faithfulness, usefulness). Report agreement (e.g. Cohen’s κ) and head-to-head win/tie/lose rates for student vs strongest open baseline.

### Efficiency (secondary)
For open models only: trainable parameter count, wall-clock train time, tokens/sec at decode, peak VRAM. Not a quality substitute; useful when two models are close on task metrics.

---

## How to read the comparison
* **Did SFT help?** Student vs untuned base on the same automatic metrics (paired).
* **Is the gain competitive?** Student vs few-shot base and vs PatentBERT (IPC codes only).
* **Ceiling / reference?** Frontier API on the same sample (clearly labeled as unmatched compute).
* **Quality beyond n-grams?** LLM judge + human on abstract drafting and IPC justifications, where ROUGE can mislead.

## Non-goals
* Optimizing only for ROUGE on drafting without a faithfulness or human check.
* Averaging IPC accuracy with ROUGE into a single unweighted “overall score” without stating the aggregation rule.
