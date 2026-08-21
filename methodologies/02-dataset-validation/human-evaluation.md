# Mode 3: Human evaluation
---

## Objective
Obtain a small, trustworthy labeled slice to (a) estimate true seed-data quality, (b) calibrate LLM-as-a-judge thresholds, and (c) report inter-annotator + human–judge agreement for the paper.

Use the **same Accept definition as Mode 2** (`methodologies/02-dataset-validation/llm-as-a-judge.md`). Human↔judge disagreement should measure rater/model noise, not two different rubrics.

## Best-guess protocol
Full double-annotation of thousands of long patent examples is unrealistic. Prefer a **stratified audit** plus clear rubrics.

### Sample
* **Size:** ~100–200 examples total, stratified across the three tasks (e.g. ~35–65 each). Prefer the **pinned Mode 2 ID list** (`llm_judge/done_ids.txt`) so every audited row already has a judge grade.
* **Stratify further within task** when possible: IPC section (IPC reasoning), abstract length buckets, MRC answer type (number / composition / structure).
* **Annotators:** ideally 2 people with patent/technical literacy (examiner trainee, patent attorney, or ML researcher + domain consult). One primary + one secondary for agreement.

### Interface (lightweight)
Blind CSV from `scripts/calibrate_llm_judge.py --export-csv` (no judge scores), or a spreadsheet with:

| Field | Content |
|-------|---------|
| `application_number`, `task` | ids |
| shown `instruction`, `input`, `output` | full text (truncate UI with expand) |
| for IPC reasoning: show `primary_ipc` + official IPC title/definition from catalog | grounding for the rater |
| ratings | see below |
| free-text note | optional |

Do **not** show LLM-judge scores to humans on the first pass (blind).

### Rating dimensions (binary Accept + optional 1–5)

Shared (primary paper metric):
* **Overall accept?** (`yes` / `no` / `fix`) — “usable SFT row,” matching Mode 2 `pass`. Map `yes` → accept; `no` and `fix` → reject when computing agreement.

Task-specific (optional 1–5, same fail/in-scope rules as Mode 2):
* **ipc_reasoning:** justification faithfulness to claims given the **fixed** office `primary_ipc`; no invented codes; definitions consistent with the provided WIPO catalog text. Do **not** reject because a different IPC would fit better.
* **abstract_drafting:** pair coherence (instruction / claims / gold abstract are the same invention; not corrupted; not claim-1 pasted as the abstract). Do **not** reject brief/generic official abstracts that still match the invention, and do not score writing style.
* **mrc:** question answerable from claims; answer span-supported; no speculation.

### Process
1. Pilot 10 examples → refine rubric wording (keep Mode 2 Accept language).
2. Double-annotate ~30% of the audit set; single-annotate the rest.
3. Compute Cohen’s κ or % agreement on Accept; discuss disagreements.
4. Compare human Accept vs LLM-judge `pass` (Mode 2) with `scripts/calibrate_llm_judge.py` → confusion matrix and `pass_score_min` sweep {3,4,5}.
5. Freeze the audited labels as `data/derived/instruction_generation_validation/{model_slug}/human_audit.jsonl`.

`human_audit.jsonl` rows:

```json
{"application_number": "2024396373", "task": "abstract_drafting", "accept": "yes", "score": 4, "note": ""}
```

`accept` is required (`yes` / `no` / `fix`). `score` and `note` are optional.

### What to report in the paper
* Human accept rate by task
* Inter-annotator agreement
* Human ↔ LLM-judge agreement (echoing Zheng et al.’s validation style, at our domain scale)
* Chosen `pass_score_min` after the sweep
* Qualitative failure modes (bullet list from notes)

## Non-goals
* Crowdsourcing non-experts on full claim sets (too error-prone without heavy screening).
* Replacing Mode 1/2 — human audit is calibration and credibility, not the only filter.
