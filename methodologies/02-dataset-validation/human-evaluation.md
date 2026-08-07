# Mode 3: Human evaluation
---

## Objective
Obtain a small, trustworthy labeled slice to (a) estimate true seed-data quality, (b) calibrate LLM-as-a-judge thresholds, and (c) report inter-annotator + human–judge agreement for the paper.

## Best-guess protocol
Full double-annotation of thousands of long patent examples is unrealistic. Prefer a **stratified audit** plus clear rubrics.

### Sample
* **Size:** ~100–200 examples total, stratified across the three tasks (e.g. ~35–65 each), drawn from the programmatically filtered set.
* **Stratify further within task** when possible: IPC section (IPC reasoning), abstract length buckets, MRC answer type (number / composition / structure).
* **Annotators:** ideally 2 people with patent/technical literacy (examiner trainee, patent attorney, or ML researcher + domain consult). One primary + one secondary for agreement.

### Interface (lightweight)
Spreadsheet or simple Label Studio / Argilla project with columns:

| Field | Content |
|-------|---------|
| `application_number`, `task` | ids |
| shown `instruction`, `input`, `output` | full text (truncate UI with expand) |
| for IPC reasoning: show `meta.primary_ipc` + official IPC title/definition from catalog | grounding for the rater |
| ratings | see below |
| free-text note | optional |

Do **not** show LLM-judge scores to humans on the first pass (blind).

### Rating dimensions (1–5 Likert + binary)
Shared:
* **Overall accept?** (`yes` / `no` / `fix`) — primary paper metric for “usable SFT row.”

Task-specific (1–5):
* **ipc_reasoning:** *IPC plausibility*, *justification faithfulness to claims*, *no hallucinated facts/codes*
* **abstract_drafting:** *coverage of independent claim*, *no contradiction with claims*, *appropriate abstract style*
* **mrc:** *answer supported by span in claims*, *question specificity*, *no speculation*

### Process
1. Pilot 10 examples → refine rubric wording.
2. Double-annotate ~30% of the audit set; single-annotate the rest.
3. Compute Cohen’s κ or % agreement on Accept; discuss disagreements.
4. Compare human Accept vs LLM-judge `pass` (Mode 2) → set operating threshold / report confusion matrix.
5. Freeze the audited labels as `data/interim/instruction_generation_validation/human_audit.jsonl`.

### What to report in the paper
* Human accept rate by task
* Inter-annotator agreement
* Human ↔ LLM-judge agreement (echoing Zheng et al.’s validation style, at our domain scale)
* Qualitative failure modes (bullet list from notes)

## Non-goals
* Crowdsourcing non-experts on full claim sets (too error-prone without heavy screening).
* Replacing Mode 1/2 — human audit is calibration and credibility, not the only filter.
