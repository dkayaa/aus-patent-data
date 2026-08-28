# Mode 2: LLM-as-a-judge
---

## Objective
Use a frontier model as an automated grader over seed instruction examples for all three tasks (IPC reasoning, abstract drafting, MRC), producing scalable quality scores and short rationales before human audit.

## Scope
**Sample-based by default** (not a 500k full-pass): Mode 1 survivors at `data/derived/instruction_generation_validation/{model_slug}/<task>/passed/`, default `sample_size: 50` per task (`--limit` overrides). Prefer judging after programmatic filters (Mode 1). Pin a frozen ID list with `--ids-file` when re-grading a calibration sample.

Optional **cheap cascade screen** (Claude Haiku 4.5, full Mode 1 `passed/`): `scripts/cheap_judge_instruction_data.py` → `{task}/cheap_judge/`. Same rubric. Compare to this Mode 2 sample with `scripts/calibrate_llm_judge.py` (`cheap_judge_cascade.json`). Not a training filter. Archived GPT-4o-mini IPC screen: `{task}/cheap_judge_gpt4o_mini/`.

## Run

```bash
set -a && source .env && set +a
.venv/bin/python scripts/judge_instruction_data.py --task ipc_reasoning
.venv/bin/python scripts/judge_instruction_data.py --all --limit 50
.venv/bin/python scripts/judge_instruction_data.py --all --limit 50 --workers 12
.venv/bin/python scripts/judge_instruction_data.py --task ipc_reasoning \
  --generator meta-llama/llama-3.3-70b-instruct \
  --ids-file data/derived/instruction_generation_validation/meta-llama-llama-3.3-70b-instruct/ipc_reasoning/llm_judge/done_ids.txt
.venv/bin/python scripts/cheap_judge_instruction_data.py --task ipc_reasoning \
  --generator meta-llama/llama-3.3-70b-instruct \
  --ids-file data/derived/instruction_generation_validation/meta-llama-llama-3.3-70b-instruct/ipc_reasoning/llm_judge/done_ids.txt
```

Implementation: `dataset-validation/` (`config/llm_judge.yaml`, `config/cheap_judge.yaml`, `src/run_llm_judge.py`, `src/judge_prompts.py`). Mode 2 outputs: `{model_slug}/<task>/llm_judge/{passed,rejected,report.json,done_ids.txt}`. Cheap judge: `{task}/cheap_judge/` (never overwrites Mode 2).

Distributions and a later human↔judge threshold sweep: `scripts/calibrate_llm_judge.py`.

## Judge setup
* **Judge model:** frontier via OpenRouter (e.g. `anthropic/claude-sonnet-4.6` or stronger), **different from or at least not weaker than** the generator when possible. Temperature **0**. Record the actual model id in `report.json`.
* **Protocol:** single-answer pointwise scoring with an explicit rubric per task.
* **Model JSON:** `{ "rationale": "...", "score": 1-5, "failure_tags": [] }` — chain of thought in `rationale` (check each criterion, quote spans), then `score`. No `pass` field from the model.
* **`pass`:** derived in code as `score >= pass_score_min` (default 4). Pre-human operating point; Mode 3 may change the threshold.

## Accept definition (shared with Mode 3)

**Usable SFT row?** Binary `pass` iff `score >= pass_score_min`.

| Task | High score / Accept means | Fail (in scope) | Out of scope (do not fail) |
|------|---------------------------|-----------------|----------------------------|
| **ipc_reasoning** | Office `primary_ipc` is gold. Justification maps claims to that **fixed** place; no invented codes; definitions match the WIPO catalog text in the payload | Unfaithful to claims; fabricated definition (vs catalog); boilerplate; invented IPC token; scrape corruption | A different IPC would be “better”; obsolete-code arguments; re-classification |
| **abstract_drafting** | Gold abstract is a matching pair for the claims (instruction asks for an abstract; input=claims; output=that invention’s abstract) | Topic mismatch; swapped/truncated/corrupted fields; instruction mismatch; claim-1 pasted as the abstract | Brief/generic official abstracts that still match the invention; USPTO style; missing dependent-claim detail |
| **mrc** | Question answerable from claims; answer span-supported; no speculation | Unanswerable; unsupported / hallucinated answer; speculation; swapped fields | Stylistic phrasing of an otherwise extractive answer |

IPC payload includes WIPO `ipc_title` + `definition_statement` from `data/ipc-codes/`. “Fabricated definition” means contradicting that catalog text, not the judge’s parametric memory.

Failure tags are a **closed set** per task (unknown tags → `other`). IPC tags `wrong_ipc`, `obsolete_ipc_code`, `better_ipc_exists`, `suboptimal_ipc_selection`, and `classification_mismatch` are stripped if they leak.

## Bias mitigations
* Fixed rubric + forced JSON (reduce verbosity gaming); CoT evaluation steps before the score field.
* Do not reveal generator model identity in the judge prompt.
* Temperature 0 for stability.
* Calibrate `pass_score_min` against the human slice (Mode 3) via `scripts/calibrate_llm_judge.py`.

## Outputs
* Per-example judge JSONL alongside source ids (`meta.llm_judge`)
* Aggregate: mean score, pass rate by task, closed-set failure tags
* Optional quarantine of `pass: false` (sample only — not a training filter)

## Implementation note
Runnable code lives under `dataset-validation/` and reuses `instruction-generation/src/llm.py` (`LLMClient`, `chat_json`, OpenRouter) and `ipc_lookup.py` for WIPO grounding. Rubrics in `judge_prompts.py` strip generator `meta.model` / provider from the judge-visible payload.
