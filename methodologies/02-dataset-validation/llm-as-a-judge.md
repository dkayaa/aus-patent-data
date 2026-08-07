# Mode 2: LLM-as-a-judge
---

## Objective
Use a frontier model as an automated grader over seed instruction examples for all three tasks (IPC reasoning, abstract drafting, MRC), producing scalable quality scores and short rationales before human audit.

## Citation
This mode follows the **LLM-as-a-judge** paradigm established by:

> Lianmin Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, NeurIPS 2023 Datasets and Benchmarks.  
> arXiv: [2306.05685](https://arxiv.org/abs/2306.05685)

They show strong LLM judges can reach ~human–human agreement levels on open-ended preference/quality judgments, with known biases (position, verbosity, self-enhancement) that we mitigate below.

Optional complementary rubric style: Liu et al., *G-Eval* (EMNLP 2023) — step-by-step criteria then score — useful when we want dimensioned rubrics rather than a single preference vote.

## Scope
**Sample-based by default** (not a full-corpus judge pass): Mode 1 survivors at `data/interim/instruction_generation_validation/<task>/passed/`, default `sample_size: 50` per task (`--limit` overrides). Prefer judging after programmatic filters (Mode 1).

## Run

```bash
set -a && source .env && set +a
.venv/bin/python scripts/judge_instruction_data.py --task ipc_reasoning
.venv/bin/python scripts/judge_instruction_data.py --all --limit 50
```

Implementation: `dataset-validation/` (`config/llm_judge.yaml`, `src/run_llm_judge.py`, `src/judge_prompts.py`). Outputs: `<task>/llm_judge/{passed,rejected,report.json,done_ids.txt}`.

## Judge setup
* **Judge model:** frontier via OpenRouter (e.g. `anthropic/claude-sonnet-4.6` or stronger), **different from or at least not weaker than** the generator when possible. Temperature **0**.
* **Protocol:** single-answer pointwise scoring (not pairwise MT-Bench battles), with an explicit rubric per task — closer to G-Eval-style rating than Chatbot Arena win/lose.
* **Output schema (JSON):** `{ "score": 1-5, "pass": bool, "rationale": "...", "failure_tags": [] }`

## Task rubrics (summary)

| Task | High score means |
|------|------------------|
| **ipc_reasoning** | Treat `primary_ipc` / Classification as gold — do not re-classify. Justification maps claims to that fixed place; no invented codes; not boilerplate |
| **abstract_drafting** | Gold abstract is a fair compression of claims (judge checks *pair* coherence: instruction asks for abstract, input=claims, output=abstract); no obvious claim–abstract mismatch |
| **mrc** | Question is answerable **only** from the claims; answer is extractive/verbatim-supported; no hallucination |

For abstract drafting the “output” is gold patent text — the judge mainly flags **misaligned triples** (bad instruction, truncated input, wrong field pairing), not “is this abstract well written by an LLM.”

## Bias mitigations (from Zheng et al.)
* Fixed rubric + forced JSON (reduce verbosity gaming).
* Do not reveal generator model identity in the judge prompt.
* Optional: second judge pass or temperature 0 for stability.
* Calibrate thresholds against the human slice (Mode 3).

## Outputs
* Per-example judge JSONL alongside source ids
* Aggregate: mean score, pass rate by task, top failure tags
* Optional quarantine of `pass: false`

## Implementation note
Runnable code lives under `dataset-validation/` and reuses `instruction-generation/src/llm.py` (`LLMClient`, `chat_json`, OpenRouter). Rubrics in `judge_prompts.py` strip generator `meta.model` / provider from the judge-visible payload; IPC reasoning keeps `primary_ipc` / `ipc_title`.
