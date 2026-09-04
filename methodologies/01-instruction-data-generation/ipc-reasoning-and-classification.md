# Task 1: IPC Reasoning & Classification
---

## Objective
Generate Alpaca-style SFT rows where the student must **assign** the office IPC code and produce a short WIPO-grounded justification. The office `primary_ipc` is gold in the target; the justification prose is synthetic. This tests high-cardinality classification from technical text plus constrained explanatory writing.

## Data Schema
*   **Input:** Abstract + Claims Text
*   **Target:** IPC Code + Synthetic WIPO-grounded Justification

## Instruction pool
Before pairing patents, build a **cached pool of diverse instruction phrasings** for this task (same idea as abstract drafting):

1. Send a meta-prompt to the generator LLM asking for N professional variants (examiner note, attorney file note, brief rationale, classification log) that all ask the model to **assign an IPC code and write a short justification** mapping claim features to that place.
2. Parse the JSON list of strings and append until the pool reaches the configured size (default 40).
3. Persist under `data/derived/instruction_generation/_pools/ipc_reasoning.json`.
4. For each training example, **randomly sample one** pool string as the Alpaca `instruction` field.

Student-facing pool wordings must match the eval task (assign + justify from abstract/claims; the gold code is **not** in `input`). Do **not** ask for multi-section examiner memoranda, hierarchical walk-downs, or multi-code comparisons. The *teacher* prompt that synthesizes the justification still treats the office `primary_ipc` as gold — that is data construction only. Rebuild with `--rebuild-pool` (requires a single `--task`). If `--limit` is omitted, generation is skipped (`limit=0`) so only the pool is rewritten. Pass `--limit N` to rebuild and then generate N records.

This is instruction-phrasing diversification (Self-Instruct-style expansion of wordings for a *fixed* task), not full Evol-Instruct (no iterative seed evolution). The pool avoids overfitting a single fixed template at fine-tune time.

## Application sampling (class imbalance)

The cleaned AU register dump is heavily skewed toward a small set of frequent IPC symbols (especially A61 / C12 pharma–biotech). Walking `patent_search_clean` in shard order and taking the first N eligible apps **amplifies** that head: a 1k–10k ipc_reasoning set would otherwise be dominated by a handful of codes (e.g. `A61P35/00`), so exact-IPC metrics and SFT would mostly reflect common labels rather than broad classification skill.

**What we do:** Before generation at scale, freeze an application list with a **per-`primary_ipc` cap**:

* Eligible apps: claims + abstract + `primary_ipc` present in the WIPO catalog (same eligibility as ipc_reasoning generation).
* Shuffle with a fixed seed (default 42), then greedily accept apps until `--target` (e.g. 10 000), refusing any symbol that already has `⌊target × max_per_symbol_frac⌋` apps (default **1%** → cap 100 at target 10k).
* No section-level quota: section A may still be the plurality; we only stop a **single full symbol** from eating the set.
* Artifact: `scripts/sample_ipc_apps.py` → `data/derived/instruction_generation/_samples/<name>.txt` (+ `.manifest.json`). Generation uses `--only-ids` on that file (still resumes via `done_ids.txt`).

**Why not section caps / equal-per-code:** Equalizing all ~58k symbols is impossible; hard section quotas fight the natural register and scarce sections. A 1% symbol cap is a light, auditable rebalance that preserves AU realism while making head-vs-tail evaluation meaningful. Report exact IPC **overall**, and prefer also **by section** and **head vs tail** at scoring time (eval split remains temporal + app-holdout; it does not re-stratify).

## Generation Workflow
1.  **Sample apps (scaled runs):** Optionally build the capped id list above; otherwise shard order applies.
2.  **Extract:** Pull `abstractText`, `claimsText`, and the `primary_ipc` code from the IP Australia JSON payload.
3.  **Lookup:** Query the WIPO IPC catalog for the official definition of `primary_ipc`. **Skip** patents whose code has no `definition_statement` (title-only places).
4.  **Sample instruction:** Draw one string from the instruction pool (build/load pool first if missing).
5.  **Prompt the Generator LLM (justification only):** The teacher sees abstract, claims, IPC code, title, and WIPO definition — these grounding fields are *not* copied into the student `input`.
    *   **System Prompt:** "You are an expert Australian Patent Examiner."
    *   **User Prompt:** Treat the assigned code as gold. Short prose (~120–220 words) mapping claimed subject matter to the **provided** WIPO definition. No invented scope, no other IPC codes. Do not prescribe a sentence recipe or a stock opening; structure may vary.
6.  **Assemble:**
    *   `instruction` = sampled pool string
    *   `input` = abstract + claims only
    *   `output` = `Classification: [Code]\nJustification: [LLM response]`
