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

## Generation Workflow
1.  **Extract:** Pull `abstractText`, `claimsText`, and the `primary_ipc` code from the IP Australia JSON payload.
2.  **Lookup:** Query the WIPO IPC catalog for the official definition of `primary_ipc`. **Skip** patents whose code has no `definition_statement` (title-only places).
3.  **Sample instruction:** Draw one string from the instruction pool (build/load pool first if missing).
4.  **Prompt the Generator LLM (justification only):** The teacher sees abstract, claims, IPC code, title, and WIPO definition — these grounding fields are *not* copied into the student `input`.
    *   **System Prompt:** "You are an expert Australian Patent Examiner."
    *   **User Prompt:** Treat the assigned code as gold. Short prose (~120–220 words) mapping claimed subject matter to the **provided** WIPO definition. No invented scope, no other IPC codes. Do not prescribe a sentence recipe or a stock opening; structure may vary.
5.  **Assemble:**
    *   `instruction` = sampled pool string
    *   `input` = abstract + claims only
    *   `output` = `Classification: [Code]\nJustification: [LLM response]`
