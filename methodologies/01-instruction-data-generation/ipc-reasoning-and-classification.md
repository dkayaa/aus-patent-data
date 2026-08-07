# Task 1: IPC Reasoning & Classification
---

## Objective
Generate a technically grounded justification for the true IPC code assigned to a patent. This tests a model's ability to categorize technical text according to rigid IPC definitions and perform zero-shot generalization on high-cardinality taxonomies.

## Data Schema
*   **Input:** Abstract + Claims Text
*   **Target:** IPC Code + Synthetic WIPO-grounded Justification

## Instruction pool
Before pairing patents, build a **cached pool of diverse instruction phrasings** for this task (same idea as abstract drafting):

1. Send a meta-prompt to the generator LLM asking for N professional variants (e.g. examiner memo, attorney note, brief rationale) that all ask for an IPC justification from abstract + claims.
2. Parse the JSON list of strings and append until the pool reaches the configured size (default 40).
3. Persist under `data/derived/instruction_generation/_pools/ipc_reasoning.json`.
4. For each training example, **randomly sample one** pool string as the Alpaca `instruction` field.

This is instruction-phrasing diversification (Self-Instruct-style expansion of wordings for a *fixed* task), not full Evol-Instruct (no iterative seed evolution). The pool avoids overfitting a single fixed template at fine-tune time.

## Generation Workflow
1.  **Extract:** Pull `abstractText`, `claimsText`, and the `primary_ipc` code from the IP Australia JSON payload.
2.  **Lookup:** Query the WIPO IPC catalog for the official definition of `primary_ipc`. **Skip** patents whose code has no `definition_statement` (title-only places).
3.  **Sample instruction:** Draw one string from the instruction pool (build/load pool first if missing).
4.  **Prompt the Generator LLM (justification only):** The teacher sees abstract, claims, IPC code, title, and WIPO definition — these grounding fields are *not* copied into the student `input`.
    *   **System Prompt:** "You are an expert Australian Patent Examiner."
    *   **User Prompt:** Ask for a 2-sentence technical justification mapping claims to the definition.
5.  **Assemble:**
    *   `instruction` = sampled pool string
    *   `input` = abstract + claims only
    *   `output` = `Classification: [Code]\nJustification: [LLM response]`
