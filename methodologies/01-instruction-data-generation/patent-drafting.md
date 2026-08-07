# Task 3: Patent Drafting (Generation)
---

## Objective
Generate diverse instructions to map official abstracts to their corresponding primary independent claims. This tests the model's ability to draft legally robust, structured claims based on high-level summaries.

## Data Schema
*   **Input:** Official Abstract
*   **Target:** Claim 1 (Primary independent claim)

## Instruction pool
Build a **cached pool of diverse instruction phrasings** so the fine-tune does not see one fixed prompt for every example:

1. Send a meta-prompt to the generator LLM asking for N diverse instructions that ask a drafting assistant to write a first independent claim from an abstract.
2. Parse the JSON list of strings; repeat in batches until the pool reaches the configured size (default 40).
3. Persist under `data/interim/instruction_generation/_pools/patent_drafting.json`.
4. For each patent example, **randomly sample one** pool string as `instruction`.

This diversifies instruction wording for a fixed task (Self-Instruct-style). It is not full Evol-Instruct (no iterative evolution of a seed set). The LLM is **not** used to write claim 1 — that remains the official gold claim.

### Meta-prompt (pool builder)
> Generate 5 diverse instructions asking a patent drafting assistant to write a first independent method/apparatus claim based on an abstract. (e.g., 'Draft claim 1...', 'Based on this abstract, write a legally robust independent claim...'). Output only a JSON list of strings.

## Generation Workflow
1.  **Extract:** Pull `abstractText` and the text of Claim 1 from the IP Australia JSON payload.
2.  **Load / build instruction pool** (see above).
3.  **Assemble:** Sample one pool instruction. Pair it with `abstractText` as `input` and official Claim 1 as `output`.
