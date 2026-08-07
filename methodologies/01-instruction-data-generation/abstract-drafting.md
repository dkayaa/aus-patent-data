# Task 2: Abstract Drafting (Summarization)
---

## Objective
Generate diverse instructions to map "Gold Standard" claims to their official abstracts. This evaluates a model's capacity for constraint-preserving compression, ensuring it can synthesize dense technical boundaries without omitting critical parameters or hallucinating.

## Data Schema
*   **Input:** Full Claims Text (or primary independent claims)
*   **Target:** Official Abstract

## Instruction pool
Build a **cached pool of diverse instruction phrasings** so the fine-tune does not see one fixed prompt for every example:

1. Send a meta-prompt to the generator LLM asking for N diverse, professional instructions that ask a patent attorney to summarize claims into an abstract (vary length and tone).
2. Parse the JSON list of strings; repeat in batches until the pool reaches the configured size (default 40).
3. Persist under `data/interim/instruction_generation/_pools/abstract_drafting.json`.
4. For each patent example, **randomly sample one** pool string as `instruction`.

This diversifies instruction wording for a fixed task (Self-Instruct-style). It is not full Evol-Instruct (no iterative evolution of a seed set). The LLM is **not** used to write the abstract — that remains the official gold abstract.

### Meta-prompt (pool builder)
> I am building an instruction-tuning dataset. Generate 5 diverse, professional instructions asking a patent attorney to summarize a set of claims into an abstract. Vary the length and tone (e.g., 'Draft an abstract...', 'Distill the following claims...', 'Provide a technical summary...'). Output only a JSON list of strings.

## Generation Workflow
1.  **Extract:** Pull `claimsText` and `abstractText` from the IP Australia JSON payload.
2.  **Load / build instruction pool** (see above).
3.  **Assemble:** Sample one pool instruction. Pair it with `claimsText` as `input` and the official `abstractText` as `output`.
